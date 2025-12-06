"""
PredRNN++ & UNet++ Combined Model for Rainfall Prediction

Based on official implementations:
- PredRNN++: https://github.com/thuml/predrnn-pytorch
- UNet++: https://github.com/MrGiovanni/UNetPlusPlus

Architecture:
    Input (B, 30, 24, 35, 35) 
    → PredRNN++ → (B, 14, 24, 35, 35)
    → UNet++ → (B, 14, 1, 35, 35)
    
Training flow: Forward both models → Single backward pass
"""

import torch
import torch.nn as nn


# ==================== PredRNN++ COMPONENTS ====================


class SpatioTemporalLSTMCell(nn.Module):
    """
    Spatio-Temporal LSTM Cell - EXACT Official Implementation
    
    From: https://github.com/thuml/predrnn-pytorch/blob/master/core/layers/SpatioTemporalLSTMCell.py
    
    Key features:
    - Additive gate computation: i_t = sigmoid(i_x + i_h) (NOT concat)
    - Forget bias: 1.0
    - Dual memory: temporal (C) and spatial (M)
    - Output gate uses both memories
    
    NOTE: Official uses only 'width' parameter (assumes square input).
    """

    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 7,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 7, width, width]),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 4, width, width]),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden * 3, width, width]),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    num_hidden * 2,
                    num_hidden,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([num_hidden, width, width]),
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 7,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    num_hidden * 2,
                    num_hidden,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False,
                ),
            )
        self.conv_last = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x_t, h_t, c_t, m_t):
        """
        Forward pass through ST-LSTM cell
        
        Args:
            x_t: Input at time t (B, C, H, W)
            h_t: Hidden state (B, num_hidden, H, W)
            c_t: Temporal memory (B, num_hidden, H, W)
            m_t: Spatial memory (B, num_hidden, H, W)
            
        Returns:
            h_new: New hidden state
            c_new: New temporal memory
            m_new: New spatial memory
        """
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        
        # Split into gates
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # Temporal memory gates (additive)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        # Spatial memory gates (additive)
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        # Output gate (uses both memories)
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class PredRNNPlusPlus(nn.Module):
    """
    PredRNN++ Model - Official Architecture
    
    Based on: https://github.com/thuml/predrnn-pytorch/blob/master/core/models/predrnn.py
    
    Key features:
    1. Stack of SpatioTemporalLSTMCells
    2. Zigzag memory flow: M flows from last layer of t-1 to first layer of t
    3. Additive gate computation (official implementation)
    
    Input: (B, T_in, C, H, W)
    Output: (B, T_out, C, H, W)
    """

    def __init__(
        self,
        num_layers,
        num_hidden,
        in_channel,
        width,
        filter_size=5,
        stride=1,
        layer_norm=True,
        output_channel=None,
    ):
        super(PredRNNPlusPlus, self).__init__()

        self.num_layers = num_layers
        self.num_hidden = num_hidden  # List of hidden dims per layer
        self.in_channel = in_channel
        self.width = width
        self.output_channel = output_channel if output_channel else in_channel

        # Build cell stack (official structure)
        cell_list = []
        for i in range(num_layers):
            in_ch = in_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel=in_ch,
                    num_hidden=num_hidden[i],
                    width=width,
                    filter_size=filter_size,
                    stride=stride,
                    layer_norm=layer_norm,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Output projection
        self.conv_last = nn.Conv2d(
            num_hidden[-1],
            self.output_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, frames, future_length=14, teacher_forcing_ratio=0.0):
        """
        Forward pass through PredRNN++
        
        Args:
            frames: Input tensor (B, T_in, C, H, W)
            future_length: Number of future frames to predict
            teacher_forcing_ratio: Probability of using ground truth (0 for inference)
            
        Returns:
            predictions: (B, future_length, C, H, W)
        """
        B, T_in, C, H, W = frames.shape
        device = frames.device

        # Initialize hidden states
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros(B, self.num_hidden[i], H, W, device=device)
            h_t.append(zeros)
            c_t.append(zeros)

        # Initialize spatiotemporal memory (shared, zigzag flow)
        memory = torch.zeros(B, self.num_hidden[0], H, W, device=device)

        # ==================== Encoding Phase ====================
        # Process input sequence to build up hidden states
        for t in range(T_in):
            net = frames[:, t]  # (B, C, H, W)

            # Forward through all layers
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                # Input to layer i is output of layer i-1
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )

        # ==================== Prediction Phase ====================
        # Generate future frames autoregressively
        predictions = []

        # First prediction input is projection of last hidden state
        x_gen = self.conv_last(h_t[-1])  # (B, C, H, W)

        for t in range(future_length):
            # Forward through all layers
            h_t[0], c_t[0], memory = self.cell_list[0](x_gen, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )

            # Generate prediction for this timestep
            x_gen = self.conv_last(h_t[-1])
            predictions.append(x_gen)

        # Stack predictions: (B, T_out, C, H, W)
        predictions = torch.stack(predictions, dim=1)

        return predictions


# ==================== UNet++ COMPONENTS ====================


class ConvBlock(nn.Module):
    """
    Standard Convolution Block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    Used in UNet++ encoder and decoder paths.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlusCore(nn.Module):
    """
    UNet++ (Nested U-Net) - Official Architecture
    
    Based on: https://github.com/MrGiovanni/UNetPlusPlus
    
    Key features:
    1. Dense skip connections: X[i,j] = f(concat(X[i,0:j], Up(X[i+1,j-1])))
    2. Optional deep supervision
    3. Reduces semantic gap between encoder and decoder
    
    Single frame: Input (B, C, H, W) -> Output (B, out_ch, H, W)
    """

    def __init__(
        self,
        in_channels=24,
        out_channels=1,
        features=[32, 64, 128, 256],
        deep_supervision=False,
    ):
        super(UNetPlusPlusCore, self).__init__()

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)

        # ==================== Encoder (Backbone) ====================
        # X0,0
        self.conv0_0 = ConvBlock(in_channels, features[0])
        # X1,0
        self.conv1_0 = ConvBlock(features[0], features[1])
        # X2,0
        self.conv2_0 = ConvBlock(features[1], features[2])
        # X3,0
        self.conv3_0 = ConvBlock(features[2], features[3])
        # X4,0 (Bottleneck)
        self.conv4_0 = ConvBlock(features[3], features[3] * 2)

        # ==================== Decoder with Nested Skip Connections ====================
        # Upsampling layers
        self.up1_0 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)

        self.up1_2 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)

        self.up1_3 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)

        # Dense blocks - Column 1
        # X0,1: concat(X0,0, Up(X1,0))
        self.conv0_1 = ConvBlock(features[0] + features[0], features[0])
        # X1,1: concat(X1,0, Up(X2,0))
        self.conv1_1 = ConvBlock(features[1] + features[1], features[1])
        # X2,1: concat(X2,0, Up(X3,0))
        self.conv2_1 = ConvBlock(features[2] + features[2], features[2])
        # X3,1: concat(X3,0, Up(X4,0))
        self.conv3_1 = ConvBlock(features[3] + features[3], features[3])

        # Dense blocks - Column 2
        # X0,2: concat(X0,0, X0,1, Up(X1,1))
        self.conv0_2 = ConvBlock(features[0] * 2 + features[0], features[0])
        # X1,2: concat(X1,0, X1,1, Up(X2,1))
        self.conv1_2 = ConvBlock(features[1] * 2 + features[1], features[1])
        # X2,2: concat(X2,0, X2,1, Up(X3,1))
        self.conv2_2 = ConvBlock(features[2] * 2 + features[2], features[2])

        # Dense blocks - Column 3
        # X0,3: concat(X0,0, X0,1, X0,2, Up(X1,2))
        self.conv0_3 = ConvBlock(features[0] * 3 + features[0], features[0])
        # X1,3: concat(X1,0, X1,1, X1,2, Up(X2,2))
        self.conv1_3 = ConvBlock(features[1] * 3 + features[1], features[1])

        # Dense blocks - Column 4
        # X0,4: concat(X0,0, X0,1, X0,2, X0,3, Up(X1,3))
        self.conv0_4 = ConvBlock(features[0] * 4 + features[0], features[0])

        # ==================== Output ====================
        if deep_supervision:
            self.final1 = nn.Conv2d(features[0], out_channels, 1)
            self.final2 = nn.Conv2d(features[0], out_channels, 1)
            self.final3 = nn.Conv2d(features[0], out_channels, 1)
            self.final4 = nn.Conv2d(features[0], out_channels, 1)
        else:
            self.final = nn.Conv2d(features[0], out_channels, 1)

    def _match_size(self, x, target):
        """Match spatial dimensions via bilinear interpolation"""
        if x.shape[2:] != target.shape[2:]:
            x = nn.functional.interpolate(
                x, size=target.shape[2:], mode="bilinear", align_corners=True
            )
        return x

    def forward(self, x):
        """
        Forward pass through UNet++ core
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            output: (B, out_channels, H, W)
        """
        # ==================== Encoder ====================
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # ==================== Decoder Column 1 ====================
        x0_1 = self.conv0_1(
            torch.cat([x0_0, self._match_size(self.up1_0(x1_0), x0_0)], dim=1)
        )
        x1_1 = self.conv1_1(
            torch.cat([x1_0, self._match_size(self.up2_0(x2_0), x1_0)], dim=1)
        )
        x2_1 = self.conv2_1(
            torch.cat([x2_0, self._match_size(self.up3_0(x3_0), x2_0)], dim=1)
        )
        x3_1 = self.conv3_1(
            torch.cat([x3_0, self._match_size(self.up4_0(x4_0), x3_0)], dim=1)
        )

        # ==================== Decoder Column 2 ====================
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self._match_size(self.up1_1(x1_1), x0_0)], dim=1)
        )
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self._match_size(self.up2_1(x2_1), x1_0)], dim=1)
        )
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, self._match_size(self.up3_1(x3_1), x2_0)], dim=1)
        )

        # ==================== Decoder Column 3 ====================
        x0_3 = self.conv0_3(
            torch.cat(
                [x0_0, x0_1, x0_2, self._match_size(self.up1_2(x1_2), x0_0)], dim=1
            )
        )
        x1_3 = self.conv1_3(
            torch.cat(
                [x1_0, x1_1, x1_2, self._match_size(self.up2_2(x2_2), x1_0)], dim=1
            )
        )

        # ==================== Decoder Column 4 ====================
        x0_4 = self.conv0_4(
            torch.cat(
                [x0_0, x0_1, x0_2, x0_3, self._match_size(self.up1_3(x1_3), x0_0)],
                dim=1,
            )
        )

        # ==================== Output ====================
        if self.deep_supervision and self.training:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return (out1 + out2 + out3 + out4) / 4  # Average for training stability
        else:
            return self.final(x0_4) if hasattr(self, "final") else self.final4(x0_4)


class UNetPlusPlus(nn.Module):
    """
    UNet++ Wrapper for Sequence Processing
    
    Theo CS313-data-mining README:
    - Model 2 (UNet++) core:
        - Input: (B, C, H, W) - single frame
        - Output: (B, 1, H, W) - single frame
        
    Wrapper này xử lý sequence bằng cách:
    1. Flatten (B, T, C, H, W) → (B*T, C, H, W)
    2. Forward qua UNet++ core cho từng frame độc lập
    3. Reshape (B*T, 1, H, W) → (B, T, 1, H, W)
    """

    def __init__(
        self,
        in_channels=24,
        out_channels=1,
        features=[32, 64, 128, 256],
        deep_supervision=False,
    ):
        super(UNetPlusPlus, self).__init__()

        self.unet = UNetPlusPlusCore(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            deep_supervision=deep_supervision,
        )

    def forward(self, x):
        """
        Forward pass through UNet++ wrapper
        
        Args:
            x: Input tensor (B, T, C, H, W) - sequence from PredRNN++
            
        Returns:
            output: (B, T, 1, H, W) - processed sequence
            
        Internal flow (per frame):
            (B, C, H, W) → UNet++ → (B, 1, H, W)
        """
        B, T, C, H, W = x.shape

        # Flatten sequence: (B, T, C, H, W) → (B*T, C, H, W)
        x_flat = x.view(B * T, C, H, W)

        # Forward through UNet++ (each frame independently)
        # (B*T, C, H, W) → (B*T, 1, H, W)
        out_flat = self.unet(x_flat)

        # Reshape back to sequence: (B*T, 1, H, W) → (B, T, 1, H, W)
        out_ch = out_flat.shape[1]
        output = out_flat.view(B, T, out_ch, H, W)

        return output


# ==================== COMBINED MODEL ====================


class PredRNNpp_UNetpp(nn.Module):
    """
    Combined PredRNN++ and UNet++ Model
    
    Training Flow (theo CS313-data-mining README):
    
    Model 1 (PredRNN++):
        - Input: (B, time_in, C, H, W) = (B, 30, 24, 35, 35)
        - Output: (B, time_out, C, H, W) = (B, 14, 24, 35, 35)
        
    Model 2 (UNet++) - xử lý từng frame:
        - Input: (B, C, H, W) = (B, 24, 35, 35)
        - Output: (B, 1, H, W) = (B, 1, 35, 35)
        
    Combined flow:
        Input → PredRNN++ → [14 frames] → UNet++ (per frame) → Output
        (B,30,24,H,W) → (B,14,24,H,W) → (B,14,1,H,W)
        
    Single backward pass để update cả 2 models.
    """

    def __init__(
        self,
        time_in=30,
        time_out=14,
        in_channels=24,
        out_channels=1,
        height=35,
        width=35,
        predrnn_num_hidden=[64, 64, 64, 64],
        predrnn_num_layers=4,
        predrnn_filter_size=5,
        predrnn_stride=1,
        predrnn_layer_norm=True,
        unetpp_features=[32, 64, 128, 256],
        unetpp_deep_supervision=False,
    ):
        super(PredRNNpp_UNetpp, self).__init__()

        self.time_in = time_in
        self.time_out = time_out
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Stage 1: PredRNN++ (Temporal Encoder)
        # NOTE: Official PredRNN uses only 'width' (assumes square input)
        self.predrnn = PredRNNPlusPlus(
            num_layers=predrnn_num_layers,
            num_hidden=predrnn_num_hidden,
            in_channel=in_channels,
            width=width,  # Official uses only width (square input)
            filter_size=predrnn_filter_size,
            stride=predrnn_stride,
            layer_norm=predrnn_layer_norm,
            output_channel=in_channels,  # Output same channels as input
        )

        # Stage 2: UNet++ (Spatial Decoder - per frame)
        # Core: (B, C, H, W) → (B, 1, H, W)
        # Wrapper handles sequence: (B, T, C, H, W) → (B, T, 1, H, W)
        self.unetpp = UNetPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            features=unetpp_features,
            deep_supervision=unetpp_deep_supervision,
        )

    def forward(self, x):
        """
        Forward pass through combined model
        
        Args:
            x: Input tensor (B, T_in, C, H, W) = (B, 30, 24, 35, 35)
            
        Returns:
            output: (B, T_out, 1, H, W) = (B, 14, 1, 35, 35)
        """
        # Stage 1: PredRNN++ - Temporal prediction
        # (B, 30, 24, H, W) → (B, 14, 24, H, W)
        temporal_features = self.predrnn(x, future_length=self.time_out)

        # Stage 2: UNet++ - Spatial decoding (per frame)
        # (B, 14, 24, H, W) → (B, 14, 1, H, W)
        # Internally: each frame (B, 24, H, W) → (B, 1, H, W)
        output = self.unetpp(temporal_features)

        return output

    def get_intermediate_output(self, x):
        """
        Get both intermediate and final outputs for analysis
        
        Args:
            x: Input tensor (B, T_in, C, H, W)
            
        Returns:
            temporal_features: (B, T_out, C, H, W) - Output of PredRNN++
            output: (B, T_out, 1, H, W) - Final output
        """
        temporal_features = self.predrnn(x, future_length=self.time_out)
        output = self.unetpp(temporal_features)
        return temporal_features, output
