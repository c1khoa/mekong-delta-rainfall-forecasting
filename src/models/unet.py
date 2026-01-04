import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, in_channels=21, out_channels=21, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder_layers.append(self._conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1]*2)

        # Decoder
        self.upconv_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for feature in reversed(features):
            self.upconv_layers.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder_layers.append(self._conv_block(feature*2, feature))

        # Output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconv_layers)):
            x = self.upconv_layers[idx](x)
            skip_connection = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = self._crop_to_fit(x, skip_connection)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_layers[idx](x)

        return self.final_conv(x)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _crop_to_fit(self, x, target):
        """
        Crop or pad x to match target's H and W
        """
        _, _, H, W = target.shape
        h_diff = H - x.size(2)
        w_diff = W - x.size(3)

        # Pad nếu x nhỏ hơn target
        x = nn.functional.pad(x, (0, max(0, w_diff), 0, max(0, h_diff)))

        # Crop nếu x lớn hơn target
        x = x[:, :, :H, :W]
        return x