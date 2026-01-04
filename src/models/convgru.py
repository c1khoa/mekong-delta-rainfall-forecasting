import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.bias = bias

        # Gộp x_t và h_{t-1} rồi conv ra 2 * hidden_dim (z, r)
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

        # Conv cho candidate hidden state \tilde{h}_t
        self.conv_can = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x, h_cur):
        """
        x: [B, C_in, H, W]
        h_cur: [B, hidden_dim, H, W]
        """
        # Tính z_t, r_t
        combined = torch.cat([x, h_cur], dim=1)  # [B, C_in + hidden_dim, H, W]
        gates = self.conv_gates(combined)        # [B, 2*hidden_dim, H, W]
        z_gate, r_gate = torch.split(gates, self.hidden_dim, dim=1)

        z = torch.sigmoid(z_gate)
        r = torch.sigmoid(r_gate)

        # Candidate hidden state
        combined_reset = torch.cat([x, r * h_cur], dim=1)
        h_tilde = torch.tanh(self.conv_can(combined_reset))

        # Cập nhật hidden
        h_next = (1 - z) * h_cur + z * h_tilde
        return h_next

    def init_hidden(self, batch_size, image_size, device):
        H, W = image_size
        return torch.zeros(batch_size, self.hidden_dim, H, W, device=device)

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_in = input_dim if i == 0 else self.hidden_dim[i-1]
            self.cell_list.append(
                ConvGRUCell(cur_in, self.hidden_dim[i], kernel_size, bias=bias)
            )

    def forward(self, x, hidden_state=None):
        """
        x: [B, T, C, H, W] nếu batch_first=True
        hidden_state: list length = num_layers, mỗi phần tử là h_0 shape [B, hidden_dim, H, W]
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # [T,B,C,H,W] -> [B,T,C,H,W]

        B, T, C, H, W = x.size()
        device = x.device

        # Khởi tạo hidden nếu chưa có
        if hidden_state is None:
            hidden_state = [cell.init_hidden(B, (H, W), device) for cell in self.cell_list]

        layer_output_list = []
        last_state_list = []

        cur_input = x
        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []

            for t in range(T):
                h = self.cell_list[layer_idx](cur_input[:, t], h)  # [B, hidden_dim, H, W]
                output_inner.append(h)

            layer_out = torch.stack(output_inner, dim=1)  # [B, T, hidden_dim, H, W]
            cur_input = layer_out

            layer_output_list.append(layer_out)
            last_state_list.append(h)  # chỉ có h, không có c

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]   # list 1 phần tử
            last_state_list = last_state_list[-1:]       # list 1 phần tử

        return layer_output_list, last_state_list


class ConvGRUForecaster(nn.Module):
    def __init__(self, input_channels, hidden_dims=[64, 32, 16],
                 kernel_size=3, dropout_rate=0.2):
        super().__init__()
        self.convgru = ConvGRU(
            input_dim=input_channels,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=len(hidden_dims),
            batch_first=True
        )
        self.batch_norm = nn.BatchNorm2d(hidden_dims[-1])
        self.dropout = nn.Dropout2d(dropout_rate)
        self.output_conv = nn.Conv2d(
            in_channels=hidden_dims[-1],
            out_channels=input_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        """
        x: [B, T_in, C, H, W]
        return: [B, 1, C, H, W]
        """
        layer_out, last_state = self.convgru(x)
        # last_state là list (1 phần tử nếu return_all_layers=False)
        # mỗi phần tử là tensor [B, hidden_dim_last, H, W]
        h_last = last_state[0]  # [B, hidden_dims[-1], H, W]

        h_last = self.batch_norm(h_last)
        h_last = self.dropout(h_last)
        out = self.output_conv(h_last)    # [B, C_in, H, W]
        out = torch.sigmoid(out)
        out = out.unsqueeze(1)            # [B, 1, C_in, H, W]
        return out