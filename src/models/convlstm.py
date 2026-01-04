import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        H, W = image_size
        return (torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
                torch.zeros(batch_size, self.hidden_dim, H, W, device=device))
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]*num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_in = input_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(ConvLSTMCell(cur_in, self.hidden_dim[i], kernel_size, bias=bias))

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1,0,2,3,4)
        B, T, C, H, W = x.size()
        if hidden_state is None:
            device = x.device
            hidden_state = [cell.init_hidden(B, (H,W), device) for cell in self.cell_list]

        layer_output_list = []
        last_state_list = []

        cur_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = self.cell_list[layer_idx](cur_input[:,t], [h,c])
                output_inner.append(h)
            layer_out = torch.stack(output_inner, dim=1)
            cur_input = layer_out
            layer_output_list.append(layer_out)
            last_state_list.append([h,c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

class ConvLSTMForecaster(nn.Module):
    def __init__(self, input_channels, hidden_dims=[64,32,16], kernel_size=3, dropout_rate=0.2):
        super().__init__()
        self.convlstm = ConvLSTM(input_channels, hidden_dims, kernel_size, len(hidden_dims),
                                 batch_first=True)
        self.batch_norm = nn.BatchNorm2d(hidden_dims[-1])
        self.dropout = nn.Dropout2d(dropout_rate)
        self.output_conv = nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        layer_out, last_state = self.convlstm(x)
        h_last = last_state[0][0]  # last hidden of last layer
        h_last = self.batch_norm(h_last)
        h_last = self.dropout(h_last)
        out = self.output_conv(h_last)
        out = out.unsqueeze(1)  # [B,1,C,H,W]
        return out