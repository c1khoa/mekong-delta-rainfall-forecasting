import torch
import torch.nn as nn
import torch.nn.functional as F

def wrap(input, flow):
    B, C, H, W = input.size()
    device = input.device
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(input, vgrid, align_corners=True)


class FlowGenerator(nn.Module):
    def __init__(self, input_channels, hidden_channels, L):
        super().__init__()
        self.L = L
        self.i2f_conv = nn.Conv2d(input_channels, 32, 5, padding=2)
        self.h2f_conv = nn.Conv2d(hidden_channels, 32, 5, padding=2)
        self.flows_conv = nn.Conv2d(32, L*2, 5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, h):
        f = self.i2f_conv(x) + self.h2f_conv(h)
        f = self.relu(f)
        flows = self.flows_conv(f)
        B, _, H, W = flows.size()
        flows = flows.view(B, self.L, 2, H, W)
        return [flows[:, i, :, :, :] for i in range(self.L)]


class TrajGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, L):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.L = L
        self.i2h = nn.Conv2d(input_channels, 3*hidden_channels, 3, padding=1)
        self.h2h = nn.Conv2d(hidden_channels*L, 3*hidden_channels, 1)
        self.flow_gen = FlowGenerator(input_channels, hidden_channels, L)

    def forward(self, x, h_prev):
        i2h = self.i2h(x)
        i2h_slice = torch.split(i2h, self.hidden_channels, dim=1)

        flows = self.flow_gen(x, h_prev)
        wrapped = [wrap(h_prev, flow) for flow in flows]
        wrapped_cat = torch.cat(wrapped, dim=1)

        h2h = self.h2h(wrapped_cat)
        h2h_slice = torch.split(h2h, self.hidden_channels, dim=1)

        reset_gate = torch.sigmoid(i2h_slice[0] + h2h_slice[0])
        update_gate = torch.sigmoid(i2h_slice[1] + h2h_slice[1])
        new_mem = torch.tanh(i2h_slice[2] + reset_gate * h2h_slice[2])

        h_new = update_gate * h_prev + (1 - update_gate) * new_mem
        return h_new


class TrajGRU(nn.Module):
    def __init__(self,
                 time_out=14,
                 input_channels=24,
                 hidden_channels_list=[16, 16, 8],
                 output_channels=24,
                 L=5,
                 dropout_rate=0.2):
        super().__init__()
        self.time_out=time_out
        self.num_layers = len(hidden_channels_list)
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_ch = input_channels
        for h_ch in hidden_channels_list:
            self.layers.append(TrajGRUCell(in_ch, h_ch, L))
            self.bn_layers.append(nn.BatchNorm2d(h_ch))
            self.dropout_layers.append(nn.Dropout2d(dropout_rate))
            in_ch = h_ch

        self.conv_out = nn.Conv2d(hidden_channels_list[-1], output_channels, 3, padding=1)

    def forward(self, x):
        B, time_in, C, H, W = x.size()
        device = x.device

        h = [torch.zeros(B, layer.hidden_channels, H, W, device=device) for layer in self.layers]
        outputs = []

        for t in range(time_in):
            input_t = x[:, t]
            for l, layer in enumerate(self.layers):
                h[l] = layer(input_t, h[l])
                h[l] = self.bn_layers[l](h[l])
                h[l] = self.dropout_layers[l](h[l])
                input_t = h[l]

        input_t = x[:, -1]
        for t in range(self.time_out):
            for l, layer in enumerate(self.layers):
                h[l] = layer(input_t, h[l])
                h[l] = self.bn_layers[l](h[l])
                h[l] = self.dropout_layers[l](h[l])
                input_t = h[l]
            out_t = self.conv_out(input_t)
            outputs.append(out_t)
            input_t = out_t

        return torch.stack(outputs, dim=1)  # [B, time_out, C, H, W]
