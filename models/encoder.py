import torch.nn as nn
import torch.nn.functional as F


class DilatedResConv(nn.Module):
    def __init__(self, channels, dilation=1, activation='relu', padding=1, kernel_size=3, left_pad=0):
        super().__init__()
        in_channels = channels

        if activation == 'relu':
            self.activation = lambda *args, **kwargs: F.relu(*args, **kwargs, inplace=True)
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'glu':
            self.activation = F.glu
            in_channels = channels // 2

        self.left_pad = left_pad
        self.dilated_conv = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=1,
                                      padding=dilation * padding, dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(in_channels, channels,
                                  kernel_size=1, bias=True)

    def forward(self, x):
        x_ = x

        if self.left_pad > 0:
            x_ = F.pad(x_, (self.left_pad, 0))
        x_ = self.dilated_conv(x_)
        x_ = self.activation(x_)
        x_ = self.conv_1x1(x_)

        return x + x_


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_blocks = config.blocks
        self.n_layers = config.hidden_layers
        self.channels = config.channels
        self.latent_channels = config.latent_dim
        self.activation = config.func

        try:
            self.encoder_pool = config.pool
        except AttributeError:
            self.encoder_pool = 800

        layers = []
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2 ** i
                layers.append(DilatedResConv(self.channels, dilation, self.activation))
        self.dilated_convs = nn.Sequential(*layers)

        self.start = nn.Conv1d(1, self.channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv_1x1 = nn.Conv1d(self.channels, self.latent_channels, 1)
        self.pool = nn.AvgPool1d(self.encoder_pool)

    def forward(self, x):
        x = x / 255 - .5
        if x.dim() < 3:
            x = x.unsqueeze(1)

        x = self.start(x)
        x = self.dilated_convs(x)
        x = self.conv_1x1(x)
        x = self.pool(x)

        return x
