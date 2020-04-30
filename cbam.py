import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMaxPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w*h).permute(0, 2, 1)
        pooled = F.max_pool1d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
        # _, _, c = input.size()
        # input = input.permute(0, 2, 1)
        return pooled.permute(0, 2, 1).view(n, 1, w, h)
        # return input.view(n, c, w, h)


class ChannelAvgPool(nn.AvgPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w*h).permute(0, 2, 1)
        pooled = F.avg_pool1d(input, self.kernel_size, self.stride,
                              self.padding, self.ceil_mode, self.count_include_pad)
        # _, _, c = input.size()
        # input = input.permute(0, 2, 1)
        return pooled.permute(0, 2, 1).view(n, 1, w, h)
        # return input.view(n, c, w, h)


class cbam_channel(nn.Module):
    '''The channel attention module of CBAM
    Args:
        ch (int): The channels of input
    '''

    def __init__(self, ch, w, h, reduction=4):
        super(cbam_channel, self).__init__()
        self.channel_max = nn.MaxPool2d((w, h), 1)
        self.channel_avg = nn.AvgPool2d((w, h), 1)
        self.activate = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//reduction, ch, 1)
        )

    def forward(self, x):
        m = self.channel_max(x)
        m = self.conv(m)

        a = self.channel_avg(x)
        a = self.conv(a)

        return self.activate(m+a)


class cbam_spatial(nn.Module):
    '''The channel attention module of CBAM
    '''

    def __init__(self, ch):
        super(cbam_spatial, self).__init__()
        self.channel_max = ChannelMaxPool(ch)
        self.channel_avg = ChannelAvgPool(ch)
        self.activate = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        m = self.channel_max(x)
        a = self.channel_avg(x)

        x = torch.cat([m, a], dim=1)

        return self.conv(x)


class cbam(nn.Module):
    '''CBAM
    '''

    def __init__(self, w, h, ch):
        super(cbam, self).__init__()
        self.cbam_c = cbam_channel(ch, w, h)
        self.cbam_s = cbam_spatial(ch)
    def forward(self, x):
        # print(x.shape)
        ch = self.cbam_c(x)
        x = x*ch
        # sp = self.cbam_s(x)
        # x = x*sp
        return x


if __name__ == "__main__":
    net = cbam_channel(10, 64, 16)
    z = np.zeros((2, 10, 64, 16))
    z = torch.from_numpy(z).float()
    print('z:', z.shape)
    output_c = net(z)
    print('output_c:', output_c.shape)

    net = cbam_spatial(10)
    z = np.zeros((2, 10, 64, 16))
    z = torch.from_numpy(z).float()
    print('z:', z.shape)
    output_s = net(z)
    print('output_s:', output_s.shape)

    d = np.zeros((2, 10, 64, 16))
    d = torch.from_numpy(d).float()
    print('d:', d.shape)
    d = d*output_c
    print('d*output_c:', d.shape)
    d = d*output_s
    print('d*output_s:', d.shape)
