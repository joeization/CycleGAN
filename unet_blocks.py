import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import add, mul

from utility import center_crop
from cbam import cbam


class GELU(nn.Module):
    '''Gaussian Error Linear Unit.

    Dan Hendrycks∗, Kevin Gimpel

    GAUSSIAN ERROR LINEAR UNITS (GELUS), 2016

    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    '''

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual_Bottleneck(nn.Module):
    def __init__(self, in_ch, downsample=False):
        super(Residual_Bottleneck, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, in_ch, 1, stride=1,
                               padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn1 = nn.InstanceNorm2d(in_ch)
        # self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.InstanceNorm2d(in_ch)
        self.bn3 = nn.InstanceNorm2d(in_ch)
        if downsample:
            self.conv2 = nn.Conv2d(in_ch, in_ch, 3, stride=2,
                                   padding=1, bias=False)
            self.conv3 = nn.Conv2d(in_ch, in_ch*2, 3, stride=1,
                                   padding=1, bias=False)
            # self.bn3 = nn.BatchNorm2d(in_ch)
            self.downasmple = nn.Sequential(
                nn.Conv2d(in_ch, in_ch*2, 3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(in_ch*2),
            )
        else:
            self.conv2 = nn.Conv2d(in_ch, in_ch, 3, stride=1,
                                   padding=1, bias=False)
            self.conv3 = nn.Conv2d(in_ch, in_ch*2, 3, stride=1,
                                   padding=1, bias=False)
            # self.bn3 = nn.BatchNorm2d(in_ch)
            self.downasmple = None

    def forward(self, x):

        y = self.bn1(x)
        y = self.activate(y)
        y = self.conv1(y)

        y = self.bn2(y)
        y = self.activate(y)
        y = self.conv2(y)

        y = self.bn3(y)
        y = self.activate(y)
        y = self.conv3(y)

        if self.downasmple is not None:
            x = self.downasmple(x)
        y = y+x

        return y


class unet_double_conv(nn.Module):
    '''The double convolution part of UNet

    Args:
        in_ch (int): The channels of input
        out_ch (int): The desired channels of output
        ker (int): The convolution kernal size, default to 3
    '''

    def __init__(self, in_ch, out_ch, ker=3):
        super(unet_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ker, padding=ker//2, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(out_ch//4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ker, padding=ker//2, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(out_ch//4, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class unet_bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, ker=3):
        super(unet_bottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        # self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn1 = nn.GroupNorm(out_ch//4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
        #self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.GroupNorm(out_ch//4, out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 1)
        # self.bn3 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.GroupNorm(out_ch//4, out_ch)
        self.s_conv = nn.Conv2d(in_ch, out_ch, 3)
        # self.s_bn = nn.BatchNorm2d(out_ch)
        self.s_bn = nn.GroupNorm(out_ch//4, out_ch)

    def forward(self, x):
        xp = self.conv1(x)
        xp = self.bn1(xp)
        xp = self.relu(xp)
        xp = self.conv2(xp)
        xp = self.bn2(xp)
        xp = self.relu(xp)
        xp = self.conv3(xp)
        xp = self.bn3(xp)

        x = self.s_conv(x)
        # print(x.shape, xp.shape)
        # x = F.interpolate(x, xp[0][0].size())
        x = self.s_bn(x)
        return self.relu(xp+x)


class unet_input_conv(nn.Module):
    def __init__(self, in_ch, out_ch, ker=3):
        super(unet_input_conv, self).__init__()
        self.conv = unet_double_conv(in_ch, out_ch, ker)
        '''
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, ker, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_ch, out_ch, ker, padding=1)
        '''

    def forward(self, x):
        '''
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.conv3(y)
        x = F.interpolate(x, y[0][0].size())
        y = y+x
        '''
        y = self.conv(x)
        return y


class unet_down_conv(nn.Module):
    '''The encoder part of UNet

    Args:
        in_ch (int): The channels of input, downsampling part(i.e. pooling) will not perform for in_ch <= 3(assuming input image)
        out_ch (int): The desired channels of output
        ker (int): The convolution kernal size, default to 3
    '''

    def __init__(self, in_ch, out_ch, ker=3):
        super(unet_down_conv, self).__init__()
        if in_ch == 1 or in_ch == 2 or in_ch == 3:
            '''
            self.conv = nn.Sequential(
                unet_double_conv(in_ch, out_ch, ker),
                # unet_bottleneck(in_ch, out_ch, ker),
            )
            '''
            self.conv = unet_input_conv(in_ch, out_ch, ker)
        else:
            self.conv = Residual_Bottleneck(in_ch, True)
            '''
            self.conv = nn.Sequential(
                # nn.MaxPool2d(2),
                nn.Conv2d(in_ch, in_ch, 3, stride=2),
                unet_double_conv(in_ch, out_ch, ker),
                # unet_bottleneck(in_ch, out_ch, ker),
            )
            '''

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        return x


def torch_concat(x, y):
    return torch.cat([x, y], 1)


class unet_double_conv_preavtivate(nn.Module):
    '''The double convolution part of UNet

    Args:
        in_ch (int): The channels of input
        out_ch (int): The desired channels of output
        ker (int): The convolution kernal size, default to 3
    '''

    def __init__(self, in_ch, out_ch, ker=3):
        super(unet_double_conv_preavtivate, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(in_ch+out_ch),
            nn.InstanceNorm2d(in_ch+out_ch),
            # nn.GroupNorm((in_ch+out_ch)//4, in_ch+out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch+out_ch, out_ch, ker, padding=ker//2, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            # nn.GroupNorm(out_ch//4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ker, padding=ker//2, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class unet_up_conv(nn.Module):
    '''The decoder part of UNet

    Args:
        in_ch (int): The channels of input
        cat_ch (int): The channels after concatenate features from the corresponding encoder
        out_ch (int): The desired channels of output
    '''

    def __init__(self, in_ch, cat_ch, out_ch, mode='add'):
        super(unet_up_conv, self).__init__()
        # self.up = nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2)
        # self.up1 = nn.ConvTranspose2d(in_ch, out_ch, ker, stride=2)
        # self.up2 = nn.ConvTranspose2d(out_ch, out_ch, ker, stride=2)
        # self.bridge = nn.Conv2d(out_ch, out_ch, ker, stride=1, padding=0)
        # self.conv = unet_double_conv(cat_ch, out_ch) for concat
        # self.conv = unet_double_conv(cat_ch, out_ch)
        self.conv = unet_double_conv_preavtivate(cat_ch, out_ch)
        if mode == 'add':
            self.arithm = add
            self.mapping = nn.Conv2d(cat_ch, in_ch, 1, 1)
        elif mode == 'mul':
            self.arithm = mul
            self.mapping = nn.Conv2d(cat_ch, in_ch, 1, 1)
        elif mode == 'cat':
            self.arithm = torch_concat
            self.mapping = None
        else:
            self.arithm = add
            self.mapping = nn.Conv2d(cat_ch, in_ch, 1, 1)
        # self.drop = nn.Dropout2d(dp)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, scale_factor=2)
        # x2 = self.up(x2)
        y = F.interpolate(x1, x2[0][0].size())  # , mode='bilinear')
        if self.mapping != None:
            y2 = self.mapping(y)
        else:
            y2 = y
        x2 = self.arithm(x2, y2)
        x2 = self.conv(x2)
        x2 = x2+y
        return x2


class unet_output_conv(nn.Module):
    '''Output part of UNet

    Does not contain softmax or sigmoid layer.

    Args:
        in_ch (int): The channels of input
        out_ch (int): The desired channels of output(categories)
        ker (int): The convolution kernal size, default to 1
        stride (int): convolution stride, default to 1
    '''

    def __init__(self, in_ch, out_ch, ker=1, stride=1):
        super(unet_output_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ker, stride=stride)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        return x
