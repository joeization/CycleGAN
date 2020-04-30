import torch
import torch.nn as nn

from cbam import cbam_channel, cbam_spatial, cbam
from unet_blocks import unet_down_conv, unet_output_conv, unet_up_conv
from utility import center_crop


class UNet(nn.Module):
    '''Implement of UNet

    O. Ronneberger, P. Fischer, and T. Brox.
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    International Conference on Medical Image Computing and Computer-Assisted Intervention, 2015.

    Args:
        in_ch (int): The channels of input
        n_classes (int): The output categories, default to 1
    '''

    def __init__(self, in_ch, n_classes=1, mode='add'):
        super(UNet, self).__init__()

        # self.drop = nn.Dropout2d(p=0.5)
        base = 32

        # self.inconv = unet_double_conv(in_ch, base)

        self.down1 = unet_down_conv(in_ch, base*1)
        self.down2 = unet_down_conv(base*1, base*2)
        self.down3 = unet_down_conv(base*2, base*4)
        self.down4 = unet_down_conv(base*4, base*8)
        self.down5 = unet_down_conv(base*8, base*16)

        if mode == 'cat':
            self.up1 = unet_up_conv(
                base*16, base*16, base*8, mode=mode)
            self.up2 = unet_up_conv(
                base*8, base*8, base*4, mode=mode)
            self.up3 = unet_up_conv(
                base*4, base*4, base*2, mode=mode)
            self.up4 = unet_up_conv(
                base*2, base*2, base*1, mode=mode)
        else:
            self.up1 = unet_up_conv(
                base*16, base*8, base*8, mode=mode)
            self.up2 = unet_up_conv(
                base*8, base*4, base*4, mode=mode)
            self.up3 = unet_up_conv(
                base*4, base*2, base*2, mode=mode)
            self.up4 = unet_up_conv(
                base*2, base*1, base*1, mode=mode)

        '''UNet++
        self.up11 = unet_up_conv(base*2, base*1*2, base*1)

        self.up21 = unet_up_conv(base*4, base*2*2, base*2)
        self.up22 = unet_up_conv(base*2, base*1*3, base*1)

        self.up31 = unet_up_conv(base*8, base*4*2, base*4)
        self.up32 = unet_up_conv(base*4, base*2*3, base*2)
        self.up33 = unet_up_conv(base*2, base*1*4, base*1)

        self.up41 = unet_up_conv(base*16, base*8*2, base*8)
        self.up42 = unet_up_conv(base*8, base*4*3, base*4)
        self.up43 = unet_up_conv(base*4, base*2*4, base*2)
        self.up44 = unet_up_conv(base*2, base*1*5, base*1)

        self.ds1 = unet_output_conv(base, n_classes)
        self.ds2 = unet_output_conv(base, n_classes)
        self.ds3 = unet_output_conv(base, n_classes)
        '''

        self.outconv = unet_output_conv(base, n_classes)

    def forward(self, x):
        # x00 = self.inconv(x00)

        x00 = self.down1(x)
        x10 = self.down2(x00)
        x20 = self.down3(x10)
        x30 = self.down4(x20)
        x40 = self.down5(x30)

        # x30 = self.drop(x30)
        # x40 = self.cbam1(x40)
        # x30 = self.drop(x30)

        x31 = self.up1(x30, x40)
        x22 = self.up2(x20, x31)
        # x22 = self.cbam2(x22)
        x13 = self.up3(x10, x22)
        # x13 = self.cbam3(x13)
        x04 = self.up4(x00, x13)

        '''UNet++
        x01 = self.up11([x00], x10)

        x11 = self.up21([x10], x20)
        x02 = self.up22([x00, x01], x11)

        x21 = self.up31([x20], x30)
        x12 = self.up32([x10, x11], x21)
        x03 = self.up33([x00, x01, x02], x12)


        x31 = self.up41([x30], x40)
        x22 = self.up42([x20, x21], x31)
        x13 = self.up43([x10, x11, x12], x22)
        x04 = self.up44([x00, x01, x02, x03], x13)

        x01 = center_crop(x01, x04.size()[2], x04.size()[3])
        x02 = center_crop(x02, x04.size()[2], x04.size()[3])
        x03 = center_crop(x03, x04.size()[2], x04.size()[3])
        x1 = self.ds1(x01)
        x2 = self.ds2(x02)
        x3 = self.ds3(x03)

        '''
        x = self.outconv(x04)
        x = torch.tanh(x)
        # x = torch.sigmoid(x)
        return x
