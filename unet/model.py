"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    (Model Part: my customized convolutional model)
    Hua Wang
"""

from unet.mylib import *


class Unet(nn.Module):
    def __init__(self, in_channel):
        super(Unet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.down1 = DoubleConv(in_channel, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)
        self.up6 = Up(512, 256)
        self.up7 = Up(256, 128)
        self.up8 = Up(128, 64)
        self.up9 = Up(64, 32)
        # output a single channel (binary)
        self.up10 = nn.ConvTranspose2d(32, 1, stride=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, im_input):
        im_down1 = self.down1(im_input)
        im_down1_pooled = self.max_pool(im_down1)
        im_down2 = self.down2(im_down1_pooled)
        im_down2_pooled = self.max_pool(im_down2)
        im_down3 = self.down3(im_down2_pooled)
        im_down3_pooled = self.max_pool(im_down3)
        im_down4 = self.down4(im_down3_pooled)
        im_down4_pooled = self.max_pool(im_down4)
        im_down5 = self.down5(im_down4_pooled)  # max pooling is not followed
        im_up6 = self.up6(im_down5, im_down4)
        im_up7 = self.up7(im_up6, im_down3)
        im_up8 = self.up8(im_up7, im_down2)
        im_up9 = self.up9(im_up8, im_down1)
        im_output = self.up10(im_up9)
        return self.sigmoid(im_output)


def main():
    # Your code replaces the pass statement here:
    pass


if __name__ == '__main__':
    main()
