"""
    CS5001 Fall 2022
    Assignment number of info
    Name / Partner
"""

import numpy
import rasterio.plot as plot

from unet.utils import *
from unet.model import *
from unet.mylib import *


def predict(test_img: numpy.ndarray, device, unet: Unet):
    # load data into device
    test_content_tensor = torch.from_numpy(test_img).to(device).float().unsqueeze(0)
    test_content_tensor = test_content_tensor.permute(0, 3, 1, 2)

    unet.eval()  # close BatchNorm2d during testing
    # calculate prediction of validation image with no autograd mechanism
    with torch.no_grad():
        pred_mask_tensor = unet.forward(test_content_tensor)
        pred_mask_tensor = torch.round(pred_mask_tensor)

    return pred_mask_tensor


def main():
    # Your code replaces the pass statement here:
    pass


if __name__ == '__main__':
    main()
