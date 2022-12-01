import numpy
from sklearn.model_selection import train_test_split
from unet.utils import *
from unet.model import *
from unet.mylib import *


def predict(test_img: numpy.ndarray, device, unet: Unet = None):
    # load my pretrained model is unet is not defined
    if unet is None:
        unet = Unet(3)
        model_params = torch.load("unet/unet_epoche20_iter40.pth", map_location=torch.device('cpu'))
        unet.load_state_dict(model_params['model'])

    # load data into device
    test_content_tensor = torch.from_numpy(test_img).to(device).float().unsqueeze(0)
    test_content_tensor = test_content_tensor.permute(0, 3, 1, 2)

    unet.eval()  # close BatchNorm2d during testing
    # calculate prediction of validation image with no autograd mechanism
    with torch.no_grad():
        pred_mask_tensor = unet.forward(test_content_tensor)
        pred_mask_tensor = torch.round(pred_mask_tensor)

    return pred_mask_tensor
