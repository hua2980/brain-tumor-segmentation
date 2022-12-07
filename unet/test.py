"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation (Model Part)
    Hua Wang
"""
from unet.utils import *
from unet.model import *
from unet.mylib import *


def test(unet: Unet, test_img: pd.DataFrame, device, batch_size: int = 10):
    # load data into device
    test_content, test_mask = load_data(test_img, shuffle=True, batch_size=batch_size)
    test_content, test_mask = adjust_data(test_content, test_mask)
    test_content_tensor = torch.from_numpy(test_content).to(device).float()
    test_mask_tensor = torch.from_numpy(test_mask).to(device).float()

    unet.eval()  # close BatchNorm2d during testing
    # calculate prediction of validation image with no autograd mechanism
    with torch.no_grad():
        pred_mask_tensor = unet.forward(test_content_tensor)
        pred_mask_tensor = torch.round(pred_mask_tensor)
        pred_dice_score = dice_score(pred_mask_tensor, test_mask_tensor)
        pred_iou_score = iou_score(pred_mask_tensor, test_mask_tensor)

    return test_mask_tensor, pred_mask_tensor, pred_dice_score, pred_iou_score


def run_test(option):
    # read data from csv
    test_dirs = pd.read_csv("../data/image_dirs/test_data.csv")

    # load my pretrained1 model
    unet = Unet(3)
    model_params = torch.load(option, map_location=torch.device('cpu'))
    unet.load_state_dict(model_params['model'])
    print("model loaded")

    # make prediction with pretrained1 unet
    true_mask, pred_mask, pred_dice_score, pred_iou_score = test(unet, test_dirs, torch.device('cpu'), batch_size=393)

    print("IoU score: %s \n F1 score: %s" % (pred_iou_score, pred_dice_score))


if __name__ == '__main__':
    model = "pretrained1"
    run_test(f"../data/{model}/{model}.pth")
