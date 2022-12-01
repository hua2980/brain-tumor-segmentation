"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation (Model Part)
    Hua Wang
"""
import torch
from sklearn.model_selection import train_test_split
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


def main():
    root_dir = "./kaggle_3m"
    # select all file paths into two dataframes
    masks, contents = extract_paths(root_dir)
    # sort paths and combine dataframes
    dir_df = sort_combine_paths(masks, contents)
    # split training set, validation set and test set, not depend on patient id
    train_dirs, test_dirs = train_test_split(dir_df, test_size=0.1)

    # load my pretrained model
    unet = Unet(3)
    model_params = torch.load("unet/unet_epoche20_iter40.pth", map_location=torch.device('cpu'))
    unet.load_state_dict(model_params['model'])
    print("model loaded")

    # make prediction with pretrained unet
    true_mask, pred_mask, pred_dice_score, pred_iou_score = test(unet, test_dirs, torch.device('cpu'), batch_size=20)
    print("IoU score: %s \n F1 score: %s" % (pred_iou_score.item(), pred_dice_score.item()))

    # get a view of one test image
    rand = random.randint(0, 19)
    view_pred_mask = pred_mask[rand, :]
    view_true_mask = true_mask[rand, :]
    show_from_tensor(view_true_mask)
    show_from_tensor(view_pred_mask)


if __name__ == '__main__':
    main()
