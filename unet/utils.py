"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    (Model part: utilities)
    Hua Wang
"""

import os
import warnings
import numpy as np
import pandas as pd
import random
import rasterio
import torch
import matplotlib.pyplot as plt


def extract_paths(directory):
    """
    The function reads all files names in the given root directory and returns two
    dataframe: contents and masks.

    :param directory: str
    :return:
    contents:
            dataframe containing info of content image paths and content image id
    masks:
            dataframe contain info of mask image path and mask image id
    """

    masks = []
    contents = []
    # walk through all files in the given directory
    for (root, dirs, files) in os.walk(directory, topdown=False):
        # extract patient id from root path
        patient_id = root.split("/")[-1]

        # skip the main file
        if "TCGA" not in patient_id:
            continue

        for file_name in files:
            # full directory string for the file
            full_dir = os.path.join(root, file_name)
            if "mask" in full_dir:
                mask_id = int(full_dir[56:-9])
                masks.extend([patient_id, full_dir, mask_id])
            else:
                content_id = int(full_dir[56:-4])
                contents.extend([patient_id, full_dir, content_id])

    # split patient_id and full_dir
    mask_patient_ids = masks[::3]
    mask_full_dirs = masks[1::3]
    mask_ids = masks[2::3]
    content_patient_ids = contents[::3]
    content_full_dirs = contents[1::3]
    content_ids = contents[2::3]

    # combine two list (patient_id and full_dir) into a dataframe
    # 1. convert two list into a dictionary
    masks = {"patient_id": mask_patient_ids,
             "full_mask_dir": mask_full_dirs,
             "mask_id": mask_ids}
    contents = {"patient_id": content_patient_ids,
                "full_content_dir": content_full_dirs,
                "content_id": content_ids}
    # 2. convert the dict into a dataframe
    masks = pd.DataFrame(masks)
    contents = pd.DataFrame(contents)

    return masks, contents


def sort_combine_paths(masks, contents):
    """
    The function takes path dataframe for mask images and path dataframe for content images,
    sorts and combines the two dataframes so that the mask file path and the content file
    path are matched in a single row in a single dataframe

    :param masks: dataframe
    :param contents: dataframe
    :return: dataframe
    """

    # sort dataframes with a key function, making sure that mask path and
    # content path is corresponding to each other at each row index
    contents = contents.sort_values(by=["patient_id", "content_id"], ignore_index=True)
    masks = masks.sort_values(by=["patient_id", "mask_id"], ignore_index=True)
    # contents = sorted(original_contents["full_content_dir"].values, key=lambda x: int(x[56:-4]))
    # masks = sorted(masks["full_mask_dir"].values, key=lambda x: int(x[56:-9]))

    # combine two dataframe together
    content_paths = contents.iloc[:, 1]
    mask_paths = masks.iloc[:, 1]
    patient_ids = contents.iloc[:, 0]
    dir_df = pd.DataFrame({
        "patient_id": patient_ids,
        "content_path": content_paths,
        "mask_path": mask_paths
    })

    # randomly select a row, check if paths at the same row are not matched
    idx = random.randint(0, len(dir_df) - 1)
    content_path = dir_df.iloc[idx, 1]
    mask_path = dir_df.iloc[idx, 2]
    if content_path[:-4] != mask_path[:-9]:
        raise Exception("Something failed for matching process")

    return dir_df


def load_data(df_dir: pd.DataFrame, start_idx: int = 0, shuffle: bool = False, batch_size: int = 1):
    """
    The function read images by paths in df_dir and return content image arrays and mask image arrays.
    :param start_idx: int
    :param batch_size: int
    :param shuffle: boolean
    :param df_dir: dataframe
    :return: ndarray, ndarray
    """
    # check if random shuffle
    if shuffle:
        df_dir.sample(frac=1).reset_index(drop=True)

    # check if batch size is valid
    if batch_size > df_dir.shape[0]:
        raise IndexError("batch size is too large")

    # ignore warning
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # read images and convert into tensor
    content_images = []
    mask_images = []
    for i in range(batch_size):
        # read content image
        content_path = df_dir.iloc[start_idx + i, 1]
        mask_path = df_dir.iloc[start_idx + i, 2]

        # read image by rasterio
        content_image = rasterio.open(content_path).read()
        mask_image = rasterio.open(mask_path).read()
        content_images.append(content_image)
        mask_images.append(mask_image)

    return np.array(content_images), np.array(mask_images)


def adjust_data(img, mask):
    """
    The function rescaled data in img to [0, 1] and convert data in mask to binary
    :param img: dataframe
    :param mask: dataframe
    :return: dataframes
    """
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return img, mask


def iou_score(pred, target, smooth: int = 0.001):
    pred = torch.round(pred)  # convert into binary mask

    # intersection
    intersect = torch.sum(pred * target)

    # union
    union = torch.ceil((pred + target)/2)

    # Jaccard = |A∩B| / |A∪B|
    iou_score = (intersect + smooth) / (torch.sum(union) + smooth)

    return iou_score


def dice_score(pred, target, smooth: int=0.001):
    # flatten pred and target
    pred_flattened = pred.reshape(-1)
    target_flattened = target.reshape(-1)

    # intersection
    intersect = torch.dot(pred_flattened, target_flattened)

    # sum
    sum_two = torch.sum(pred_flattened) + torch.sum(target_flattened)

    # dice_score = 2 * |A∩B| / (|A| + |B|)
    dice_score = (2 * intersect + smooth) / (sum_two + smooth)
    return dice_score


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0)
    return img


def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def main():
    # Your code replaces the pass statement here:
    pass


if __name__ == '__main__':
    main()
