"""
    CS5001 Fall 2022
    Assignment number of info
    Name / Partner
"""

from unet.test import *
import pandas as pd


def get_loss_accuracy_dict(log_file):
    # read lines from log_file
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # convert string into list
    loss_accuracy_list = []
    for line in lines:
        if 'loss' in line:
            loss_accuracy_list.append(line)

    # convert list into dict
    loss_accuracy_dict = {}
    for loss_accuracy in loss_accuracy_list:
        epoch_iteration = loss_accuracy.split('-')[1] + '-' + loss_accuracy.split('-')[3]
        loss = loss_accuracy.split('loss')[1].split('accuracy')[0].strip()
        accuracy = loss_accuracy.split('accuracy')[1].strip()
        if 'valid' in accuracy:
            accuracy = accuracy.split("valid")[0].strip()
        loss_accuracy_dict[epoch_iteration] = [loss, accuracy]
    return loss_accuracy_dict


def convert_dict_to_df(loss_accuracy_dict):
    df = pd.DataFrame(columns=['epoch-iteration', 'iterations', 'loss', 'accuracy'])
    for key, value in loss_accuracy_dict.items():
        epoch_iteration = key.split(":")[0].split("-")
        epoch = int(epoch_iteration[0])
        iteration = int(epoch_iteration[1])
        df_ = pd.DataFrame([[key, (epoch-1)*88+iteration, float(value[0]), float(value[1])]], columns=['epoch-iteration', 'iterations', 'loss', 'accuracy'])
        df = pd.concat([df, df_])
    return df


def draw_curve(loss_df):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_df['iterations'], loss_df['accuracy'], label='accuracy')
    plt.legend()
    plt.show()


def main():
    loss_accuracy_dict = get_loss_accuracy_dict('log_file.txt')
    loss_accuracy_df = convert_dict_to_df(loss_accuracy_dict)
    draw_curve(loss_accuracy_df)


if __name__ == '__main__':
    main()
