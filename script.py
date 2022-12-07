"""
    CS5001 Fall 2022
    Assignment number of info
    Name / Partner
"""
import pandas as pd

# read lines from log_file.txt
with open("data/pretrained2/log_file.txt", 'r') as f:
    lines = f.readlines()
# convert string into list
loss_accuracy_list = []
valid_train_list = []
prev = ""
for line in lines:
    if 'loss' in line:
        loss_accuracy_list.append(line)
        prev = line
    elif 'valid' in line:
        line = prev[:-1] + " " + line
        loss_accuracy_list[-1] = line

with open('data/pretrained2/log_file.txt', 'w') as f:
    for item in loss_accuracy_list:
        f.write("%s" % item)

test = pd.read_csv("data/image_dirs/test_data.csv")

for i in range(len(test)):
    test.iloc[i, 1] = "." + test.iloc[i, 1]
    test.iloc[i, 2] = "." + test.iloc[i, 2]

