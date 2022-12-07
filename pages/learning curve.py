"""
    CS5001 Fall 2022
    Assignment number of info
    Name / Partner
"""

import streamlit as st
from PIL import Image

# set up necessary variables
path = "data/data_analysis"
model_dict = {'dice loss + cross entropy loss, lrate = 0.0001': "pretrained1",
              'dice loss, lrate = 0.0001': "pretrained2",
              'dice loss + cross entropy loss, lrate = 0.001': "pretrained3",
              'dice loss + cross entropy loss, lrate = 0.01': "pretrained4",
              'dice loss + cross entropy loss, lrate = 0.1': "pretrained5"}

# show learning curves of multiple model with different learning rate
st.subheader("Compare learning curves")
multiple_loss = Image.open(f"{path}/accuracy_curves_multiple_loss.png")
st.image(multiple_loss)

# show details of selected model
st.subheader("See details:")
option = st.selectbox("select a model", ('dice loss + cross entropy loss, lrate = 0.0001',
                                         'dice loss, lrate = 0.0001',
                                         'dice loss + cross entropy loss, lrate = 0.001',
                                         'dice loss + cross entropy loss, lrate = 0.01',
                                         'dice loss + cross entropy loss, lrate = 0.1'),
                      label_visibility='collapsed')
chosen_model = model_dict[option]
accuracy_curve = Image.open(f"{path}/{chosen_model}_accuracy_curve.png")
train_valid_dices = Image.open(f"{path}/{chosen_model}_train_valid_dice.png")

st.image(accuracy_curve)
st.image(train_valid_dices)
