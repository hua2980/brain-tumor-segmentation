"""
    CS5001 Fall 2022
    Assignment number of info
    Name / Partner
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
from unet.predict import predict
from unet import model

model_dict = {'dice loss + cross entropy loss, lrate = 0.0001': "pretrained1",
              'dice loss, lrate = 0.0001': "pretrained2",
              'dice loss + cross entropy loss, lrate = 0.001': "pretrained3",
              'dice loss + cross entropy loss, lrate = 0.01': "pretrained4",
              'dice loss + cross entropy loss, lrate = 0.1': "pretrained5"}

st.subheader("Compare learning curves")


st.subheader("See details:")
option = st.selectbox("select a model", ('dice loss + cross entropy loss, lrate = 0.0001',
                                         'dice loss, lrate = 0.0001',
                                         'dice loss + cross entropy loss, lrate = 0.001',
                                         'dice loss + cross entropy loss, lrate = 0.01',
                                         'dice loss + cross entropy loss, lrate = 0.1'),
                      label_visibility='collapsed')