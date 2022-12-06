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


# import pretrained model
@st.cache
def load_model(model_name: str):
    unet = model.Unet(3)
    model_params = torch.load(f"unet/{model_name}.pth", map_location=torch.device('cpu'))
    unet.load_state_dict(model_params['model'])
    return unet


# model option
st.subheader("Select a pre-trained model:")
option = st.selectbox("select a model", ('unet_epoche19_iter80', 'unet_epoche20_iter60',
                                         'unet_epoche21_iter80', 'unet_epoche23_iter80',
                                         'unet_epoche24_iter20', 'unet_epoche25_iter60',),
                      label_visibility='collapsed')

