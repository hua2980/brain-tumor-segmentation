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
    model_params = torch.load(f"unet/pretrained/{model_name}.pth", map_location=torch.device('cpu'))
    unet.load_state_dict(model_params['model'])
    return unet


# model option
st.subheader("Select a pre-trained model:")
option = st.selectbox("select a model", ('unet_epoche19_iter80', 'unet_epoche20_iter60',
                                         'unet_epoche21_iter80', 'unet_epoche23_iter80',
                                         'unet_epoche24_iter20', 'unet_epoche25_iter60',),
                      label_visibility='collapsed')

# file uploader
st.subheader("Select a brain MRI image:")
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'tif'])

col1, col2 = st.columns(2)
with col1:
    # image cache
    image_cache = st.container()
    if uploaded_file is not None:
        # convert image into np array
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img)

        # check if the file is valid (3 * 256 * 256)
        if img_array.shape != (256, 256, 3):
            st.write("Image size should be 256*256")
        else:
            # display image
            image_cache.subheader("Your uploaded file:")
            image_cache.image(img_array)

            # store in streamlit session
            st.session_state.img_array = img_array
            img_array = img_array / 255
    elif 'img_array' in st.session_state:
        img_array = st.session_state.img_array
        # display image
        image_cache.subheader("Your uploaded file:")
        image_cache.image(img_array)
    img_pred_button = st.button('Predict')

with col2:
    if img_pred_button:
        if "img_array" not in st.session_state:
            st.write("You haven't upload any file!")
        else:
            # get predicted mask image
            st.subheader("Model Prediction:")
            pred_img = st.session_state.img_array / 255
            pred_mask = predict(pred_img, torch.device('cpu'), load_model(option))
            pred_mask = pred_mask[0].permute(1, 2, 0)
            st.session_state.pred_mask = pred_mask
            st.image(pred_mask.numpy())
            clear = st.button("clear prediction")

            # clear prediction and content
            if clear:
                del st.session_state.pred_maskl
                img_pred_button = False
