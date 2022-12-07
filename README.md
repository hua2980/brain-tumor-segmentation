# brain-tumor-segmentation
A UNet model for brain tumor segmentation. Pytorch version. 
A web app to visualize learning curves of pretrained models and make prediction with pretrained models.

<a href="https://hua2980-brain-tumor-segmentation-home-0fuyzc.streamlit.app">visit web page<a>

### data source
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation


## Install Dependencies
run `pip install -r requirements.txt` in terminal 

## Unet

<img src="data/data_analysis/unet_brain_mri.png" style="zoom:30%;" />


## Training Model

To train the model, run `train.py`:

<img src="data/data_analysis/training process.png" style="zoom:80%"/>

## Make Prediction

To start the web app, run `streamlit run home.py` in terminal.

<img src="data/data_analysis/prediction page.png" style="zoom:50%"/>
