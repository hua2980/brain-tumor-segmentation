# brain-tumor-segmentation
A UNet model for brain tumor segmentation. Pytorch version. 
A web app to visualize learning curves of pretrained models and make prediction with pretrained models.

<a href="https://hua2980-brain-tumor-segmentation-home-0fuyzc.streamlit.app">visit web page<a>
<a href="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation">data source<a>

## Unet

<img src="data/data_analysis/unet_brain_mri.png" style="zoom:30%;" />


## Training Model

 To start the web app on your own computer:
1. Download project
2. Install dependencies by runing `pip install -r requirements.txt` in terminal 
3. Train the model by runninng `train.py`:

A quick view of training process:
<img src="data/data_analysis/training process.png" style="zoom:80%"/>

## Make Prediction

To start the web app on your own computer:
1. Download project
2. install dependencies by runing `pip install -r requirements.txt` in terminal 
3. open the page by running `streamlit run home.py` in terminal.

A quick view of prediction page:
<img src="data/data_analysis/prediction page.png" style="zoom:50%"/>
