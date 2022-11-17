import streamlit as st
from PIL import Image
from max import predict,process_image
import time
from torchvision.models import resnet50
import torch
from torch import nn
model = resnet50(
    pretrained=True
)  # to use more models, see https://pytorch.org/vision/stable/models.html
model.fc = nn.Linear(
    model.fc.in_features, 196
)  # set fc layer of model with exact class number of current dataset
model.load_state_dict(torch.load('max_acc.pth'))
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Car Simple Image Classification App")
st.write("")
st.write("")
option = st.selectbox(
     'Choose the model you want to use?',
     ('resnet50', 'other'))
""

file_up = st.file_uploader("Upload an image", type="jpg")
if file_up is None:

    image=Image.open("image/001808.jpg")
    file_up="image/01186.jpg"
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    img=process_image(file_up)
    labels= predict(model,img)

    # print out the top 5 prediction labels with scores
    st.success('successful prediction')

    st.write("Prediction ", labels)


    st.write("")


else:
    if option == 'resnet50':
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")

        img = process_image(file_up)
        labels = predict(model,img)
        # print out the top 5 prediction labels with scores
        st.success('successful prediction')

        st.write("Prediction ", labels)

        st.write("")
