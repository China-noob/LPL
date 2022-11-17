import streamlit as st
from PIL import Image
from max import predict,process_image
import time
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
    labels= predict(img)

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
        labels = predict(img)
        # print out the top 5 prediction labels with scores
        st.success('successful prediction')

        st.write("Prediction ", labels)

        st.write("")
