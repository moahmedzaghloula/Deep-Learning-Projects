import streamlit as st
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

# Load your trained model
model = load_model('chest_xray.h5')

# Set a beautiful background image
background_image = 'G:\\University\\Graduation Project\\Chest Model\\db37d1babbbe55e711ebb30f31c48493.jpg'
background_style = f"""
    <style>
        body {{
            background-image: url("{background_image}");
            background-size: cover;
        }}
    </style>
"""
st.markdown(background_style, unsafe_allow_html=True)

st.title("Chest X-ray Prediction")

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_data = preprocess_input(img_array)
        prediction = model.predict(img_data)
        result = int(prediction[0][0])

        if result == 0:
            st.success("Person is Affected By PNEUMONIA")
        else:
            st.success("Result is Normal")
    except Exception as e:
        st.error("Error processing the image. {}".format(e))
