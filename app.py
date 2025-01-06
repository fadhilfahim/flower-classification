import os
import keras
import tensorflow as tf
import numpy as np
from keras.models import load_model
import streamlit as st
from PIL import Image


st.header('floral classification')
flower_names = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']

# Load the trained model
model = load_model('flower_classification_model.keras')

# pridict the sample of this model

# Function to classify image
def classify_image(uploaded_file):
    # Open the uploaded file as a PIL image
    image = Image.open(uploaded_file)
    
    # Resize the image to the required input size for the model
    input_image = image.resize((180, 180))
    
    # Convert the image to a NumPy array and add batch dimension
    input_image_array = tf.keras.preprocessing.image.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    
    # Predict using the model
    prediction = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(prediction[0])  # Apply softmax to get probabilities
    
    # Get the predicted class and confidence
    outcome = f" this is predicted as a {flower_names[np.argmax(result)]} with accuracy : {max(result)} "
    

    return outcome


# Streamlit file uploader
uploaded_file = st.file_uploader("Upload the image")

# When a file is uploaded, classify the image and display the result
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Classify the image and display the result
    result = classify_image(uploaded_file)
    st.markdown(result)

