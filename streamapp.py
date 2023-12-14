from keras.models import load_model
from custom_objects import bce_dice_loss, iou_metric
import streamlit as st
import numpy as np
from PIL import Image
import io
from enum import Enum

# Set page title
st.title("Brain tumor segmentation")
st.write("This app segments brain tumors from MRI images")

# default value for input parameter
class ModelName(str, Enum):
    unet = "Unet"
    unetpp = "Unet++"
    model3 = "Model 3"

# Function to load the selected model
def load_selected_model(model_name):
    if model_name == ModelName.unet:
        return load_model('models/model_best_checkpoint_unet.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou_metric': iou_metric})
    elif model_name == ModelName.unetpp:
        return load_model('models/model_best_checkpoint_unet++.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'iou_metric': iou_metric})
    elif model_name == ModelName.model3:
        # Load your third model here
        pass

# Sidebar for selecting the model
model_option = st.sidebar.selectbox("Select Model", ["Unet", "Unet++", "Model 3"])

# Load the selected model
model = load_selected_model(model_option)
THRESHOLD = 0.2

# Main content
uploaded_file = st.file_uploader("Upload an image to process...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    content = uploaded_file.read()
    image = Image.open(io.BytesIO(content)).convert("L")  # Convert to grayscale
    image = image.resize((128, 128))
    image_array = np.array(image).reshape((1, 128, 128, 1)).astype("float32")# / 255

    # Make prediction
    prediction = (model.predict(image_array) > THRESHOLD) * 1
    prediction = prediction.reshape((128, 128)) * 255
    prediction = np.stack((prediction,) * 3, axis=-1)

    # Tạo một mảng mask để xác định vị trí màu trắng trong hình ảnh
    white_mask = np.all(prediction == [255, 255, 255], axis=-1)

    # Thay thế các điểm màu trắng bằng màu đỏ trong kết quả
    result_image = image_array.reshape((128, 128))
    result_image = np.stack((result_image,) * 3, axis=-1)
    result_image[white_mask] = [219, 0, 0]
    result_image = result_image/255
    
    # Display results
    col1, col2 = st.columns(2)
    col1.header("Original")
    col2.header("Prediction")
    col1.image(image, use_column_width=True)
    col2.image(result_image, use_column_width=True)