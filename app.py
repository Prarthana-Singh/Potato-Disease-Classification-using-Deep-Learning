import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from tensorflow.keras.utils import img_to_array  # Fixed import
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')

def streamlit_config():
    st.set_page_config(page_title='Classification', layout='centered')

    page_background_color = """
    <style>
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    st.markdown("""
        <style>
        @keyframes glow {
            0% { text-shadow: 0 0 10px green, 0 0 20px green; }
            50% { text-shadow: 0 0 20px limegreen, 0 0 40px limegreen; }
            100% { text-shadow: 0 0 10px green, 0 0 20px green; }
        }
        .glowing-text {
            font-size: 40px;
            font-weight: bold;
            color: white;
            text-align: center;
            animation: glow 1.5s infinite alternate ease-in-out;
        }
        </style>
        <h1 class="glowing-text">Potato Disease Classification</h1>
    """, unsafe_allow_html=True)
    add_vertical_space(4)

streamlit_config()

def prediction(image_path, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    img = Image.open(image_path)
    image_path.seek(0)  # Reset file position
    img_resized = img.resize((256, 256))
    img_array = img_to_array(img_resized)  # Fixed function
    img_array = np.expand_dims(img_array, axis=0)

    model = tf.keras.models.load_model('model.h5')  # Check path
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    add_vertical_space(1)
    st.markdown(f'<h4 style="color: orange;">Predicted Class: {predicted_class}<br>Confidence: {confidence}%</h4>',
                unsafe_allow_html=True)

    add_vertical_space(1)
    st.image(img.resize((400, 300)))

col1, col2, col3 = st.columns([0.1, 0.9, 0.1])
with col2:
    input_image = st.file_uploader(label='Upload the Image', type=['jpg', 'jpeg', 'png'])

if input_image is not None:
    col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
    with col2:
        prediction(input_image)
