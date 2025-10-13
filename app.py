# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide", page_icon="🐾")

# Advanced CSS for cards and badges
st.markdown("""
<style>
h1 {
    text-align:center;
    color:#6A0DAD;
    font-family:'Segoe UI', sans-serif;
}

.prediction-card {
    background-color:#F3F4F6;
    border-radius:15px;
    padding:20px;
    margin-top:20px;
    text-align:center;
    box-shadow:0 8px 20px rgba(0,0,0,0.15);
}

.animal-badge {
    display:inline-block;
    background-color:#E0E7FF;
    color:#1E3A8A;
    padding:5px 12px;
    border-radius:12px;
    margin:4px;
    font-weight:bold;
}
.animal-badge.predicted {
    background-color:#6A0DAD;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🐾 Animal Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image and the AI will predict the animal!</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_classifier_model.h5")

model = load_model()

# Classes
CLASS_NAMES = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
               'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
               'Panda', 'Tiger', 'Zebra']

# File uploader
uploaded_file = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Columns for image and prediction
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.image(image, use_column_width=True, caption="Uploaded Image")
    
    # Preprocess
    img = image.resize((224,224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Prediction
    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])
    pred_idx = np.argmax(score)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(score) * 100)
    
    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🧠 Predicted Animal:</h2>
            <h1 style="color:#6A0DAD;">{pred_class}</h1>
            <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        st.progress(confidence/100)
    
    # Show all animals and highlight predicted one
    st.markdown("### Possible Animals Model Can Predict:")
    badges_html = ""
    for cls in CLASS_NAMES:
        if cls == pred_class:
            badges_html += f'<span class="animal-badge predicted">{cls}</span>'
        else:
            badges_html += f'<span class="animal-badge">{cls}</span>'
    st.markdown(badges_html, unsafe_allow_html=True)
    
    st.divider()
    st.success("🎉 Upload another image to test the model again!")
