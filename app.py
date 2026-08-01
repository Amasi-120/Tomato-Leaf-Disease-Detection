import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image and let the AI detect whether it is Healthy or Diseased.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("TomatoLeaf_Model.h5")

model = load_model()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # Resize image
    img = image.resize((224, 224))

    # Convert to NumPy and normalize
    img = np.array(img).astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0][0]

    st.write(f"Prediction Value: {prediction:.6f}")

    # Classification
    if prediction < 0.5:
        label = "Diseased"
        confidence = (1 - prediction) * 100
        st.error(f"Prediction: {label}")
    else:
        label = "Healthy"
        confidence = prediction * 100
        st.success(f"Prediction: {label}")

    st.write(f"Confidence: {confidence:.2f}%")