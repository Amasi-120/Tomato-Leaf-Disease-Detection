import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Tomato AI Detector",
    page_icon="🍅",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #fafafa;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:800;
    color:#d62828;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#555;
}

.card {
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.08);
    margin-top:20px;
}

.result {
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

.footer {
    text-align:center;
    color:#777;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<div class='title'>🍅 Tomato Leaf Disease Detection</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-powered system for detecting tomato leaf diseases using Deep Learning</div>",
    unsafe_allow_html=True
)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "TomatoLeaf_Model.h5"
    )


model = load_model()


# -----------------------------
# Upload Section
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Tomato Leaf Image",
    type=["jpg","jpeg","png"]
)

st.markdown("</div>", unsafe_allow_html=True)



if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")


    col1, col2 = st.columns(2)


    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


    # preprocessing
    img = image.resize((224,224))

    img = np.array(img).astype("float32") / 255.0

    img = np.expand_dims(img,axis=0)


    prediction = model.predict(
        img,
        verbose=0
    )[0][0]


    # -----------------------------
    # Result
    # -----------------------------

    if prediction >= 0.5:

        label = "Healthy 🍃"
        confidence = prediction*100

        color="green"

    else:

        label="Diseased 🦠"
        confidence=(1-prediction)*100

        color="red"



    with col2:

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class='result' style='color:{color};'>
            {label}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        st.progress(
            int(confidence)
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )



# -----------------------------
# Model Information
# -----------------------------

st.markdown("---")

st.markdown(
"""
### 🧠 Model Information

- Architecture: **MobileNetV2**
- Technique: **Transfer Learning**
- Classes:
    - 🍃 Healthy
    - 🦠 Diseased
- Framework: TensorFlow / Keras
"""
)


st.markdown(
"""
<div class='footer'>
Developed by Amasi Al-Sahbi | Artificial Intelligence Student
</div>
""",
unsafe_allow_html=True
)
