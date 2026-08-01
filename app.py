import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="wide"
)


# =====================================================
# Custom CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        180deg,
        #f8fafc 0%,
        #ffffff 100%
    );
}


/* Header */

.hero {

    background:
    linear-gradient(
        135deg,
        #166534,
        #dc2626
    );

    padding: 55px;

    border-radius: 35px;

    text-align:center;

    color:white;

    margin-bottom:35px;

}


.hero h1 {

    font-size:52px;

    font-weight:900;

    margin-bottom:15px;

}


.hero p {

    font-size:22px;

}



/* Cards */

.card {

    background:white;

    padding:35px;

    border-radius:30px;

    box-shadow:
    0 15px 35px rgba(0,0,0,0.08);

    margin-bottom:25px;

}



/* Upload */

.upload-box {

    background:#f0fdf4;

    border:3px dashed #16a34a;

    padding:35px;

    border-radius:30px;

}



/* Result */

.result {

    text-align:center;

    font-size:42px;

    font-weight:900;

    padding:20px;

}


.confidence-box {

    background:#f8fafc;

    border-radius:25px;

    padding:25px;

    text-align:center;

}



/* Sections */

.section-title {

    font-size:32px;

    font-weight:800;

    color:#166534;

    margin-top:30px;

}



/* Feature cards */

.feature {

    background:white;

    padding:25px;

    border-radius:25px;

    text-align:center;

    box-shadow:
    0 8px 25px rgba(0,0,0,0.06);

}



/* Footer */

.footer {

    text-align:center;

    color:#64748b;

    padding:30px;

}


</style>

""", unsafe_allow_html=True)



# =====================================================
# Header
# =====================================================

st.markdown("""
<div class="hero">

<h1>
🍅 Tomato Leaf Disease Detection
</h1>


<p>
AI-powered system for detecting tomato leaf diseases
using Deep Learning and Computer Vision
</p>


</div>

""", unsafe_allow_html=True)



# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "TomatoLeaf_Model.h5"
    )

    return model



model = load_model()



# =====================================================
# Upload Section
# =====================================================


col1, col2 = st.columns(
    [1,1]
)



with col1:

    st.markdown("""
    <div class="upload-box">

    <h2>
    📤 Upload Tomato Leaf Image
    </h2>

    Upload an image of a tomato leaf
    and let the AI model analyze it.

    </div>

    """,
    unsafe_allow_html=True)



    uploaded_file = st.file_uploader(
        "",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )




with col2:

    st.markdown("""
    <div class="card">


    <h2>
    🧠 Model Information
    </h2>


    <p>
    <b>Architecture:</b>
    MobileNetV2
    </p>


    <p>
    <b>Technique:</b>
    Transfer Learning
    </p>


    <p>
    <b>Framework:</b>
    TensorFlow / Keras
    </p>


    <p>
    <b>Classes:</b>
    </p>

    🍃 Healthy

    <br>

    🦠 Diseased


    </div>

    """,
    unsafe_allow_html=True)




# =====================================================
# Prediction
# =====================================================


if uploaded_file is not None:


    image = Image.open(
        uploaded_file
    ).convert("RGB")



    img = image.resize(
        (224,224)
    )



    img = np.array(
        img
    ).astype(
        "float32"
    ) / 255.0



    img = np.expand_dims(
        img,
        axis=0
    )



    prediction = model.predict(
        img,
        verbose=0
    )[0][0]




    if prediction >= 0.5:


        label = "Healthy 🍃"

        confidence = prediction * 100

        color = "#16a34a"



    else:


        label = "Diseased 🦠"

        confidence = (1-prediction) * 100

        color = "#dc2626"




    st.markdown("---")



    image_col, result_col = st.columns(
        [1,1]
    )



    with image_col:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        st.markdown(
        "</div>",
        unsafe_allow_html=True
        )





    with result_col:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.markdown(
        f"""

        <div class="result"
        style="color:{color};">

        {label}

        </div>

        """,
        unsafe_allow_html=True
        )



        st.markdown(
        f"""

        <div class="confidence-box">

        <h3>
        Confidence Score
        </h3>


        <h1>
        {confidence:.2f}%
        </h1>


        </div>

        """,
        unsafe_allow_html=True
        )



        st.progress(
            int(confidence)
        )



        st.markdown(
        "</div>",
        unsafe_allow_html=True
        )




# =====================================================
# How It Works
# =====================================================


st.markdown("---")


st.markdown(
"""
<div class="section-title">
⚙️ How It Works
</div>
""",
unsafe_allow_html=True
)



c1,c2,c3 = st.columns(3)



with c1:

    st.markdown("""
    <div class="feature">

    📷

    <h3>
    Upload Image
    </h3>

    User provides tomato leaf image

    </div>

    """,
    unsafe_allow_html=True)




with c2:

    st.markdown("""
    <div class="feature">

    🤖

    <h3>
    AI Processing
    </h3>

    Deep learning image analysis

    </div>

    """,
    unsafe_allow_html=True)




with c3:

    st.markdown("""
    <div class="feature">

    📊

    <h3>
    Prediction
    </h3>

    Disease classification result

    </div>

    """,
    unsafe_allow_html=True)




# =====================================================
# Footer
# =====================================================


st.markdown("""
<div class="footer">

🍅 Tomato Leaf Disease Detection

<br>

Deep Learning Based Plant Disease Classification System

</div>

""",
unsafe_allow_html=True)
