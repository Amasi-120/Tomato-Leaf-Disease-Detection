import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Tomato AI Vision",
    page_icon="🍅",
    layout="wide"
)


# ---------------------------
# CSS Design
# ---------------------------
st.markdown("""
<style>

body {
background-color:#f8fafc;
}

.hero {
background: linear-gradient(135deg,#e63946,#ff8c42);
padding:45px;
border-radius:25px;
color:white;
text-align:center;
margin-bottom:30px;
}

.hero h1 {
font-size:50px;
margin-bottom:10px;
}

.hero p {
font-size:20px;
}


.card {
background:white;
padding:30px;
border-radius:25px;
box-shadow:0 10px 30px rgba(0,0,0,0.08);
margin:15px;
}


.upload-box {
border:2px dashed #e63946;
padding:25px;
border-radius:20px;
text-align:center;
}


.result-title {
font-size:35px;
font-weight:800;
text-align:center;
}


.metric-box {
background:#f1f5f9;
padding:20px;
border-radius:20px;
text-align:center;
}


.section-title {
font-size:28px;
font-weight:700;
color:#1d3557;
}


.footer {
text-align:center;
padding:20px;
color:#64748b;
}


</style>

""", unsafe_allow_html=True)



# ---------------------------
# Header
# ---------------------------

st.markdown("""
<div class="hero">

<h1>🍅 Tomato AI Vision</h1>

<p>
Deep Learning System for Tomato Leaf Disease Detection
</p>

<p>
Powered by MobileNetV2 + Transfer Learning
</p>

</div>
""",
unsafe_allow_html=True)



# ---------------------------
# Load Model
# ---------------------------

@st.cache_resource
def load_model():

    model=tf.keras.models.load_model(
        "TomatoLeaf_Model.h5"
    )

    return model


model=load_model()



# ---------------------------
# Upload Area
# ---------------------------

left,right=st.columns([1,1])


with left:

    st.markdown(
    """
    <div class="card">

    <h2>📤 Upload Leaf Image</h2>

    Upload a tomato leaf image and let AI analyze its health condition.

    </div>
    """,
    unsafe_allow_html=True
    )


    uploaded_file=st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )



with right:

    st.markdown(
    """
    <div class="card">

    <h2>🧠 AI Model</h2>

    <p>
    Model: MobileNetV2
    </p>

    <p>
    Task: Binary Classification
    </p>

    <p>
    Classes:
    🍃 Healthy
    🦠 Diseased
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )



# ---------------------------
# Prediction
# ---------------------------

if uploaded_file:


    image=Image.open(uploaded_file).convert("RGB")


    img=image.resize((224,224))

    img=np.array(img).astype("float32")/255

    img=np.expand_dims(img,axis=0)


    prediction=model.predict(
        img,
        verbose=0
    )[0][0]



    if prediction>=0.5:

        result="Healthy 🍃"
        confidence=prediction*100
        color="#16a34a"


    else:

        result="Diseased 🦠"
        confidence=(1-prediction)*100
        color="#dc2626"



    st.markdown("---")


    col1,col2=st.columns(2)



    with col1:

        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

        st.markdown(
        '</div>',
        unsafe_allow_html=True
        )



    with col2:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.markdown(
        f"""

        <div class="result-title"
        style="color:{color}">
        {result}
        </div>

        """,
        unsafe_allow_html=True
        )


        st.write("")


        st.markdown(
        f"""
        <div class="metric-box">

        <h2>
        Confidence
        </h2>

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
        '</div>',
        unsafe_allow_html=True
        )



# ---------------------------
# How it works
# ---------------------------


st.markdown("---")


st.markdown(
"""
<div class="section-title">
⚙️ How It Works
</div>
""",
unsafe_allow_html=True
)


c1,c2,c3=st.columns(3)


with c1:
    st.info(
    """
    📷
    Upload Image

    User uploads tomato leaf image
    """
    )


with c2:
    st.info(
    """
    🤖
    AI Analysis

    MobileNetV2 processes image
    """
    )


with c3:
    st.info(
    """
    📊
    Prediction

    Disease classification result
    """
    )



st.markdown(
"""
<div class="footer">

🍅 Tomato Leaf Disease Detection  
Developed by Amasi Al-Sahbi  
Artificial Intelligence Student

</div>
""",
unsafe_allow_html=True
)
