import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Tomato Vision AI",
    page_icon="🍅",
    layout="wide"
)


# ==========================
# Premium CSS
# ==========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        180deg,
        #f8fafc,
        #ffffff
    );
}


.hero {
    background:
    linear-gradient(
        135deg,
        #166534,
        #dc2626
    );

    padding:55px;
    border-radius:30px;
    color:white;
    text-align:center;
}


.hero h1 {
    font-size:55px;
    font-weight:900;
}


.hero p {
    font-size:22px;
}



.card {

    background:white;

    padding:35px;

    border-radius:30px;

    box-shadow:
    0px 15px 35px
    rgba(0,0,0,0.08);

}



.upload-card {

    border:3px dashed #16a34a;

    border-radius:25px;

    padding:30px;

    background:#f0fdf4;

}



.result {

    text-align:center;

    font-size:42px;

    font-weight:900;

}



.confidence {

    background:#f8fafc;

    padding:25px;

    border-radius:25px;

    text-align:center;

}



.section {

    font-size:32px;

    font-weight:800;

    color:#166534;

}



.small-card {

    background:white;

    padding:25px;

    border-radius:25px;

    text-align:center;

    box-shadow:
    0px 8px 25px
    rgba(0,0,0,0.06);

}


.footer {

    text-align:center;

    color:#64748b;

    padding:30px;

}


</style>

""",
unsafe_allow_html=True)



# ==========================
# Header
# ==========================

st.markdown("""
<div class="hero">

<h1>🍅 Tomato Vision AI</h1>

<p>
Intelligent Tomato Leaf Disease Detection System
</p>

<p>
Powered by Deep Learning & Computer Vision
</p>

</div>
""",
unsafe_allow_html=True)



# ==========================
# Model Loading
# ==========================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "TomatoLeaf_Model.h5"
    )


model = load_model()



# ==========================
# Main Section
# ==========================


left,right = st.columns(
    [1,1]
)



with left:

    st.markdown(
    """
    <div class="upload-card">

    <h2>
    📤 Upload Image
    </h2>

    Upload a tomato leaf image to analyze
    its health condition using AI.

    </div>
    """,
    unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )



with right:

    st.markdown(
    """
    <div class="card">

    <h2>
    🧠 AI Model
    </h2>


    <p>
    <b>Architecture:</b>
    MobileNetV2
    </p>


    <p>
    <b>Method:</b>
    Transfer Learning
    </p>


    <p>
    <b>Classification:</b>
    </p>

    🍃 Healthy

    <br>

    🦠 Diseased


    </div>

    """,
    unsafe_allow_html=True
    )



# ==========================
# Prediction
# ==========================

if uploaded_file:


    image = Image.open(
        uploaded_file
    ).convert("RGB")


    processed = image.resize(
        (224,224)
    )


    processed = np.array(
        processed
    ).astype(
        "float32"
    ) / 255.0


    processed = np.expand_dims(
        processed,
        axis=0
    )


    prediction = model.predict(
        processed,
        verbose=0
    )[0][0]



    if prediction >= 0.5:

        label = "Healthy 🍃"

        confidence = prediction * 100

        color="#16a34a"


    else:

        label = "Diseased 🦠"

        confidence = (1-prediction)*100

        color="#dc2626"



    st.markdown("---")



    img_col,result_col = st.columns(
        [1,1]
    )


    with img_col:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.image(
            image,
            caption="Analyzed Image",
            use_container_width=True
        )


        st.markdown(
        '</div>',
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
        style="color:{color}">
        {label}
        </div>
        """,
        unsafe_allow_html=True
        )


        st.markdown(
        f"""
        <div class="confidence">

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
        '</div>',
        unsafe_allow_html=True
        )



# ==========================
# Features
# ==========================


st.markdown("---")


st.markdown(
"""
<div class="section">
✨ System Features
</div>
""",
unsafe_allow_html=True
)



a,b,c = st.columns(3)



with a:

    st.markdown(
    """
    <div class="small-card">

    📷

    <h3>
    Image Analysis
    </h3>

    AI-based image processing

    </div>

    """,
    unsafe_allow_html=True
    )


with b:

    st.markdown(
    """
    <div class="small-card">

    🤖

    <h3>
    Deep Learning
    </h3>

    MobileNetV2 classification

    </div>

    """,
    unsafe_allow_html=True
    )


with c:

    st.markdown(
    """
    <div class="small-card">

    📊

    <h3>
    Fast Prediction
    </h3>

    Real-time disease detection

    </div>

    """,
    unsafe_allow_html=True
    )



st.markdown(
"""
<div class="footer">

🍅 Tomato Vision AI  
Deep Learning Based Plant Disease Detection

</div>
""",
unsafe_allow_html=True
)
