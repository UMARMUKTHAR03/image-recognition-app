import streamlit as st
from PIL import Image
from clf import predict

st.set_page_config(
    page_title="Image Recognition App • Deep Learning Vision Studio",
    page_icon="👁️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #06b6d4, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #0f172a;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">👁️ Image Recognition App — Deep Learning Vision Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Engineered by <b>Umar Mukthar</b> • Pre-trained PyTorch ResNet-101 (ImageNet 1,000 Categories)</p>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Image Ingestion & Input Feed")
    file_up = st.file_uploader("Upload an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    
    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption=f"Target: {file_up.name} ({image.size[0]}×{image.size[1]} px)", use_container_width=True)
    else:
        st.info("👆 Please upload an image above to run real-time PyTorch ResNet-101 classification.")

with col2:
    st.subheader("📊 Top-5 ImageNet Predictions")
    
    if file_up is not None:
        with st.spinner("Executing ResNet-101 forward inference pass..."):
            labels = predict(file_up)

        top_label, top_score = labels[0]
        st.success(f"🎯 **Top Prediction:** `{top_label}` with **{top_score:.2f}%** confidence.")

        st.markdown("### Confidence Distribution")
        for rank, (name, score) in enumerate(labels, 1):
            with st.container():
                st.write(f"**{rank}. {name}** — `{score:.2f}%`")
                st.progress(min(max(float(score) / 100.0, 0.01), 1.0))
                
        with st.expander("🔍 ConvNet Architectural Telemetry"):
            st.markdown("""
            - **Model Backbone:** PyTorch `torchvision.models.resnet101`
            - **Layers:** 101 Deep Convolutional Residual Layers
            - **Input Tensor Dimensions:** `[1, 3, 224, 224]`
            - **Normalization Coefficients:** `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
            - **Target Output Dimension:** 1,000 Classes Softmax
            """)
    else:
        st.write("Awaiting image upload to compute classification probabilities...")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Image Recognition App • Designed & Engineered by <b>Umar Mukthar</b> • "
    "<a href='https://github.com/UMARMUKTHAR03/image-recognition-app' target='_blank'>GitHub Repository</a> • MIT License"
    "</div>",
    unsafe_allow_html=True
)
