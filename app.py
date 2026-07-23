"""Streamlit app for Dog Breed Classification.

Run with:
    streamlit run app.py
"""

import os
from typing import Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
from PIL import Image

# Suppress TensorFlow and TensorFlow Hub logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dog_breed_model.keras")
BREEDS_PATH = os.path.join(BASE_DIR, "dog_breeds.npy")
IMG_SIZE = 224


@st.cache_resource(show_spinner=True)
def load_model_and_labels() -> Tuple[keras.Model, np.ndarray]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train and save the model by running main.py first."
        )
    if not os.path.exists(BREEDS_PATH):
        raise FileNotFoundError(
            f"Breed label file not found at {BREEDS_PATH}. Train and save the model by running main.py first."
        )

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"KerasLayer": hub.KerasLayer},
    )
    breeds = np.load(BREEDS_PATH, allow_pickle=True)
    return model, breeds


def preprocess_image(image: Image.Image) -> tf.Tensor:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype("float32") / 255.0
    return tf.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)


def main() -> None:
    # Page configuration
    st.set_page_config(
        page_title="Dog Breed Classifier AI",
        page_icon="🐶",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Premium AI SaaS UI Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        * {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            box-sizing: border-box;
        }

        /* App Background */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 40%, #f1f5f9 100%) !important;
            color: #0f172a;
        }

        /* Main Container Spacing */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1080px !important;
        }

        /* Header / Navigation elements cleanup */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Hero Section Container */
        .hero-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(238, 242, 255, 0.75) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.9);
            border-radius: 28px;
            padding: 3rem 2rem;
            text-align: center;
            box-shadow: 0 20px 40px -15px rgba(99, 102, 241, 0.12), 0 0 0 1px rgba(99, 102, 241, 0.05);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            animation: fadeInDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: #4F46E5;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 0.4rem 1.1rem;
            border-radius: 50px;
            margin-bottom: 1.25rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: 3.25rem;
            font-weight: 800;
            line-height: 1.15;
            background: linear-gradient(135deg, #1E1B4B 0%, #4F46E5 40%, #8B5CF6 70%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.85rem;
            letter-spacing: -0.025em;
        }

        .hero-subtitle {
            font-size: 1.15rem;
            color: #475569;
            font-weight: 500;
            max-width: 620px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* Glass Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.85);
            box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.06), 0 4px 12px rgba(0, 0, 0, 0.02);
            border-radius: 24px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-card:hover {
            box-shadow: 0 18px 38px -10px rgba(99, 102, 241, 0.12), 0 8px 20px rgba(0, 0, 0, 0.03);
            transform: translateY(-2px);
        }

        /* STREAMLIT FILE UPLOADER SPECIFIC SCOPED STYLING */
        div[data-testid="stFileUploader"] {
            width: 100% !important;
            margin-bottom: 1.5rem !important;
        }

        /* Main Dropzone Section Container */
        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploader"] > section {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 2px dashed #818CF8 !important;
            border-radius: 24px !important;
            padding: 2.5rem 1.75rem !important;
            text-align: center !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 1rem !important;
            cursor: pointer !important;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.08) !important;
        }

        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover,
        div[data-testid="stFileUploader"] > section:hover {
            border-color: #4F46E5 !important;
            background: rgba(255, 255, 255, 0.98) !important;
            box-shadow: 0 18px 36px -10px rgba(99, 102, 241, 0.18), 0 0 0 4px rgba(99, 102, 241, 0.08) !important;
            transform: translateY(-2px);
        }

        /* SVG Upload Icon */
        div[data-testid="stFileUploader"] section svg {
            width: 48px !important;
            height: 48px !important;
            color: #6366F1 !important;
            fill: #6366F1 !important;
            margin-bottom: 0.25rem !important;
            transition: transform 0.3s ease !important;
        }

        div[data-testid="stFileUploader"] section:hover svg {
            transform: scale(1.12) translateY(-2px);
            color: #4F46E5 !important;
            fill: #4F46E5 !important;
        }

        /* Uploader Text inside dropzone */
        div[data-testid="stFileUploader"] section span,
        div[data-testid="stFileUploader"] section p,
        div[data-testid="stFileUploader"] section div {
            color: #0F172A !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            text-align: center !important;
        }

        div[data-testid="stFileUploader"] section small {
            color: #64748B !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            text-align: center !important;
        }

        /* Inner Browse Button inside Dropzone ONLY */
        div[data-testid="stFileUploaderDropzone"] button,
        div[data-testid="stFileUploader"] section button[data-testid="stBaseButton-secondary"] {
            background: #EEF2FF !important;
            color: #4F46E5 !important;
            border: 1px solid #C7D2FE !important;
            border-radius: 12px !important;
            padding: 0.5rem 1.4rem !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1) !important;
            transition: all 0.25s ease !important;
            width: auto !important;
            min-height: auto !important;
            height: auto !important;
            margin: 0.5rem auto 0 auto !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div[data-testid="stFileUploaderDropzone"] button:hover,
        div[data-testid="stFileUploader"] section button[data-testid="stBaseButton-secondary"]:hover {
            background: #4F46E5 !important;
            color: #FFFFFF !important;
            border-color: #4F46E5 !important;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.3) !important;
            transform: translateY(-1px) !important;
        }

        div[data-testid="stFileUploaderDropzone"] button p,
        div[data-testid="stFileUploader"] section button[data-testid="stBaseButton-secondary"] p {
            color: inherit !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            margin: 0 !important;
        }

        /* LIGHT PREVIEW FILE CARD (NO BLACK BOXES) */
        div[data-testid="stFileUploaderFile"],
        div[data-testid="stFileUploaderFileData"],
        section[data-testid="stFileUploaderFileData"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            padding: 0.85rem 1.25rem !important;
            margin-top: 0.75rem !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03) !important;
            color: #0F172A !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
        }

        div[data-testid="stFileUploaderFile"] *,
        div[data-testid="stFileUploaderFileData"] * {
            color: #0F172A !important;
            background: transparent !important;
        }

        div[data-testid="stFileUploaderFileName"] {
            color: #0F172A !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
        }

        /* ELEGANT COMPACT REMOVE (X) BUTTON */
        div[data-testid="stFileUploaderFile"] button,
        button[data-testid="stFileUploaderDeleteFile"],
        button[aria-label="Remove file"] {
            background: #F1F5F9 !important;
            color: #64748B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            min-width: 32px !important;
            min-height: 32px !important;
            max-width: 32px !important;
            max-height: 32px !important;
            padding: 0 !important;
            margin: 0 0 0 0.75rem !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            flex-shrink: 0 !important;
        }

        div[data-testid="stFileUploaderFile"] button:hover,
        button[data-testid="stFileUploaderDeleteFile"]:hover,
        button[aria-label="Remove file"]:hover {
            background: #FEE2E2 !important;
            color: #EF4444 !important;
            border-color: #FCA5A5 !important;
            transform: scale(1.08) !important;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2) !important;
        }

        div[data-testid="stFileUploaderFile"] button svg,
        button[data-testid="stFileUploaderDeleteFile"] svg,
        button[aria-label="Remove file"] svg {
            width: 16px !important;
            height: 16px !important;
            fill: currentColor !important;
            color: currentColor !important;
        }

        /* ADD FILE (+) BUTTON STYLING (IF PRESENT) */
        button[data-testid="stFileUploaderAddFile"] {
            background: #EEF2FF !important;
            color: #4F46E5 !important;
            border: 1px solid #C7D2FE !important;
            border-radius: 12px !important;
            padding: 0.4rem 0.85rem !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            width: auto !important;
            height: auto !important;
            min-width: auto !important;
            min-height: auto !important;
            box-shadow: none !important;
        }

        button[data-testid="stFileUploaderAddFile"]:hover {
            background: #4F46E5 !important;
            color: #FFFFFF !important;
        }

        /* DIRECT ANALYZE BREED BUTTON TARGETING (OVERRIDE STREAMLIT DARK THEME) */
        div[data-testid="stButton"] > button,
        div.stButton > button,
        div[data-testid="stButton"] button[kind="secondary"],
        div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #8B5CF6 100%) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 1.15rem !important;
            padding: 0.95rem 2.5rem !important;
            border-radius: 16px !important;
            border: none !important;
            box-shadow: 0 10px 25px -4px rgba(79, 70, 229, 0.45) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            letter-spacing: 0.02em !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        /* FORCE ALL TEXT INSIDE ANALYZE BUTTON TO BE PURE WHITE */
        div[data-testid="stButton"] > button *,
        div.stButton > button *,
        div[data-testid="stButton"] > button p,
        div[data-testid="stButton"] > button span,
        div[data-testid="stButton"] > button div {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 1.15rem !important;
        }

        div[data-testid="stButton"] > button:hover,
        div.stButton > button:hover,
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #4338CA 0%, #4F46E5 50%, #7C3AED 100%) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            box-shadow: 0 16px 35px -4px rgba(99, 102, 241, 0.6) !important;
            transform: translateY(-3px) scale(1.01) !important;
        }

        div[data-testid="stButton"] > button:active,
        div.stButton > button:active {
            transform: translateY(-1px) scale(0.98) !important;
        }

        /* Streamlit Progress Bar Enhancement */
        div[data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 50%, #06B6D4 100%) !important;
            border-radius: 12px !important;
        }
        
        div[data-testid="stProgress"] {
            height: 10px !important;
            border-radius: 12px !important;
            background-color: rgba(226, 232, 240, 0.8) !important;
        }

        /* Custom Keyframe Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero Section
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">
                ✨ Deep Learning Vision Intelligence
            </div>
            <h1 class="hero-title">🐶 Dog Breed Classifier AI</h1>
            <p class="hero-subtitle">
                Identify dog breeds with high precision using MobileNetV2 and TensorFlow transfer learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model and Labels Loading
    model = None
    breeds = None
    load_error = None
    try:
        model, breeds = load_model_and_labels()
    except Exception as e:  # noqa: BLE001
        load_error = str(e)

    if load_error:
        st.error(load_error)
        st.stop()

    # Upload Section Header Card
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.25rem;">
            <div style="width: 64px; height: 64px; margin: 0 auto 0.85rem auto; background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.12)); border-radius: 20px; border: 1px solid rgba(99, 102, 241, 0.2); display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 8px 20px rgba(99, 102, 241, 0.08);">
                📤
            </div>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin: 0 0 0.35rem 0;">Drag & Drop your dog image here</h3>
            <p style="font-size: 0.9rem; color: #64748B; margin: 0;">or click below to browse from your device • JPG, JPEG, or PNG (Max 200MB)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a dog photo (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Image Preview & Information Cards
        col1, col2 = st.columns([1.1, 0.9], gap="medium")

        with col1:
            st.markdown(
                """
                <div class="glass-card" style="padding: 1.25rem;">
                    <div style="font-size: 0.9rem; font-weight: 700; color: #475569; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.4rem;">
                        📸 Image Preview
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            file_size_bytes = uploaded_file.size
            if file_size_bytes < 1024 * 1024:
                file_size_str = f"{file_size_bytes / 1024:.1f} KB"
            else:
                file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"

            st.markdown(
                f"""
                <div class="glass-card" style="height: calc(100% - 1.5rem); display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem;">
                            <span style="background: rgba(99, 102, 241, 0.1); padding: 0.4rem 0.6rem; border-radius: 10px; color: #4F46E5;">📋</span> Image Information
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.85rem; background: rgba(241, 245, 249, 0.7); border-radius: 14px;">
                                <span style="font-size: 0.875rem; color: #64748B; font-weight: 600;">Resolution</span>
                                <span style="font-size: 0.9rem; color: #0F172A; font-weight: 700;">{image.width} × {image.height} px</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.85rem; background: rgba(241, 245, 249, 0.7); border-radius: 14px;">
                                <span style="font-size: 0.875rem; color: #64748B; font-weight: 600;">Format</span>
                                <span style="font-size: 0.9rem; color: #0F172A; font-weight: 700;">{image.format or 'JPEG'}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.85rem; background: rgba(241, 245, 249, 0.7); border-radius: 14px;">
                                <span style="font-size: 0.875rem; color: #64748B; font-weight: 600;">File Size</span>
                                <span style="font-size: 0.9rem; color: #0F172A; font-weight: 700;">{file_size_str}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0.85rem; background: rgba(241, 245, 249, 0.7); border-radius: 14px;">
                                <span style="font-size: 0.875rem; color: #64748B; font-weight: 600;">Color Space</span>
                                <span style="font-size: 0.9rem; color: #0F172A; font-weight: 700;">RGB</span>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 1.25rem; text-align: center; background: rgba(99, 102, 241, 0.08); padding: 0.65rem; border-radius: 14px; font-size: 0.85rem; color: #4F46E5; font-weight: 700;">
                        ✓ Ready for Deep Learning Analysis
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Analyze Button
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            analyze_clicked = st.button("✨ Analyze Dog Breed ✨", use_container_width=True)

        if analyze_clicked:
            # Animated Loading Experience
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown(
                    """
                    <div class="glass-card" style="text-align: center; padding: 2.5rem 1.5rem; margin-top: 1rem;">
                        <div style="width: 50px; height: 50px; border: 4px solid rgba(99, 102, 241, 0.15); border-top: 4px solid #6366F1; border-radius: 50%; animation: spin 0.9s linear infinite; margin: 0 auto 1.25rem auto;"></div>
                        <h4 style="font-size: 1.25rem; font-weight: 800; color: #1E293B; margin-bottom: 0.4rem;">🧠 AI is analyzing your image...</h4>
                        <p style="color: #64748B; font-size: 0.9rem;">Extracting visual features & computing classification probabilities</p>
                    </div>
                    <style>
                        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

            # Model Inference
            input_tensor = preprocess_image(image)
            preds = model.predict(input_tensor, verbose=0)[0]  # shape (num_classes,)

            # Top 5 predictions
            top_k = 5
            top_indices = preds.argsort()[::-1][:top_k]
            top_breeds = breeds[top_indices]
            top_scores = preds[top_indices]

            loading_placeholder.empty()

            # Top Prediction Hero Card
            top_breed_raw = top_breeds[0]
            top_breed_formatted = str(top_breed_raw).replace("_", " ").title()
            top_score_pct = float(top_scores[0]) * 100

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #8B5CF6 100%); backdrop-filter: blur(16px); border-radius: 26px; padding: 2.25rem; color: white; box-shadow: 0 20px 45px -10px rgba(79, 70, 229, 0.4); margin-top: 1.5rem; margin-bottom: 2rem; position: relative; overflow: hidden;">
                    <div style="position: absolute; right: -15px; bottom: -25px; font-size: 10rem; opacity: 0.12; user-select: none; pointer-events: none;">🐕</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
                        <div style="background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3); padding: 0.35rem 1rem; border-radius: 50px; font-size: 0.85rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase;">
                            🏆 Top Prediction
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3); padding: 0.35rem 1rem; border-radius: 50px; font-size: 0.85rem; font-weight: 700;">
                            Confidence Score: {top_score_pct:.1f}%
                        </div>
                    </div>
                    <h2 style="font-size: 2.5rem; font-weight: 800; margin: 0 0 0.5rem 0; color: #FFFFFF; letter-spacing: -0.025em;">{top_breed_formatted}</h2>
                    <div style="background: rgba(0, 0, 0, 0.15); border-radius: 12px; height: 12px; width: 100%; overflow: hidden; margin-top: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.25);">
                        <div style="background: linear-gradient(90deg, #38BDF8 0%, #A855F7 100%); height: 100%; width: {top_score_pct:.1f}%; border-radius: 12px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top 5 Predictions Section
            st.markdown(
                """
                <div style="margin-bottom: 1.25rem; margin-top: 1rem;">
                    <h3 style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
                        📊 Top 5 Predictions
                    </h3>
                    <p style="font-size: 0.9rem; color: #64748B; margin: 0;">Probability distribution across top candidates</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            badge_colors = [
                "linear-gradient(135deg, #6366F1, #4F46E5)",
                "linear-gradient(135deg, #8B5CF6, #7C3AED)",
                "linear-gradient(135deg, #06B6D4, #0891B2)",
                "linear-gradient(135deg, #64748B, #475569)",
                "linear-gradient(135deg, #94A3B8, #64748B)",
            ]

            for i, (breed_raw, score) in enumerate(zip(top_breeds, top_scores), 1):
                breed_fmt = str(breed_raw).replace("_", " ").title()
                pct = float(score) * 100
                badge_bg = badge_colors[min(i - 1, len(badge_colors) - 1)]

                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 1.15rem 1.5rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.65rem; flex-wrap: wrap; gap: 0.5rem;">
                            <div style="display: flex; align-items: center; gap: 0.85rem;">
                                <span style="background: {badge_bg}; color: white; font-weight: 800; font-size: 0.85rem; padding: 0.3rem 0.75rem; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);">#{i}</span>
                                <span style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">{breed_fmt}</span>
                            </div>
                            <div style="font-size: 0.95rem; font-weight: 800; color: #4F46E5; background: rgba(99, 102, 241, 0.08); padding: 0.3rem 0.85rem; border-radius: 10px;">
                                {pct:.1f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(score))

    else:
        # Empty State
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 3.5rem 2rem; margin-top: 1rem;">
                <div style="width: 80px; height: 80px; margin: 0 auto 1.5rem auto; background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2.75rem; box-shadow: 0 10px 25px rgba(99, 102, 241, 0.1);">
                    🐶
                </div>
                <h3 style="font-size: 1.5rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem;">Upload an image to begin</h3>
                <p style="color: #64748B; font-size: 0.95rem; max-width: 440px; margin: 0 auto 1.5rem auto; line-height: 1.5;">Select or drag & drop a clear image of a dog to analyze its breed with MobileNetV2 Deep Learning model.</p>
                <div style="display: flex; justify-content: center; gap: 1.5rem; font-size: 0.85rem; color: #6366F1; font-weight: 600; flex-wrap: wrap;">
                    <span>✦ Supports JPG, JPEG, PNG</span>
                    <span>✦ 120+ Dog Breeds</span>
                    <span>✦ Instant Deep Learning Prediction</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
        <div style="margin-top: 4rem; padding: 2rem 1rem; border-top: 1px solid rgba(226, 232, 240, 0.8); text-align: center; color: #64748B; font-size: 0.875rem;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; font-weight: 600; color: #475569;">
                <span>Built with</span>
                <span style="background: rgba(99, 102, 241, 0.1); color: #4F46E5; padding: 0.2rem 0.65rem; border-radius: 8px; font-size: 0.8rem;">TensorFlow</span>
                <span>•</span>
                <span style="background: rgba(139, 92, 246, 0.1); color: #7C3AED; padding: 0.2rem 0.65rem; border-radius: 8px; font-size: 0.8rem;">TensorFlow Hub</span>
                <span>•</span>
                <span style="background: rgba(6, 182, 212, 0.1); color: #0891B2; padding: 0.2rem 0.65rem; border-radius: 8px; font-size: 0.8rem;">MobileNetV2</span>
                <span>•</span>
                <span style="background: rgba(244, 63, 94, 0.1); color: #E11D48; padding: 0.2rem 0.65rem; border-radius: 8px; font-size: 0.8rem;">Streamlit</span>
            </div>
            <p style="margin: 0; font-weight: 500;">Made with ❤️ by <strong>Shlok Yadav</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
