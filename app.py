import streamlit as st
import os
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# --- FEATURE EXTRACTION FUNCTION ---
def extract_chroma(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return np.mean(chroma.T, axis=0)
    except Exception as e:
        st.error(f"Error parsing audio track: {e}")
        return None

# --- WEB APPLICATION LAYOUT ---
st.set_page_config(page_title="Audio Forgery Detector", layout="centered")

st.title("🛡️ Audio Forgery Detection Platform")
st.write("Upload a `.wav` file to analyze its structural authenticity using your trained machine learning architectures.")

# 1. File Upload UI Module
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

# 2. Dropdown Selector UI Module
model_option = st.selectbox(
    "Select Classification Pipeline Architecture:",
    ("Traditional Model (Chroma + Random Forest)", "Hybrid Model (1D CNN + SVM)")
)

if uploaded_file is not None:
    # Audio Playback Widget
    st.audio(uploaded_file, format="audio/wav")
    
    # Temporarily store the uploaded file to disk for librosa to process
    temp_path = "temp_uploaded_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("Analyze Audio File"):
        with st.spinner("Extracting acoustic feature configurations..."):
            features = extract_chroma(temp_path)
            
        if features is not None:
            # --- MODEL PIPELINE A: TRADITIONAL ---
            if model_option == "Traditional Model (Chroma + Random Forest)":
                if os.path.exists('random_forest_model.pkl'):
                    with open('random_forest_model.pkl', 'rb') as f:
                        rf_model = pickle.load(f)
                    
                    prediction = rf_model.predict(features.reshape(1, -1))[0]
                    
                    st.subheader("Analysis Results:")
                    if prediction == 0:
                        st.success("✅ **Result: Genuine (Bona-fide) Audio**")
                    else:
                        st.error("🚨 **Result: Forged (Spoofed) Audio Detected**")
                else:
                    st.warning("Model file 'random_forest_model.pkl' not found. Please run train_traditional.py first.")

            # --- MODEL PIPELINE B: HYBRID ---
            else:
                if os.path.exists('cnn_feature_extractor.h5') and os.path.exists('svm_classifier.pkl'):
                    cnn_extractor = load_model('cnn_feature_extractor.h5')
                    with open('svm_classifier.pkl', 'rb') as f:
                        svm_backend = pickle.load(f)
                        
                    # Prepare array structure for 1D CNN: shape (1, 12, 1)
                    hybrid_input = np.expand_dims(np.expand_dims(features, axis=0), axis=-1)
                    
                    # Extract features and predict
                    deep_embeddings = cnn_extractor.predict(hybrid_input)
                    prediction = svm_backend.predict(deep_embeddings)[0]
                    
                    st.subheader("Analysis Results:")
                    if prediction == 0:
                        st.success("✅ **Result: Genuine (Bona-fide) Audio**")
                    else:
                        st.error("🚨 **Result: Forged (Spoofed) Audio Detected**")
                else:
                    st.warning("Hybrid model components not found. Please run train_hybrid.py first.")
                    
    # Clean up temporary disk file
    if os.path.exists(temp_path):
        os.remove(temp_path)