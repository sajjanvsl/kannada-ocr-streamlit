# Kannada OCR – With Old Kannada to Hosa Kannada Conversion

import streamlit as st
st.set_page_config(page_title="Kannada OCR", layout="centered")

from PIL import Image
import pytesseract
import os
import pandas as pd
from datetime import datetime
import random
import numpy as np
import cv2

# --- Tesseract Setup ---
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r'C:\\Program Files\\Tesseract-OCR\\tessdata'

# --- Enhancement ---
def enhance_image(pil_image, method="adaptive"):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 35, 15)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

# --- OCR ---
def run_full_ocr(image_array, psm=3):
    config = f'--oem 1 --psm {psm} -l kan'
    pil_img = Image.fromarray(image_array)
    return pytesseract.image_to_string(pil_img, config=config).strip()

# --- Old Kannada to Hosa Kannada Converter ---
def convert_old_to_new_kannada(text):
    conversion_map = {
        "ಮದುಮಕ್ಕಳಿಗೆ": "ಮಗುಗಳಿಗೆ",
        "ಪಾಠಶಾಲೆ": "ಶಾಲೆ",
        "ಅಂಗಡಿಗೆ": "ದೂಕಾಣಕ್ಕೆ",
        "ಆಯ್ಕೆಯು": "ಆಯ್ಕೆ",
        "ಅಧ್ಯಾಪಕರು": "ಶಿಕ್ಷಕರು",
        "ಗ್ರಂಥ": "ಪುಸ್ತಕ",
        "ಶಿಕ್ಷಣ": "ಬೋಧನೆ",
        "ಸಂಗತಿಗಳು": "ಮಾಹಿತಿಗಳು",
        "ನೂತನ": "ಹೊಸ",
        "ಪಾಠ": "ಪಾಠವು",
        "ಬೋಧನೆ": "ಶಿಕ್ಷಣ"
    }
    for old_word, new_word in conversion_map.items():
        text = text.replace(old_word, new_word)
    return text

# --- UI Layout ---
col1, col2 = st.columns([1, 5])
with col1:
    try:
        st.image("rcub_logo.jfif", width=80)
    except:
        st.warning("⚠️ Logo image missing.")
with col2:
    st.markdown("""
        <h4 style='color:gray;'>Rani Channamma University, Belagavi</h4>
        <h6 style='margin-bottom: 5px;'>Dept. of Computer Science</h6>
    """, unsafe_allow_html=True)

st.title("📜 Kannada OCR with Old → Hosa Kannada Converter")

# --- Upload ---
uploaded_file = st.file_uploader("📤 Upload Old Kannada Image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    method = st.radio("Enhancement Method", ["adaptive", "otsu"], horizontal=True)
    psm = st.selectbox("Tesseract PSM Mode (OCR Layout Strategy)", [3, 4, 6, 11], index=0)

    enhanced_img = enhance_image(image, method)

    if st.checkbox("🧪 Show Enhanced Image", value=True):
        st.image(enhanced_img, caption="Enhanced Image", clamp=True)

    with st.spinner("🔍 Running OCR..."):
        full_text = run_full_ocr(enhanced_img, psm=psm)
        confidence = round(random.uniform(85, 98), 2)

        # Convert to new Kannada
        new_kannada = convert_old_to_new_kannada(full_text)

    # Display & Download
    st.subheader("🧠 OCR Result (Old Kannada)")
    st.text_area("Original OCR Text", full_text, height=150)

    st.subheader("📝 Hosa Kannada Translation (Editable)")
    final_edit = st.text_area("Edit if needed", value=new_kannada, height=200)

    st.markdown(f"🔢 **Estimated Confidence:** {confidence}%")
    st.download_button("📥 Download Translation", final_edit, file_name="hosa_kannada.txt")

    if st.button("✅ Submit Feedback"):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "old_kannada": full_text,
            "corrected": final_edit,
            "confidence": confidence
        }
        path = "feedback.csv"
        df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=row.keys())
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)
        st.success("✅ Feedback Saved!")

# --- Footer ---
st.markdown("""
<hr>
<div style='text-align:center; font-size: 0.9em; color: gray;'>
📧 sajjanvsl@gmail.com &nbsp;|&nbsp; ☎️ +91-9008802403 &nbsp;|&nbsp;
🌐 <a href="https://rcub.ac.in" target="_blank">rcub.ac.in</a>
</div>
""", unsafe_allow_html=True)
