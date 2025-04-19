import streamlit as st
from PIL import Image
import pytesseract
import os
from io import BytesIO
import base64
import pandas as pd
from datetime import datetime

# Optional: Set Tesseract path if needed (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r'C:\\Program Files\\Tesseract-OCR\\tessdata'

st.set_page_config(page_title="Old Kannada to Hosa Kannada Converter", layout="centered")

# Sticky header and custom theme colors
st.markdown("""
    <style>
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #e6e6fa;
        padding: 10px 0;
        text-align: center;
        border-bottom: 2px solid #4B0082;
        font-size: 1.1em;
        color: #4B0082;
        font-weight: bold;
    }
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #4B0082;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #333;
        margin-bottom: 2em;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        margin-top: 3em;
        color: #666;
    }
    </style>
    <div class="sticky-header">
        <img src="https://upload.wikimedia.org/wikipedia/en/4/45/Rani_Channamma_University_logo.png" width="80">
        <br>
        Developed and hosted by Dept. of Computer Science, Rani Channamma University, Belagavi
    </div>
    <div class="title">📜 Old Kannada to Hosa Kannada Converter</div>
    <div class="subtitle">Upload an inscription image to convert ancient Kannada to modern Kannada script using OCR and transliteration.</div>
""", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("📤 Upload Old Kannada Manuscript Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Running OCR and Translating to Hosa Kannada..."):
        try:
            old_text = pytesseract.image_to_string(image, lang='kan')
            st.markdown("### 🧠 OCR Output (Old Kannada)")
            st.text_area("ocr_output", old_text, height=150)

            translit_map = {
                "ಮದುಮಕ್ಕಳಿಗೆ": "ಮಗುಗಳಿಗೆ",
                "ಊಟವಿಲ್ಲ": "ಊಟವಿಲ್ಲ"
            }

            new_text = old_text
            for old, new in translit_map.items():
                new_text = new_text.replace(old, new)

            st.markdown("### 🔁 Translated (Hosa Kannada)")
            corrected_text = st.text_area("Correct if needed:", new_text.strip(), height=150)

            # Download button
            b64 = base64.b64encode(corrected_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="translated_kannada.txt">📥 Download Translated Text</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Feedback submission (stores in CSV)
            if st.button("Submit Feedback/Correction"):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                feedback_data = pd.DataFrame([[now, old_text.strip(), corrected_text.strip()]],
                                             columns=["timestamp", "old_kannada", "hosa_kannada"])
                if os.path.exists("feedback.csv"):
                    feedback_data.to_csv("feedback.csv", mode='a', header=False, index=False)
                else:
                    feedback_data.to_csv("feedback.csv", index=False)
                st.success("✅ Thank you! Your correction has been saved for model improvement.")

        except Exception as e:
            st.error(f"OCR failed: {e}")
else:
    st.info("👆 Please upload an image to begin.")

# Footer with contact info and social links
st.markdown("""
    <div class="footer">
        📧 Contact: csdept@rcub.ac.in | ☎️ +91-9008802403<br>
        🌐 <a href="https://rcub.ac.in" target="_blank">rcub.ac.in</a> |
        📘 <a href="https://facebook.com/rcubelgaum" target="_blank">Facebook</a>
    </div>
""", unsafe_allow_html=True)
