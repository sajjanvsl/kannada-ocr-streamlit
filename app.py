import streamlit as st
from PIL import Image
import pytesseract
import os

# Optional: Set Tesseract path if needed (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

st.set_page_config(page_title="Old Kannada to Hosa Kannada", layout="centered")

st.title("📜 Old Kannada to Hosa Kannada Converter")

uploaded_file = st.file_uploader("📤 Upload Old Kannada Manuscript Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Running OCR..."):
        try:
            old_text = pytesseract.image_to_string(image, lang='kan')
            st.text_area("🧠 OCR Output (Old Kannada)", old_text, height=150)

            translit_map = {
                "ಮದುಮಕ್ಕಳಿಗೆ": "ಮಗುಗಳಿಗೆ",
                "ಊಟವಿಲ್ಲ": "ಊಟವಿಲ್ಲ"
            }

            new_text = old_text
            for old, new in translit_map.items():
                new_text = new_text.replace(old, new)

            st.text_area("🔁 Translated (Hosa Kannada)", new_text.strip(), height=150)

        except Exception as e:
            st.error(f"OCR failed: {e}")