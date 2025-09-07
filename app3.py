# -*- coding: utf-8 -*-
"""
Kannada OCR Web App
Author: S P Sajjan
"""

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import pytesseract
import os
import numpy as np
import cv2
import gdown
import zipfile
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ===============================
#  CONFIG
# ===============================
st.set_page_config(page_title="Kannada OCR", layout="centered")

# Tesseract config (only required on Windows, else comment out)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r'C:\\Program Files\\Tesseract-OCR\\tessdata'

# Google Drive Dataset
GOOGLE_DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1G4CNR2WeaRP_s_c7lddnIyoQG2ck4nYm?usp=sharing"
DATASET_DIR = "Dataset"

# ===============================
#  DATASET HANDLING
# ===============================
def extract_folder_id(link: str) -> str:
    """Extract Google Drive folder ID from URL."""
    if "/folders/" in link:
        return link.split("/folders/")[1].split("?")[0]
    return link

@st.cache_resource
def ensure_dataset_folder():
    """Download dataset from Google Drive if not present."""
    folder_id = extract_folder_id(GOOGLE_DRIVE_FOLDER_LINK)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    if not os.path.exists(DATASET_DIR):
        st.info("📥 Downloading dataset from Google Drive...")
        gdown.download_folder(url, output=DATASET_DIR, quiet=False, use_cookies=False)
        st.success("✅ Dataset downloaded!")
    return DATASET_DIR

# ===============================
#  OCR HELPERS
# ===============================
def auto_crop_image(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_points = np.vstack(contours).astype(np.int32)
        x, y, w, h = cv2.boundingRect(all_points)
        return image_np[y:y + h, x:x + w]
    return image_np

def enhance_image(pil_image, method="adaptive"):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 15
        )
    _, binary = cv2.threshold(
        gray, 0,
        255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary

def run_full_ocr(image_array, psm=3):
    config = f"--oem 1 --psm {psm} -l kan"
    return pytesseract.image_to_string(
        Image.fromarray(image_array), config=config
    ).strip()

def convert_old_to_new_kannada(text):
    conversion_map = {
        "ಮದುಮಕ್ಕಳಿಗೆ": "ಮಗುಗಳಿಗೆ",
        "ಪಾಠಶಾಲೆ": "ಶಾಲೆ",
        "ಅಂಗಡಿಗೆ": "ದೂಕಾಣಕ್ಕೆ",
        "ಆಯ್ಕೆಯು": "ಆಯ್ಕೆ",
        "ಅಧ್ಯಾಪಕರು": "ಶಿಕ್ಷಕರು",
        "ಗ್ರಂಥ": "ಪುಸ್ತಕ",
        "ಶಿಕ್ಷÊ3": "ಬೋಧನೆ",
        "ಸಂಗತಿಗಳು": "ಮಾಹಿತಿಗಳು",
        "ನೂತನ": "ಹೊಸ",
        "ಪಾಠ": "ಪಾಠವು",
        "ಬೋಧನೆ": "ಶಿಕ್ಷಣ"
    }
    for old_word, new_word in conversion_map.items():
        text = text.replace(old_word, new_word)
    return text

# ===============================
#  CLASSIFIER PREPARATION
# ===============================
@st.cache_resource
def prepare_classifier():
    dataset_path = ensure_dataset_folder()
    X, y = [], []
    IMG_SIZE = 64
    for folder in os.listdir(dataset_path):
        path = os.path.join(dataset_path, folder)
        if os.path.isdir(path):
            for fname in os.listdir(path):
                try:
                    img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img.flatten())
                    y.append(folder)
                except:
                    continue
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), y_enc, test_size=0.2, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc

# ===============================
#  STREAMLIT UI
# ===============================
st.title("📖 Kannada OCR System")

uploaded_file = st.file_uploader("Upload a Kannada manuscript image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Upload", use_column_width=True)

    # Cropping
    cropped_img = st_cropper(image, realtime_update=True, box_color="#FF0000")
    st.image(cropped_img, caption="✂️ Cropped Image")

    # Auto-cropping
    auto_cropped = auto_crop_image(np.array(cropped_img))
    st.image(auto_cropped, caption="🤖 Auto-Cropped")

    # Enhancement
    method = st.radio("Enhancement method", ["adaptive", "otsu"])
    enhanced_img = enhance_image(Image.fromarray(auto_cropped), method)
    st.image(enhanced_img, caption="⚙️ Enhanced Image")

    # OCR
    psm = st.selectbox("Tesseract PSM Mode", [3, 6, 11, 13], index=0)
    extracted_text = run_full_ocr(enhanced_img, psm)
    st.text_area("🧾 Extracted Text", extracted_text, height=150)

    # Old → Modern Kannada
    modern_text = convert_old_to_new_kannada(extracted_text)
    st.text_area("🔤 Converted Modern Kannada", modern_text, height=150)

    # Manuscript Year Prediction (KNN)
    model, encoder, acc = prepare_classifier()
    st.write(f"📅 Manuscript Year Prediction Accuracy: **{acc:.2f}**")

    # Editable + Feedback
    corrected_text = st.text_area("📝 Edit Corrected Text", modern_text, height=150)
    feedback = st.text_input("💬 Feedback on OCR results")

    # Save feedback
    if st.button("Save Feedback"):
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | {feedback}\n")
        st.success("✅ Feedback saved!")

    # Download
    st.download_button("📥 Download Corrected Text", corrected_text, file_name="corrected_kannada.txt")
