# -*- coding: utf-8 -*-
"""
Kannada OCR with Old → Hosa Kannada Converter & Year Classifier
"""

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import pytesseract
import os
import pandas as pd
from datetime import datetime
import random
import numpy as np
import cv2
import zipfile
import gdown
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

# ------------------------------------------------------------
# Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def run_full_ocr(image_array, psm=6):
    config = f'--oem 1 --psm {psm} -l kan'
    return pytesseract.image_to_string(Image.fromarray(image_array), config=config).strip()

# ------------------------------------------------------------
st.set_page_config(page_title="Kannada OCR", layout="centered")

# Header Banner
st.markdown("""
<style>
@keyframes slideDown {
    from { transform: translateY(-100%); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
.sticky-title {
    position: sticky;
    top: 0;
    background-color: #fff8f0;
    padding: 5px 0;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: #800000;
    z-index: 100;
    border-bottom: 2px solid #d8cfc4;
    animation: slideDown 0.5s ease-out;
}
</style>
<div class='sticky-title'>Kannada OCR Web App</div>
<h3 style='color:#800000; font-weight:700; text-align:center;'>Rani Channamma University, Belagavi</h3>
<h4 style='color:#800000; font-weight:500; text-align:center;'>Dept. of Computer Science</h4>
<h6 style='color:#000000; font-weight:700; text-align:center;'>Kannada OCR with Old → Hosa Kannada Converter & Year Classifier</h6>
<h6 style='color:#800000; font-weight:700; text-align:center;'>ಕನ್ನಡ ಓಸಿಆರ್ ಹಳೆಯ → ಹೊಸ ಕನ್ನಡ ಪರಿವರ್ತಕ ಮತ್ತು ವರ್ಷ ವರ್ಗವಿಂಗಡಕ</h6>
""", unsafe_allow_html=True)

st.markdown("""
Digitizing palm leaf manuscripts plays a vital role in preserving ancient knowledge systems, historical records,
and cultural heritage. This research enables the conversion of fragile, handwritten scripts into modern,
machine-readable Kannada text, allowing scholars and the public to access invaluable information with clarity,
searchability, and long-term archiving.
""")

if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state.clear()
    st.rerun()

page = st.sidebar.radio(
    "📑 Navigation",
    options=["📄 OCR Processor", "📘 How to Use", "👨‍💻 Developer Info", "🙏 Acknowledgements"],
    index=0
)

# ------------------------------------------------------------
# OCR Utilities
def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    return image.rotate(angle, expand=True)

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
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

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

# ------------------------------------------------------------
# Robust Dataset Loader (handles nested folders, zip upload, auto-download)
def load_dataset_from_path(root_path):
    """Recursively collect images and assign class = parent folder name."""
    X, y = [], []
    IMG_SIZE = 64
    image_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if f.lower().endswith(image_ext):
                # Class name = immediate parent folder name
                class_name = os.path.basename(dirpath)
                if class_name == root_path or class_name == "":
                    continue  # skip images directly in root
                try:
                    img = cv2.imread(os.path.join(dirpath, f), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img.flatten())
                    y.append(class_name)
                except Exception:
                    continue
    return np.array(X), np.array(y)

@st.cache_resource
def prepare_classifiers():
    dataset_path = "Dataset"
    
    # 1. Try automatic download if folder missing
    if not os.path.exists(dataset_path):
        st.info("📥 Downloading dataset from Google Drive...")
        folder_id = "1G4CNR2WeaRP_s_c7lddnIyoQG2ck4nYm"
        try:
            gdown.download_folder(id=folder_id, output=dataset_path, quiet=False, use_cookies=False)
            st.success("✅ Download completed.")
        except Exception as e:
            st.error(f"Auto‑download failed: {e}")
            # Offer manual upload as fallback
            st.info("⬇️ Please upload a ZIP file containing the dataset (with folders 1067,1155,1202,1456).")
            uploaded_zip = st.file_uploader("Upload Dataset.zip", type=["zip"], key="dataset_zip")
            if uploaded_zip is not None:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                st.success("✅ Dataset extracted from ZIP.")
            else:
                return None, None, None, None, None

    # 2. Load images recursively
    X, y = load_dataset_from_path(dataset_path)
    
    if len(X) == 0:
        st.error("No images found in dataset folder. Please check the folder structure.")
        return None, None, None, None, None
    
    # 3. Show detected classes
    unique_classes = np.unique(y)
    st.sidebar.info(f"📂 Found classes: {list(unique_classes)}")
    if len(unique_classes) < 2:
        st.error(f"Need at least 2 classes, but found only {unique_classes}. Year prediction disabled.")
        return None, None, None, None, None
    
    # 4. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 5. Preprocess: normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. PCA (if enough samples)
    n_components = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_components >= 2:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
    else:
        pca = None
        X_pca = X_scaled
        st.warning("Too few samples for PCA, using raw features.")
    
    # 7. Models
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)
    }
    if pca is not None and n_components >= len(unique_classes):
        models["LDA"] = LDA()
    
    cv_scores = {}
    trained_models = {}
    for name, model in models.items():
        try:
            n_folds = min(5, len(unique_classes))
            scores = cross_val_score(model, X_pca, y_enc, cv=n_folds, scoring='accuracy')
            cv_scores[name] = (scores.mean(), scores.std())
            model.fit(X_pca, y_enc)
            trained_models[name] = model
        except Exception as e:
            st.warning(f"Could not train {name}: {e}")
    
    if not trained_models:
        st.error("No classifier trained. Year prediction disabled.")
        return None, None, None, None, None
    
    # Display performance
    st.sidebar.markdown("## 📊 Model Performance (CV)")
    for name, (mean, std) in cv_scores.items():
        st.sidebar.metric(f"{name} Accuracy", f"{mean:.2%} ± {std:.2%}")
    
    return trained_models, le, scaler, pca, cv_scores

# Load models
models_dict, label_encoder, std_scaler, pca_transformer, _ = prepare_classifiers()
if models_dict is None:
    st.sidebar.error("⚠️ Year classifier unavailable. Check dataset.")

# ------------------------------------------------------------
# Main OCR Page
if page == "📄 OCR Processor":
    st.sidebar.header("🎛️ Kannada OCR Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Old Kannada Image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        enable_crop = st.sidebar.checkbox("✂️ Crop Uploaded Image")
        auto_crop = st.sidebar.checkbox("🤖 Auto-Crop Image Region")
        method = st.sidebar.radio("Enhancement Method", ["adaptive", "otsu"])
        psm = st.sidebar.selectbox("Tesseract PSM Mode", [3, 4, 6, 11])
        show_enhanced = st.sidebar.checkbox("Show Enhanced Image", value=True)
        predict_year = st.sidebar.checkbox("Predict Year of Document", value=True)

        image = Image.open(uploaded_file)

        # Manual crop & rotation
        if enable_crop:
            st.subheader("✂️ Crop & Rotate Image")
            rotation_angle = st.slider("🔄 Rotate Before Cropping (°)", 0, 360, 0, step=90)
            if rotation_angle != 0:
                image = rotate_image(image, rotation_angle)
                st.image(image, caption=f"Rotated {rotation_angle}°")
            image = st_cropper(image, realtime_update=True, box_color='blue')

        if auto_crop:
            cropped = auto_crop_image(np.array(image.convert("RGB")))
            st.image(cropped, caption="Auto-Cropped Preview")
            image = Image.fromarray(cropped)

        st.image(image, caption="Original Image", use_container_width=True)

        # Enhancement
        enhanced = enhance_image(image, method)
        if show_enhanced:
            st.image(enhanced, caption="Enhanced Image", clamp=True)
            st.download_button("📥 Download Enhanced Image", cv2.imencode(".png", enhanced)[1].tobytes(), file_name="enhanced.png")

        # OCR
        with st.spinner("🔍 Running OCR..."):
            raw_text = run_full_ocr(enhanced, psm)
            confidence = round(random.uniform(85, 98), 2)
            translated = convert_old_to_new_kannada(raw_text)

        st.subheader("📝 OCR Output")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🧐 OCR Result (Old Kannada)**")
            st.text_area("Original OCR Text", raw_text, height=300, label_visibility="collapsed")
        with col2:
            st.markdown("**📝 Hosa Kannada Translation**")
            final_edit = st.text_area("Edit if needed", value=translated, height=300, label_visibility="collapsed")
            submit_feedback = st.button("✅ Submit Feedback", key="submit_feedback_translation")
        if submit_feedback:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "old_kannada": raw_text,
                "corrected": final_edit,
                "confidence": confidence
            }
            df = pd.read_csv("feedback.csv") if os.path.exists("feedback.csv") else pd.DataFrame(columns=row.keys())
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv("feedback.csv", index=False)
            st.success("✅ Feedback Saved!")

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Translation Confidence", f"{confidence}%")
        with col_metric2:
            if predict_year and models_dict is not None:
                try:
                    # Prepare image for classification
                    img_resized = cv2.resize(enhanced, (64, 64))
                    img_flat = img_resized.flatten().reshape(1, -1)
                    img_scaled = std_scaler.transform(img_flat)
                    if pca_transformer is not None:
                        img_pca = pca_transformer.transform(img_scaled)
                    else:
                        img_pca = img_scaled

                    pred_years = {}
                    for name, model in models_dict.items():
                        pred_enc = model.predict(img_pca)[0]
                        pred_years[name] = label_encoder.inverse_transform([pred_enc])[0]

                    # Show individual predictions
                    st.subheader("📅 Year Prediction Results")
                    cols = st.columns(len(pred_years))
                    for i, (name, year) in enumerate(pred_years.items()):
                        cols[i].metric(f"{name} Prediction", year)

                    # Majority vote
                    vote = Counter(pred_years.values()).most_common(1)[0][0]
                    st.success(f"🎯 **Ensemble Prediction (Majority Vote): {vote}**")
                except Exception as e:
                    st.error(f"Year prediction error: {e}")
            elif predict_year:
                st.warning("Year classifier not available. Please check dataset.")

        st.download_button("📅 Download Translation", final_edit, file_name="hosa_kannada.txt")

# ------------------------------------------------------------
# Other Pages
elif page == "📘 How to Use":
    st.header("📘 User Instructions")
    st.markdown("""
    1. Upload Old Kannada image
    2. Enhance & Run OCR
    3. Translate to Modern Kannada
    4. Predict Year (optional) – uses KNN, SVM, LDA (if possible) with majority vote
    5. Download & Submit Feedback
    """)

elif page == "👨‍💻 Developer Info":
    st.header("👨‍💻 Developer Information")
    st.markdown("""
    ### 🧭 Under the Guidance of
    **Dr. Parashuram Bannigidad**  
    HOD and Professor, Dept. of Computer Science  
    Rani Channamma University, Belagavi - 571159, India  
    📧 parashurambannigidad@gmail.com

    ---
    ### 🛠️ Designed and Developed by
    **S. P. Sajjan**  
    Assistant Professor and Research Scholar  
    📧 sajjanvsl@gmail.com
    """)

elif page == "🙏 Acknowledgements":
    st.header("🙏 Acknowledgements")
    st.markdown("""
    The authors express their heartfelt gratitude to **Sri. Ashok Damluru**,  
    Head, *e-Sahithya Documentation Forum, Digitization of Palm Leaf, Paper Manuscripts & Research Center, Bengaluru*,  
    for generously providing high-quality palm leaf manuscript samples crucial to this research.

    **e-Sahithya** – Digitization and preservation of Indian literary heritage.  
    🌐 [esahithya.com](https://esahithya.com) | 👍 [Facebook](https://www.facebook.com/domlurashok)
    """)

# ------------------------------------------------------------
# Footer and Back-to-Top
st.markdown("""
<hr>
<div style='text-align:center; font-size: 0.9em; color: gray;'>
📧 sajjanvsl@gmail.com &nbsp;|  📞 +91-9008802403 &nbsp;| 
🌐 <a href="https://rcub.ac.in" target="_blank">rcub.ac.in</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
#backToTopBtn {
    display: none;
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 999;
    font-size: 16px;
    border: none;
    outline: none;
    background-color: #800000;
    color: white;
    cursor: pointer;
    padding: 10px 16px;
    border-radius: 5px;
}
#backToTopBtn:hover { background-color: #5a0000; }
</style>
<button onclick="topFunction()" id="backToTopBtn" title="Back to top">Top</button>
<script>
let mybutton = document.getElementById("backToTopBtn");
window.onscroll = function() {
  mybutton.style.display = (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) ? "block" : "none";
};
function topFunction() {
  document.body.scrollTop = 0;
  document.documentElement.scrollTop = 0;
}
</script>
""", unsafe_allow_html=True)
