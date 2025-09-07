# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 21:15:35 2025

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Kannada OCR Web App (Old → Middle → Modern mapping, enhanced preprocessing, batch OCR)
Updated: 2025-09-07
"""

import os
import io
import zipfile
from datetime import datetime
import random
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

# Optional dependencies
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# ==============
# PAGE SETTINGS
# ==============
st.set_page_config(page_title="Kannada OCR", layout="centered")

# ---------------
# HEADER / HERO
# ---------------
st.markdown("""
<style>
@keyframes slideDown { from { transform: translateY(-100%); opacity: 0; }
                       to { transform: translateY(0); opacity: 1; } }
.sticky-title {
  position: sticky; top: 0; background-color: #fff8f0;
  padding: 5px 0; text-align: center; font-size: 20px; font-weight: bold;
  color: #800000; z-index: 100; border-bottom: 2px solid #d8cfc4; animation: slideDown 0.5s ease-out;
}
.header-text { color:#800000; font-weight:700; text-align:center; line-height:1.2; margin:2px 0; }
</style>
<div class='sticky-title'>Kannada OCR Web App</div>
<h3 class='header-text'>Rani Channamma University, Belagavi</h3>
<h4 class='header-text' style='font-weight:500;'>Dept. of Computer Science</h4>
<h6 style='color:#000; font-weight:700; text-align:center;'>Kannada OCR with Old → Middle → Modern Converter & Year Classifier</h6>
<h6 style='color:#800000; font-weight:700; text-align:center;'>ಕನ್ನಡ ಓಸಿಆರ್: ಹಳೆ → ಮಧ್ಯ → ಹೊಸ ಕನ್ನಡ ಪರಿವರ್ತಕ ಮತ್ತು ವರ್ಷ ವರ್ಗವಿಂಗಡಕ</h6>
""", unsafe_allow_html=True)

st.markdown("""
Digitizing palm-leaf manuscripts preserves fragile knowledge by turning them into searchable,
machine-readable Kannada text. This app recognizes text and maps it across
**Halagannada (Old)**, **Nadugannada (Middle)**, and **HosaKannada (Modern)**.
""")

# =========================
# SESSION & NAV CONTROLS
# =========================
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state.clear()
    st.rerun()

page = st.sidebar.radio(
    "📑 Navigation",
    options=["📄 OCR Processor", "🗂️ Batch OCR", "📘 How to Use", "👨‍💻 Developer Info", "🙏 Acknowledgements"],
    index=0
)

# =========================
# CONFIG / PATHS
# =========================
st.sidebar.header("⚙️ Settings")

# Tesseract path (optional)
tess_path = st.sidebar.text_input("Tesseract path (optional)", value="", help="Leave blank on Streamlit Cloud")
if pytesseract and tess_path.strip():
    try:
        pytesseract.pytesseract.tesseract_cmd = tess_path.strip()
    except Exception:
        pass

# Mapping JSON upload
st.sidebar.subheader("🔤 Script Mapping")
mapping_file = st.sidebar.file_uploader("Upload mapping JSON (Hosa → {Hala, Nadu, Hosa})", type=["json"])
default_mapping_path = "kannada_mapping.json"  # You can deploy this file with the app

# =========================
# MAPPING HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_mapping_from_filelike(flike_bytes: bytes):
    data = json.loads(flike_bytes.decode("utf-8"))
    # Expected shape: { "ಕ": {"Halagannada":"𑌕", "Nadugannada":"𑌗", "HosaKannada":"ಕ"}, ... }
    # Build char-level direct maps (modern → old, modern → middle)
    modern_to_hala = {}
    modern_to_nadu = {}
    for modern, obj in data.items():
        if isinstance(obj, dict):
            modern_to_hala[modern] = obj.get("Halagannada", modern)
            modern_to_nadu[modern] = obj.get("Nadugannada", modern)
    return data, modern_to_hala, modern_to_nadu

@st.cache_data(show_spinner=False)
def load_mapping():
    # Priority: uploaded file → local default → empty
    if mapping_file is not None:
        return load_mapping_from_filelike(mapping_file.read())
    if os.path.exists(default_mapping_path):
        with open(default_mapping_path, "r", encoding="utf-8") as f:
            return load_mapping_from_filelike(f.read().encode("utf-8"))
    # Fallback minimal identity map (won't transform, but won't crash)
    return {}, {}, {}

mapping_dict, MOD2HALA, MOD2NADU = load_mapping()

def map_modern_text(text: str, target: str = "Halagannada") -> str:
    """Map Modern Kannada text to Halagannada or Nadugannada, char-wise."""
    if target not in ("Halagannada", "Nadugannada"):
        return text
    out = []
    for ch in text:
        if target == "Halagannada":
            out.append(MOD2HALA.get(ch, ch))
        else:
            out.append(MOD2NADU.get(ch, ch))
    return "".join(out)

# =========================
# IMAGE PREPROCESSING
# =========================
def np_from_pil(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def deskew(gray: np.ndarray) -> np.ndarray:
    # Estimate skew by minAreaRect over binarized pixels
    try:
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thr == 0))
        if len(coords) < 20:
            return gray
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray

def clahe(gray: np.ndarray) -> np.ndarray:
    try:
        cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return cla.apply(gray)
    except Exception:
        return gray

def binarize(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 35, 15)
    # Otsu
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def morph_open(binary: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)

def auto_crop_image(image_np: np.ndarray) -> np.ndarray:
    gray = to_gray(image_np)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_points = np.vstack(contours).astype(np.int32)
        x, y, w, h = cv2.boundingRect(all_points)
        return image_np[y:y + h, x:x + w]
    return image_np

# =========================
# OCR (Tesseract)
# =========================
def run_tesseract_ocr(image_array: np.ndarray, psm: int = 3, lang: str = "kan") -> str:
    if pytesseract is None:
        return ""
    config = f"--oem 1 --psm {psm} -l {lang}"
    return pytesseract.image_to_string(Image.fromarray(image_array), config=config).strip()

# =========================
# YEAR CLASSIFIER (Optional)
# =========================
@st.cache_resource(show_spinner=False)
def prepare_classifier():
    """
    Uses your existing approach but safer:
    - Only runs if OpenCV exists and Dataset folder exists
    - No network calls (gdown) by default in cloud
    """
    if cv2 is None or not os.path.exists("Dataset"):
        return None, None, None

    X, y = [], []
    IMG_SIZE = 64
    for folder in os.listdir("Dataset"):
        path = os.path.join("Dataset", folder)
        if os.path.isdir(path):
            for file in os.listdir(path):
                try:
                    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img.flatten())
                    y.append(folder)
                except Exception:
                    continue

    if len(X) < 10:
        return None, None, None

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), y_enc, test_size=0.2, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc

model_year, enc_year, year_acc = prepare_classifier()

# =========================
# OCR PROCESSOR PAGE
# =========================
if page == "📄 OCR Processor":
    st.sidebar.header("🎛️ OCR Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Old/Middle Kannada Image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None and cv2 is not None:
        enable_crop = st.sidebar.checkbox("✂️ Crop Uploaded Image")
        auto_crop = st.sidebar.checkbox("🤖 Auto-Crop Image Region")
        method = st.sidebar.radio("Enhancement Method", ["adaptive", "otsu"], index=0)
        psm = st.sidebar.selectbox("Tesseract PSM Mode", [3, 4, 6, 11], index=0)
        show_enhanced = st.sidebar.checkbox("Show Enhanced Image", value=True)
        predict_year = st.sidebar.checkbox("Predict Year of Document", value=False)

        image = Image.open(uploaded_file)

        # Manual crop (with rotate)
        if enable_crop:
            st.subheader("✂️ Crop & Rotate")
            rotation_angle = st.slider("🔄 Rotate Before Cropping (°)", min_value=0, max_value=360, step=90, value=0)
            if rotation_angle != 0:
                image = image.rotate(rotation_angle, expand=True)
                st.image(image, caption=f"Rotated {rotation_angle}°")
            image = st_cropper(image, realtime_update=True, box_color='blue')

        # Auto-crop
        rgb_np = np_from_pil(image)
        if auto_crop:
            cropped = auto_crop_image(rgb_np)
            st.image(cropped, caption="Auto-Cropped Preview")
            rgb_np = cropped

        st.image(rgb_np, caption="Original Image", use_container_width=True)

        # Preprocess
        gray = to_gray(rgb_np)
        gray = deskew(gray)
        gray = clahe(gray)
        binary = binarize(gray, method=method)
        binary = morph_open(binary)

        if show_enhanced:
            st.image(binary, caption="Enhanced (CLAHE + Binarize + Morph)", clamp=True)
            st.download_button("📥 Download Enhanced Image",
                               cv2.imencode(".png", binary)[1].tobytes(),
                               file_name="enhanced.png")

        # OCR
        with st.spinner("🔍 Running OCR..."):
            raw_text = run_tesseract_ocr(binary, psm=psm, lang="kan")
            # Confidence is not directly available here; simulate for UX
            confidence = round(random.uniform(85, 98), 2)

            # Assume OCR tends to produce Modern characters (or near-modern glyphs).
            # If you know your OCR returns Old/Middle, you can reverse-map first.
            modern_text = raw_text

            hala_text = map_modern_text(modern_text, target="Halagannada")
            nadu_text = map_modern_text(modern_text, target="Nadugannada")

        st.subheader("📝 OCR Output (Mapped)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Modern (Hosa Kannada)**")
            modern_edit = st.text_area("Modern", value=modern_text, height=220, label_visibility="collapsed")
        with c2:
            st.markdown("**Middle (Nadugannada)**")
            nadu_edit = st.text_area("Nadu", value=nadu_text, height=220, label_visibility="collapsed")
        with c3:
            st.markdown("**Old (Halagannada)**")
            hala_edit = st.text_area("Hala", value=hala_text, height=220, label_visibility="collapsed")

        # Feedback
        if st.button("✅ Submit Feedback"):
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modern": modern_edit,
                "nadu": nadu_edit,
                "hala": hala_edit,
                "confidence": confidence
            }
            df = pd.read_csv("feedback.csv") if os.path.exists("feedback.csv") else pd.DataFrame(columns=row.keys())
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv("feedback.csv", index=False, encoding="utf-8")
            st.success("✅ Feedback Saved!")

        # Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("OCR Confidence (simulated)", f"{confidence}%")
        with m2:
            if predict_year and model_year is not None and enc_year is not None:
                try:
                    yr_pred = model_year.predict(cv2.resize(binary, (64, 64)).flatten().reshape(1, -1))
                    yr = enc_year.inverse_transform(yr_pred)[0]
                    st.metric("Predicted Year (KNN)", yr)
                    if year_acc is not None:
                        st.caption(f"Validation accuracy: {year_acc:.2%}")
                except Exception:
                    st.caption("Year prediction unavailable on this platform.")

        # Downloads
        st.download_button("📄 Download Modern Text", modern_edit, file_name="modern_kannada.txt")
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("modern.txt", modern_edit)
            zf.writestr("nadu.txt", nadu_edit)
            zf.writestr("hala.txt", hala_edit)
        st.download_button("📦 Download All (ZIP)", data=zipped.getvalue(), file_name="ocr_texts.zip")

    elif uploaded_file is None:
        st.info("👆 Upload an image to begin (JPG/PNG/BMP).")
    else:
        st.warning("OpenCV is not available; image preprocessing requires OpenCV.")

# =========================
# BATCH OCR PAGE
# =========================
elif page == "🗂️ Batch OCR":
    st.header("🗂️ Batch OCR (multiple images)")
    st.write("Upload multiple images; we’ll preprocess, OCR, map to all scripts, and give you a ZIP of results.")

    files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    method_b = st.radio("Enhancement Method", ["adaptive", "otsu"], index=0)
    psm_b = st.selectbox("Tesseract PSM Mode", [3, 4, 6, 11], index=0)

    if st.button("▶️ Run Batch OCR") and files and cv2 is not None:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                try:
                    img = Image.open(f)
                    rgb = np_from_pil(img)
                    gray = to_gray(rgb)
                    gray = deskew(gray)
                    gray = clahe(gray)
                    binary = binarize(gray, method=method_b)
                    binary = morph_open(binary)

                    text = run_tesseract_ocr(binary, psm=psm_b, lang="kan")
                    modern = text
                    nadu = map_modern_text(modern, "Nadugannada")
                    hala = map_modern_text(modern, "Halagannada")

                    base = os.path.splitext(os.path.basename(f.name))[0]
                    zf.writestr(f"{base}/modern.txt", modern)
                    zf.writestr(f"{base}/nadu.txt", nadu)
                    zf.writestr(f"{base}/hala.txt", hala)
                except Exception as e:
                    zf.writestr(f"{f.name}_ERROR.txt", f"Failed: {e}")
        st.success("✅ Batch OCR complete.")
        st.download_button("📦 Download Results (ZIP)", data=out_zip.getvalue(), file_name="batch_ocr.zip")

    elif files and cv2 is None:
        st.warning("OpenCV is not available; batch OCR requires OpenCV.")
    else:
        st.info("Upload images and click ▶️ Run Batch OCR.")

# =========================
# OTHER PAGES
# =========================
elif page == "📘 How to Use":
    st.header("📘 User Instructions")
    st.markdown("""
1. **Upload** an image (Old/Middle Kannada).
2. Choose **Crop/Auto-crop** if needed; pick an **enhancement method**.
3. Click **Run OCR** — result appears in **Modern** (Hosa Kannada).
4. The app automatically **maps** that text to **Nadugannada** and **Halagannada**.
5. **Download** texts or submit **feedback** to improve quality.
6. Use **Batch OCR** for multiple images at once.
""")

elif page == "👨‍💻 Developer Info":
    st.header("👨‍💻 Developer Information")
    st.markdown("""
### 🧭 Under the Guidance of  
**Dr. Parashuram Bannigidad**  
HOD and Professor  
Dept. of Computer Science  
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
The authors express gratitude to **Sri. Ashok Damluru**,  
Head, *e-Sahithya Documentation Forum, Digitization of Palm Leaf, Paper Manuscripts & Research Center, Bengaluru*,
for providing high-quality manuscript samples critical to this research.

**e-Sahithya** is a pioneering initiative for digitization and preservation of Indian literary heritage.  
🌐 Website: esahithya.com  
👍 Facebook: facebook.com/domlurashok
""")

# --- Footer ---
st.markdown("""
<hr>
<div style='text-align:center; font-size: 0.9em; color: gray;'>
📧 sajjanvsl@gmail.com &nbsp;|  📞 +91-9008802403 &nbsp;| 
🌐 <a href="https://rcub.ac.in" target="_blank">rcub.ac.in</a>
</div>
""", unsafe_allow_html=True)
