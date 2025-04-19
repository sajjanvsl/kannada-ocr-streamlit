# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 00:18:19 2025

@author: Admin
"""

# 📊 Dashboard + Training Pipeline using Feedback

import pandas as pd
import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_title="Feedback Dashboard & Trainer", layout="wide")

st.title("📋 Feedback Dashboard and Model Trainer")

# Load feedback.csv if it exists
feedback_file = "feedback.csv"

if os.path.exists(feedback_file):
    df = pd.read_csv(feedback_file)
    st.success(f"Loaded {len(df)} feedback entries")

    # Show data table
    st.subheader("📝 Collected Feedback")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Feedback as CSV", csv, "feedback_data.csv", "text/csv")

    # Prepare training data
    st.subheader("🔄 Retrain Model with Feedback")
    if st.button("Start Model Training (Simulated)"):
        st.info("Training on collected feedback...")

        # Simulate a retraining step (real model pipeline would go here)
        # You could save these pairs into a tokenizer-based dataset for a transformer model

        for idx, row in df.iterrows():
            st.text(f"Old: {row['old_kannada']}")
            st.text(f"New: {row['hosa_kannada']}")
            st.text("-")

        st.success("✅ Model training completed (simulated). You can now export the dataset.")
else:
    st.warning("⚠️ No feedback.csv found. Submit corrections from the main app first.")
