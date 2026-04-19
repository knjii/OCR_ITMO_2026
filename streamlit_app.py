import streamlit as st
import pandas as pd
import os
import json
import requests

from frontend.settings import API_BASE_URL
from frontend.api import post_ocr_page, post_preview
from frontend.drawing import draw_lines

# Загрузка CSS из файла
def load_css():
    css_file = "style.css"  
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"Файл {css_file} не найден")

load_css()

# --- UI ---
st.title('OCR Text Recognition', anchor = False) # anchor - гиперссылка справа от текста, из-за неё не центрировалось нормально

uploaded_file = st.file_uploader(label="Choose an image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "tif", "tiff"])
return_lines = st.checkbox("Return lines")
return_parts = st.checkbox("Return parts")
overlap = st.slider("Overlap", min_value=0, max_value=500, value=64)

# --- RUN OCR ---
if st.button(label='Run OCR', width="stretch"):
    if uploaded_file is None:
        st.warning("Please upload an image first")
    else:
        try:
            with st.spinner("Processing..."):
                result = post_ocr_page(
                    uploaded_file,
                    return_lines,
                    return_parts,
                    overlap
                )

            # --- TEXT ---
            st.subheader("Recognized text")
            st.text_area("OCR result", value=result["text"], height=400)

            # --- META ---
            st.caption(f"Processing time: {result['processing_ms']:.1f} ms")
            st.caption(f"Lines: {len(result.get('lines', []))}")

            # --- TABLE ---
            rows = []
            for idx, line in enumerate(result.get("lines", []), start=1):
                bbox = line["bbox"]
                rows.append({
                    "line": idx,
                    "text": line["text"],
                    "confidence": line["confidence"],
                    "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                })

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # --- DOWNLOAD ---
            st.download_button(
                "Download TXT",
                data=result["text"],
                file_name="ocr_result.txt",
                mime="text/plain",
            )

            st.download_button(
                "Download JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name="ocr_result.json",
                mime="application/json",
            )

        except requests.Timeout:
            st.error("OCR request timed out. Try a smaller image.")
        
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}")

        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")

# --- PREVIEW ---
if st.button("Preview line boxes", use_container_width=True):

    if uploaded_file is None:
        st.warning("Upload image first")
    else:
        try:
            result = post_preview(
                uploaded_file,
                return_lines,
                return_parts,
                overlap
            )

            image = draw_lines(uploaded_file, result)

            st.image(image, caption="Detected text lines", use_container_width=True)

        except Exception as e:
            st.error(f"Preview failed: {e}")