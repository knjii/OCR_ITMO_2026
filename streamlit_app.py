import json
import os

import pandas as pd
import requests
import streamlit as st

from frontend.api import get_health, post_ocr_pages, post_preview
from frontend.drawing import draw_lines
from frontend.settings import API_BASE_URL


def load_css() -> None:
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="SVTR OCR", page_icon="OCR", layout="wide")
load_css()

st.title("OCR Text Recognition", anchor=False)

with st.sidebar:
    st.caption(f"Backend: {API_BASE_URL}")
    if st.button("Check backend", use_container_width=True):
        try:
            st.json(get_health())
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")

uploaded_files = st.file_uploader(
    label="Choose one or more images",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)
return_lines = st.checkbox("Return lines", value=True)
return_parts = st.checkbox("Return parts")
overlap = st.slider("Overlap", min_value=0, max_value=256, value=64)
batch_size = st.slider("Triton batch size", min_value=1, max_value=16, value=2)

run_col, preview_col = st.columns(2)

if run_col.button(label="Run OCR", use_container_width=True):
    if not uploaded_files:
        st.warning("Please upload at least one image first")
    else:
        try:
            with st.spinner("Processing..."):
                result = post_ocr_pages(
                    uploaded_files,
                    return_lines=return_lines,
                    return_parts=return_parts,
                    overlap=overlap,
                    batch_size=batch_size,
                )

            st.subheader("Recognized text")
            page_texts = []
            for page in result["pages"]:
                page_texts.append(f"===== {page['filename']} =====\n{page['text']}")
            combined_text = "\n\n".join(page_texts)
            st.text_area("OCR result", value=combined_text, height=400)
            st.caption(f"Total processing time: {result['processing_ms']:.1f} ms")
            st.caption(f"Pages: {len(result.get('pages', []))}")

            rows = []
            for page_idx, page in enumerate(result.get("pages", []), start=1):
                for line_idx, line in enumerate(page.get("lines", []), start=1):
                    bbox = line["bbox"]
                    rows.append(
                        {
                            "page": page_idx,
                            "filename": page["filename"],
                            "line": line_idx,
                            "text": line["text"],
                            "confidence": line["confidence"],
                            "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                        }
                    )

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.download_button(
                "Download TXT",
                data=combined_text,
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
        except requests.HTTPError as exc:
            st.error(f"HTTP error: {exc}")
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")

if preview_col.button("Preview line boxes", use_container_width=True):
    if not uploaded_files:
        st.warning("Upload at least one image first")
    else:
        for uploaded_file in uploaded_files:
            try:
                result = post_preview(uploaded_file)
                image = draw_lines(uploaded_file, result)
                st.image(
                    image,
                    caption=f"Detected text lines: {uploaded_file.name}",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Preview failed for {uploaded_file.name}: {exc}")
