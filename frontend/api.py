import json
import requests
import streamlit as st

API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

def _build_options_dict(return_lines: bool, return_parts: bool, overlap: int, max_width: int = 512, binarize: bool = False):
    return {
        "return_lines": bool(return_lines),
        "return_parts": bool(return_parts),
        "max_width": int(max_width),
        "overlap": int(overlap),
        "binarize": bool(binarize),
    }

def post_ocr_page(uploaded_file, return_lines: bool, return_parts: bool, overlap: int, max_width: int = 512, binarize: bool = False, timeout: int = 90):
    
    # Send image and options to POST {API_BASE_URL}/v1/ocr/page.
    # Returns parsed JSON (dict) from backend on success.
    # May raise requests.Timeout, requests.HTTPError, requests.RequestException.
    
    options = _build_options_dict(return_lines, return_parts, overlap, max_width, binarize)

    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), getattr(uploaded_file, "type", "application/octet-stream"))}
    data = {"options": json.dumps(options)}

    response = requests.post(
        f"{API_BASE_URL}/v1/ocr/page",
        files=files,
        data=data,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()

def post_preview(uploaded_file, return_lines: bool, return_parts: bool, overlap: int, max_width: int = 512, binarize: bool = False, timeout: int = 90):
    
    # Send image and options to POST {API_BASE_URL}/v1/ocr/page/preview.
    # Returns parsed JSON (dict) from backend on success.
    # May raise requests.Timeout, requests.HTTPError, requests.RequestException.

    options = _build_options_dict(return_lines, return_parts, overlap, max_width, binarize)

    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), getattr(uploaded_file, "type", "application/octet-stream"))}
    data = {"options": json.dumps(options)}

    response = requests.post(
        f"{API_BASE_URL}/v1/ocr/page/preview",
        files=files,
        data=data,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()