import os

try:
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError
except ModuleNotFoundError:
    st = None
    StreamlitSecretNotFoundError = RuntimeError


def _get_streamlit_secret(name: str, default: str) -> str:
    if st is None:
        return default
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    _get_streamlit_secret("API_BASE_URL", "http://127.0.0.1:8080"),
).rstrip("/")
