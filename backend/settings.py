"""Runtime settings for the OCR FastAPI backend."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _split_env_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class BackendSettings:
    triton_url: str = os.getenv("TRITON_URL", "localhost:8000")
    model_name: str = os.getenv("TRITON_MODEL_NAME", "svtr_ocr_onnx")
    model_version: str = os.getenv("TRITON_MODEL_VERSION", "1")
    max_width: int = int(os.getenv("OCR_MAX_WIDTH", "512"))
    default_overlap: int = int(os.getenv("OCR_OVERLAP", "64"))
    default_batch_size: int = int(os.getenv("TRITON_BATCH_SIZE", "2"))
    max_batch_size: int = int(os.getenv("TRITON_MAX_BATCH_SIZE", "2"))
    max_upload_mb: int = int(os.getenv("OCR_MAX_UPLOAD_MB", "20"))
    cors_origins: tuple[str, ...] = _split_env_list(
        os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")
    )


settings = BackendSettings()
