"""FastAPI wrapper around the Triton OCR client.

The OCR logic stays in triton_ocr_client.py. This module only handles HTTP,
image upload validation, response shaping and health checks.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from Datasets.Dataclass import DEFAULT_OCR_ALPHABET, extract_line_boxes_horizontal_projection
from backend.schemas import (
    APIError,
    BoundingBox,
    HealthResponse,
    ModelInfoResponse,
    OCRLine,
    OCRLineResponse,
    OCRPageResponse,
    OCRPart,
    PagePreviewResponse,
)
from backend.settings import settings
from triton_ocr_client import TritonOCRClient


app = FastAPI(
    title="SVTR OCR API",
    description="HTTP wrapper for page-level OCR through Triton Inference Server.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OCRService:
    def __init__(self) -> None:
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def client(self) -> TritonOCRClient:
        try:
            client = TritonOCRClient(
                url=settings.triton_url,
                model_name=settings.model_name,
                model_version=settings.model_version,
                max_width=settings.max_width,
            )
            self._last_error = None
            return client
        except Exception as exc:  # Triton may be absent while Swagger still runs.
            self._last_error = str(exc)
            raise

    def readiness(self) -> tuple[bool, bool, str | None]:
        try:
            client = self.client()
            triton_ready = bool(client.client.is_server_ready())
            model_ready = bool(
                client.client.is_model_ready(settings.model_name, settings.model_version)
            )
            self._last_error = None
            return triton_ready, model_ready, None
        except Exception as exc:
            self._last_error = str(exc)
            return False, False, str(exc)

    def require_ready(self) -> TritonOCRClient:
        client = self.client()
        try:
            ready = client.is_ready()
        except Exception as exc:
            self._last_error = str(exc)
            ready = False

        if not ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_NOT_READY",
                    "message": (
                        f"Triton model {settings.model_name}:{settings.model_version} "
                        f"is not ready at {settings.triton_url}"
                    ),
                },
            )
        return client


def _predict_page_with_triton(image: np.ndarray, overlap: int, batch_size: int):
    client = get_service().require_ready()
    return client.predict_page(image, overlap=overlap, batch_size=batch_size)


def _predict_line_with_triton(image: np.ndarray, batch_size: int):
    client = get_service().require_ready()
    return client.predict_line_crops([image], batch_size=batch_size)


@lru_cache(maxsize=1)
def get_service() -> OCRService:
    return OCRService()


def _bbox_from_sequence(values: list[int] | tuple[int, int, int, int]) -> BoundingBox:
    x1, y1, x2, y2 = [int(item) for item in values]
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _line_from_payload(line: dict[str, Any], *, include_parts: bool) -> OCRLine:
    parts = []
    if include_parts:
        parts = [
            OCRPart(
                text=str(part.get("text", "")),
                confidence=float(part.get("confidence", 0.0)),
                bbox=_bbox_from_sequence(part["bbox"]),
                width=int(part.get("width", 0)),
            )
            for part in line.get("parts", [])
        ]

    return OCRLine(
        text=str(line.get("text", "")),
        confidence=float(line.get("confidence", 0.0)),
        bbox=_bbox_from_sequence(line["bbox"]),
        parts=parts,
    )


def _decode_image(content: bytes) -> np.ndarray:
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_IMAGE", "message": "Could not decode uploaded image"},
        )
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise HTTPException(
        status_code=400,
        detail={"code": "INVALID_IMAGE", "message": f"Unsupported image shape: {image.shape}"},
    )


async def _read_upload(file: UploadFile) -> np.ndarray:
    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if not content:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_FILE", "message": "Uploaded file is empty"},
        )
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"Maximum upload size is {settings.max_upload_mb} MB",
            },
        )
    return _decode_image(content)


@app.exception_handler(HTTPException)
async def api_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and {"code", "message"} <= set(exc.detail):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Invalid request parameters",
            "details": exc.errors(),
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    triton_ready, model_ready, error = get_service().readiness()
    return HealthResponse(
        status="ok" if triton_ready and model_ready else "degraded",
        triton_url=settings.triton_url,
        model_name=settings.model_name,
        model_version=settings.model_version,
        triton_ready=triton_ready,
        model_ready=model_ready,
        message=error,
    )


@app.get("/v1/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        triton_url=settings.triton_url,
        model_name=settings.model_name,
        model_version=settings.model_version,
        max_width=settings.max_width,
        default_batch_size=settings.default_batch_size,
        alphabet_size=len(DEFAULT_OCR_ALPHABET) + 1,
        blank_index=0,
    )


@app.post(
    "/v1/ocr/page/preview",
    response_model=PagePreviewResponse,
    responses={400: {"model": APIError}, 413: {"model": APIError}},
    tags=["ocr"],
)
async def preview_page_lines(file: UploadFile = File(...)) -> PagePreviewResponse:
    started = time.perf_counter()
    image = await _read_upload(file)
    boxes = extract_line_boxes_horizontal_projection(image)
    return PagePreviewResponse(
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        line_count=len(boxes),
        lines=[_bbox_from_sequence(box) for box in boxes],
        processing_ms=(time.perf_counter() - started) * 1000,
    )


@app.post(
    "/v1/ocr/page",
    response_model=OCRPageResponse,
    responses={
        400: {"model": APIError},
        413: {"model": APIError},
        503: {"model": APIError},
    },
    tags=["ocr"],
)
async def recognize_page(
    file: UploadFile = File(...),
    return_lines: bool = Form(True),
    return_parts: bool = Form(False),
    overlap: int = Form(settings.default_overlap, ge=0, le=256),
    batch_size: int = Form(settings.default_batch_size, ge=1, le=64),
) -> OCRPageResponse:
    started = time.perf_counter()
    image = await _read_upload(file)

    result = await run_in_threadpool(_predict_page_with_triton, image, overlap, batch_size)
    lines = []
    if return_lines:
        payload = {
            "lines": [
                {
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": list(line.bbox),
                    "parts": [
                        {
                            "text": part.text,
                            "confidence": part.confidence,
                            "bbox": list(part.bbox),
                            "width": part.width,
                        }
                        for part in line.parts
                    ],
                }
                for line in result.lines
            ]
        }
        lines = [
            _line_from_payload(line, include_parts=return_parts)
            for line in payload["lines"]
        ]

    return OCRPageResponse(
        text=result.text,
        lines=lines,
        processing_ms=(time.perf_counter() - started) * 1000,
        model_name=settings.model_name,
        model_version=settings.model_version,
    )


@app.post(
    "/v1/ocr/line",
    response_model=OCRLineResponse,
    responses={
        400: {"model": APIError},
        413: {"model": APIError},
        503: {"model": APIError},
    },
    tags=["ocr"],
)
async def recognize_line(
    file: UploadFile = File(...),
    batch_size: int = Form(1, ge=1, le=64),
) -> OCRLineResponse:
    started = time.perf_counter()
    image = await _read_upload(file)
    predictions = await run_in_threadpool(_predict_line_with_triton, image, batch_size)
    text, confidence, width = predictions[0]
    return OCRLineResponse(
        text=text,
        confidence=confidence,
        width=width,
        processing_ms=(time.perf_counter() - started) * 1000,
        model_name=settings.model_name,
        model_version=settings.model_version,
    )
