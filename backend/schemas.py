"""Pydantic response schemas for the OCR API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class APIError(BaseModel):
    code: str
    message: str


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class OCRPart(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox
    width: int


class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox
    parts: list[OCRPart] = Field(default_factory=list)


class OCRPageResponse(BaseModel):
    text: str
    lines: list[OCRLine] = Field(default_factory=list)
    processing_ms: float
    model_name: str
    model_version: str


class OCRLineResponse(BaseModel):
    text: str
    confidence: float
    width: int
    processing_ms: float
    model_name: str
    model_version: str


class PagePreviewResponse(BaseModel):
    image_width: int
    image_height: int
    line_count: int
    lines: list[BoundingBox]
    processing_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    triton_url: str
    model_name: str
    model_version: str
    triton_ready: bool
    model_ready: bool
    message: str | None = None


class ModelInfoResponse(BaseModel):
    triton_url: str
    model_name: str
    model_version: str
    max_width: int
    default_batch_size: int
    alphabet_size: int
    blank_index: int

