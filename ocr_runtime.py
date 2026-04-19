"""Lightweight OCR runtime helpers shared by Triton and FastAPI inference.

This module intentionally has no PyTorch dependency. It contains only the
pre/post-processing pieces needed after the model is already served by Triton.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DEFAULT_OCR_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,;:!?\"'()-[]/&"
)


@dataclass
class OCRPartResult:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    width: int


@dataclass
class OCRLineResult:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    parts: list[OCRPartResult]


@dataclass
class OCRPageResult:
    text: str
    lines: list[OCRLineResult]


def read_image_grayscale(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def resize_line_image(
    image: np.ndarray,
    img_height: int = 32,
    max_width: int = 512,
) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image")

    new_width = max(1, min(max_width, int(round(width * (img_height / height)))))
    return cv2.resize(image, (new_width, img_height), interpolation=cv2.INTER_AREA)


def extract_line_boxes_horizontal_projection(
    image: np.ndarray,
    min_line_height: int = 12,
    min_gap: int = 4,
    y_margin: int = 3,
    x_margin: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Find rough text line boxes with a horizontal projection profile."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    binary = _select_text_mask(gray, min_line_height, min_gap, y_margin, x_margin)
    boxes = _boxes_from_binary(binary, min_line_height, min_gap, y_margin, x_margin)
    return _split_tall_boxes(boxes, binary, y_margin, x_margin)


def _select_text_mask(
    gray: np.ndarray,
    min_line_height: int,
    min_gap: int,
    y_margin: int,
    x_margin: int,
) -> np.ndarray:
    otsu_mask = _otsu_text_mask(gray)
    otsu_boxes = _boxes_from_binary(otsu_mask, min_line_height, min_gap, y_margin, x_margin)
    if _is_reasonable_line_set(otsu_boxes, gray.shape[0]):
        return otsu_mask

    candidates = [otsu_mask]

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 15)),
    )
    _, blackhat_otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(_clean_text_mask(blackhat_otsu))
    for threshold in (10, 15, 20):
        candidates.append(_clean_text_mask((blackhat > threshold).astype(np.uint8) * 255))

    return max(
        candidates,
        key=lambda mask: _line_mask_score(
            mask,
            min_line_height=min_line_height,
            min_gap=min_gap,
            y_margin=y_margin,
            x_margin=x_margin,
        ),
    )


def _is_reasonable_line_set(
    boxes: list[tuple[int, int, int, int]],
    image_height: int,
) -> bool:
    if len(boxes) < 8:
        return False
    heights = [y2 - y1 for _, y1, _, y2 in boxes]
    typical_height = _estimate_line_height(np.array(heights, dtype=np.float32))
    return max(heights) < max(90, int(2.8 * typical_height), int(0.12 * image_height))


def _otsu_text_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _clean_text_mask(binary)


def _clean_text_mask(binary: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )


def _boxes_from_binary(
    binary: np.ndarray,
    min_line_height: int,
    min_gap: int,
    y_margin: int,
    x_margin: int,
) -> list[tuple[int, int, int, int]]:
    row_projection = (binary > 0).sum(axis=1)
    threshold = max(2, int(0.01 * binary.shape[1]))
    active_rows = row_projection > threshold

    boxes: list[tuple[int, int, int, int]] = []
    start: int | None = None
    last_active: int | None = None

    for row, is_active in enumerate(active_rows):
        if is_active:
            if start is None:
                start = row
            last_active = row
        elif start is not None and last_active is not None and row - last_active > min_gap:
            if last_active - start + 1 >= min_line_height:
                boxes.append(_line_box_from_rows(binary, start, last_active, y_margin, x_margin))
            start = None
            last_active = None

    if start is not None and last_active is not None and last_active - start + 1 >= min_line_height:
        boxes.append(_line_box_from_rows(binary, start, last_active, y_margin, x_margin))

    return boxes


def _line_mask_score(
    binary: np.ndarray,
    *,
    min_line_height: int,
    min_gap: int,
    y_margin: int,
    x_margin: int,
) -> float:
    boxes = _boxes_from_binary(binary, min_line_height, min_gap, y_margin, x_margin)
    if not boxes:
        return -1_000_000.0

    image_height, image_width = binary.shape
    heights = np.array([y2 - y1 for _, y1, _, y2 in boxes], dtype=np.float32)
    widths = np.array([x2 - x1 for x1, _, x2, _ in boxes], dtype=np.float32)
    typical_height = _estimate_line_height(heights)
    max_height = float(np.max(heights))

    too_few_penalty = max(0, 12 - len(boxes)) * 25
    too_many_penalty = max(0, len(boxes) - 80) * 3
    huge_box_penalty = 80 if max_height > 0.25 * image_height else 0
    tall_penalty = float(np.sum(heights > max(65, 1.7 * typical_height))) * 10
    page_wide_penalty = float(np.sum((widths > 0.98 * image_width) & (heights < 30))) * 4

    return (
        len(boxes) * 8
        - too_few_penalty
        - too_many_penalty
        - huge_box_penalty
        - tall_penalty
        - page_wide_penalty
    )


def _split_tall_boxes(
    boxes: list[tuple[int, int, int, int]],
    binary: np.ndarray,
    y_margin: int,
    x_margin: int,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    heights = np.array([y2 - y1 for _, y1, _, y2 in boxes], dtype=np.float32)
    typical_height = _estimate_line_height(heights)
    projection_split_threshold = max(58, 1.45 * typical_height)
    force_split_threshold = max(90, 2.25 * typical_height)
    target_height = max(24, int(round(1.35 * typical_height)))

    split_boxes: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        if box_height <= projection_split_threshold:
            split_boxes.append(box)
            continue

        sub_boxes = _split_box_by_projection(binary, y1, y2, y_margin, x_margin)
        if len(sub_boxes) > 1:
            for sub_box in sub_boxes:
                if sub_box[3] - sub_box[1] > force_split_threshold:
                    split_boxes.extend(
                        _split_box_evenly(
                            binary,
                            sub_box,
                            target_height=target_height,
                            y_margin=y_margin,
                            x_margin=x_margin,
                        )
                    )
                else:
                    split_boxes.append(sub_box)
        elif box_height > force_split_threshold:
            split_boxes.extend(
                _split_box_evenly(
                    binary,
                    box,
                    target_height=target_height,
                    y_margin=y_margin,
                    x_margin=x_margin,
                )
            )
        else:
            split_boxes.append(box)

    split_boxes = _filter_low_density_boxes(split_boxes, binary)
    return sorted(_deduplicate_boxes(split_boxes), key=lambda item: (item[1], item[0]))


def _estimate_line_height(heights: np.ndarray) -> float:
    """Estimate one-line height without letting merged boxes inflate it."""

    if heights.size == 0:
        return 40.0
    positive_heights = heights[heights > 0]
    if positive_heights.size == 0:
        return 40.0
    return float(np.percentile(positive_heights, 35))


def _filter_low_density_boxes(
    boxes: list[tuple[int, int, int, int]],
    binary: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    filtered: list[tuple[int, int, int, int]] = []
    image_width = binary.shape[1]

    for box in boxes:
        x1, y1, x2, y2 = box
        crop = binary[y1:y2, x1:x2] > 0
        width = max(1, x2 - x1)
        if crop.size == 0:
            continue

        max_row_ratio = float(crop.sum(axis=1).max()) / width
        density = float(crop.mean())
        is_wide_sparse_noise = (
            width > 0.5 * image_width
            and max_row_ratio < 0.06
            and density < 0.04
        )
        if not is_wide_sparse_noise:
            filtered.append(box)

    return filtered


def _split_box_evenly(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    target_height: int,
    y_margin: int,
    x_margin: int,
) -> list[tuple[int, int, int, int]]:
    _, y1, _, y2 = box
    height = y2 - y1
    if height <= target_height:
        return [box]

    parts = max(2, int(math.ceil(height / target_height)))
    step = height / parts
    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(parts):
        part_y1 = int(round(y1 + idx * step))
        part_y2 = int(round(y1 + (idx + 1) * step))
        if part_y2 - part_y1 >= 8:
            boxes.append(_line_box_from_rows(binary, part_y1, part_y2, y_margin, x_margin))
    return boxes


def _split_box_by_projection(
    binary: np.ndarray,
    y1: int,
    y2: int,
    y_margin: int,
    x_margin: int,
) -> list[tuple[int, int, int, int]]:
    local_projection = (binary[y1:y2] > 0).sum(axis=1).astype(np.float32)
    if local_projection.size == 0 or float(local_projection.max()) <= 0:
        return []

    window = min(9, max(3, int(local_projection.size // 20) | 1))
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(local_projection, kernel, mode="same")
    threshold = max(2.0, 0.12 * float(smoothed.max()))
    active_rows = smoothed > threshold

    sub_boxes: list[tuple[int, int, int, int]] = []
    start: int | None = None
    last_active: int | None = None

    for offset, is_active in enumerate(active_rows):
        row = y1 + offset
        if is_active:
            if start is None:
                start = row
            last_active = row
        elif start is not None and last_active is not None and row - last_active > 2:
            if last_active - start + 1 >= 8:
                sub_boxes.append(_line_box_from_rows(binary, start, last_active, y_margin, x_margin))
            start = None
            last_active = None

    if start is not None and last_active is not None and last_active - start + 1 >= 8:
        sub_boxes.append(_line_box_from_rows(binary, start, last_active, y_margin, x_margin))

    return sub_boxes


def _deduplicate_boxes(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.85,
) -> list[tuple[int, int, int, int]]:
    unique: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not any(_box_iou(box, other) > iou_threshold for other in unique):
            unique.append(box)
    return unique


def _box_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def _line_box_from_rows(
    binary: np.ndarray,
    y1: int,
    y2: int,
    y_margin: int,
    x_margin: int,
) -> tuple[int, int, int, int]:
    h, w = binary.shape
    y1 = max(0, y1 - y_margin)
    y2 = min(h, y2 + y_margin + 1)

    cols = np.where((binary[y1:y2] > 0).sum(axis=0) > 0)[0]
    if cols.size:
        x1 = max(0, int(cols[0]) - x_margin)
        x2 = min(w, int(cols[-1]) + x_margin + 1)
    else:
        x1, x2 = 0, w

    if x2 - x1 > 0.95 * w:
        x1 = max(x1, int(0.06 * w))
        x2 = min(x2, int(0.94 * w))
    return x1, y1, x2, y2


def split_long_line_crop(
    crop: np.ndarray,
    *,
    page_bbox: tuple[int, int, int, int],
    img_height: int = 32,
    max_width: int = 512,
    overlap: int = 64,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Split an over-wide line crop into model-sized chunks with overlap."""

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    height, width = crop.shape[:2]
    if _resized_width(width, height, img_height, max_width) <= max_width:
        return [(resize_line_image(crop, img_height=img_height, max_width=max_width), page_bbox)]

    max_original_width = max(1, int(math.floor(max_width * height / img_height)))
    overlap_original = max(0, int(round(overlap * height / img_height)))
    step = max(1, max_original_width - overlap_original)

    x1_page, y1_page, _, y2_page = page_bbox
    chunks: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    start = 0
    while start < width:
        end = min(width, start + max_original_width)
        if end - start <= 0:
            break

        chunk = resize_line_image(
            crop[:, start:end],
            img_height=img_height,
            max_width=max_width,
        )
        chunks.append((chunk, (x1_page + start, y1_page, x1_page + end, y2_page)))

        if end >= width:
            break
        start += step

    return chunks


def _resized_width(width: int, height: int, img_height: int, max_width: int) -> int:
    if width <= 0 or height <= 0:
        return 0
    return min(max_width, max(1, int(round(width * (img_height / height)))))


def merge_chunk_texts(texts: Sequence[str]) -> str:
    merged = ""
    for text in texts:
        merged = _merge_pair(merged, text)
    return merged.strip()


def _merge_pair(left: str, right: str, max_overlap: int = 48) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left

    max_len = min(max_overlap, len(left), len(right))
    best_len = 0
    best_score = 1.0

    for overlap_len in range(1, max_len + 1):
        suffix = left[-overlap_len:]
        prefix = right[:overlap_len]
        distance = _edit_distance(suffix, prefix)
        score = distance / max(1, overlap_len)
        if score < best_score or (score == best_score and overlap_len > best_len):
            best_score = score
            best_len = overlap_len

    if best_len >= 4 and best_score <= 0.35:
        return left + right[best_len:]

    no_space_after_left = left.endswith((" ", "-", "'", '"'))
    no_space_before_right = right.startswith((" ", ".", ",", ";", ":", "!", "?", "'", '"'))
    separator = "" if no_space_after_left or no_space_before_right else " "
    return left + separator + right


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def result_to_dict(result: OCRPageResult) -> dict:
    return {
        "text": result.text,
        "lines": [
            {
                "text": line.text,
                "confidence": line.confidence,
                "bbox": list(line.bbox),
                "parts": [
                    {
                        **asdict(part),
                        "bbox": list(part.bbox),
                    }
                    for part in line.parts
                ],
            }
            for line in result.lines
        ],
    }
