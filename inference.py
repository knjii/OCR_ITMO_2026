"""PyTorch OCR inference for line and page images.

The model is trained as a line recognizer. Page inference is implemented as:
page image -> line boxes -> line crops -> optional long-line chunks -> OCR ->
chunk merge -> page text reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from Datasets.Dataclass import (
    DEFAULT_OCR_ALPHABET,
    extract_line_boxes_horizontal_projection,
    image_to_tensor,
    resize_line_image,
)
from Train import ctc_greedy_decode
from ocr_model import build_svtr_tiny_ocr


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


def load_ocr_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
) -> torch.nn.Module:
    """Load a trained SVTR OCR checkpoint for inference."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = build_svtr_tiny_ocr(alphabet=alphabet, output_batch_first=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def read_image_grayscale(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def _resized_width(width: int, height: int, img_height: int, max_width: int) -> int:
    if width <= 0 or height <= 0:
        return 0
    return min(max_width, max(1, int(round(width * (img_height / height)))))


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

        chunk = crop[:, start:end]
        chunk = resize_line_image(chunk, img_height=img_height, max_width=max_width)
        chunks.append((chunk, (x1_page + start, y1_page, x1_page + end, y2_page)))

        if end >= width:
            break
        start += step

    return chunks


def _pad_tensors(images: Sequence[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_width = max(image.shape[-1] for image in images)
    batch = torch.full(
        (len(images), images[0].shape[0], images[0].shape[1], max_width),
        fill_value=pad_value,
        dtype=images[0].dtype,
    )
    for idx, image in enumerate(images):
        batch[idx, :, :, : image.shape[-1]] = image
    return batch


def _confidence_from_log_probs(log_probs: torch.Tensor, input_lengths: Sequence[int]) -> list[float]:
    probs = log_probs.detach().float().exp()
    max_probs = probs.max(dim=-1).values.cpu()

    confidences: list[float] = []
    for idx, length in enumerate(input_lengths):
        length = max(1, min(int(length), max_probs.shape[1]))
        confidences.append(float(max_probs[idx, :length].mean().item()))
    return confidences


@torch.inference_mode()
def predict_line_crops(
    model: torch.nn.Module,
    crops: Sequence[np.ndarray],
    *,
    device: str | torch.device | None = None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    img_height: int = 32,
    max_width: int = 512,
    batch_size: int = 32,
    normalize: str = "zero_one",
) -> list[tuple[str, float, int]]:
    """Recognize already cropped line/chunk images."""

    if not crops:
        return []

    device = torch.device(device or next(model.parameters()).device)
    results: list[tuple[str, float, int]] = []

    for start in range(0, len(crops), batch_size):
        chunk = crops[start : start + batch_size]
        tensors = []
        widths = []
        for crop in chunk:
            resized = resize_line_image(crop, img_height=img_height, max_width=max_width)
            tensor = image_to_tensor(resized, normalize=normalize)
            tensors.append(tensor)
            widths.append(tensor.shape[-1])

        images = _pad_tensors(tensors).to(device)
        input_lengths = [math.ceil(width / 4) for width in widths]
        log_probs = model(images)
        texts = ctc_greedy_decode(
            log_probs=log_probs,
            input_lengths=input_lengths,
            alphabet=alphabet,
            blank_index=0,
        )
        confidences = _confidence_from_log_probs(log_probs, input_lengths)
        results.extend(zip(texts, confidences, widths))

    return results


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

    separator = "" if left.endswith((" ", "-", "'", '"')) or right.startswith((" ", ".", ",", ";", ":", "!", "?", "'", '"')) else " "
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


def merge_chunk_texts(texts: Sequence[str]) -> str:
    merged = ""
    for text in texts:
        merged = _merge_pair(merged, text)
    return merged.strip()


def predict_page(
    model: torch.nn.Module,
    page_image: np.ndarray,
    *,
    device: str | torch.device | None = None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    img_height: int = 32,
    max_width: int = 512,
    overlap: int = 64,
    batch_size: int = 32,
) -> OCRPageResult:
    """Recognize a page image and return ordered line OCR results."""

    if page_image.ndim == 3:
        page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

    boxes = extract_line_boxes_horizontal_projection(page_image)
    all_crops: list[np.ndarray] = []
    chunk_meta: list[tuple[int, tuple[int, int, int, int]]] = []
    line_boxes: list[tuple[int, int, int, int]] = []

    for line_idx, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        crop = page_image[y1:y2, x1:x2]
        line_boxes.append(bbox)
        for chunk_crop, chunk_bbox in split_long_line_crop(
            crop,
            page_bbox=bbox,
            img_height=img_height,
            max_width=max_width,
            overlap=overlap,
        ):
            all_crops.append(chunk_crop)
            chunk_meta.append((line_idx, chunk_bbox))

    predictions = predict_line_crops(
        model,
        all_crops,
        device=device,
        alphabet=alphabet,
        img_height=img_height,
        max_width=max_width,
        batch_size=batch_size,
    )

    grouped_parts: list[list[OCRPartResult]] = [[] for _ in line_boxes]
    for (line_idx, chunk_bbox), (text, confidence, width) in zip(chunk_meta, predictions):
        grouped_parts[line_idx].append(
            OCRPartResult(text=text, confidence=confidence, bbox=chunk_bbox, width=width)
        )

    line_results: list[OCRLineResult] = []
    for bbox, parts in zip(line_boxes, grouped_parts):
        part_texts = [part.text for part in parts]
        text = merge_chunk_texts(part_texts)
        confidence = float(np.mean([part.confidence for part in parts])) if parts else 0.0
        line_results.append(
            OCRLineResult(
                text=text,
                confidence=confidence,
                bbox=bbox,
                parts=parts,
            )
        )

    page_text = "\n".join(line.text for line in line_results if line.text)
    return OCRPageResult(text=page_text, lines=line_results)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR inference on a page image.")
    parser.add_argument("--image", required=True, help="Path to a page image.")
    parser.add_argument("--checkpoint", default="checkpoints_2/best_model.pth")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_ocr_model(args.checkpoint, device=args.device)
    image = read_image_grayscale(args.image)
    result = predict_page(
        model,
        image,
        device=args.device,
        max_width=args.max_width,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    payload = result_to_dict(result)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(result.text)


if __name__ == "__main__":
    main()
