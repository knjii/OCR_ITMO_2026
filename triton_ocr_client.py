"""Triton-backed OCR client for page and line inference."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from Datasets.Dataclass import DEFAULT_OCR_ALPHABET, extract_line_boxes_horizontal_projection
from inference import (
    OCRLineResult,
    OCRPageResult,
    OCRPartResult,
    merge_chunk_texts,
    read_image_grayscale,
    result_to_dict,
    split_long_line_crop,
)


def ctc_greedy_decode_numpy(
    log_probs: np.ndarray,
    input_lengths: Sequence[int],
    alphabet: str = DEFAULT_OCR_ALPHABET,
    blank_index: int = 0,
) -> list[str]:
    predictions = np.argmax(log_probs, axis=-1)
    decoded: list[str] = []

    for row, length in zip(predictions, input_lengths):
        chars: list[str] = []
        previous: int | None = None
        for idx in row[: int(length)].tolist():
            idx = int(idx)
            if idx == blank_index:
                previous = idx
                continue
            if idx == previous:
                continue
            if 1 <= idx <= len(alphabet):
                chars.append(alphabet[idx - 1])
            previous = idx
        decoded.append("".join(chars))

    return decoded


def _resize_and_pad_crop(crop: np.ndarray, max_width: int = 512) -> tuple[np.ndarray, int]:
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Empty OCR crop")

    resized_width = max(1, min(max_width, int(round(width * (32 / height)))))
    resized = cv2.resize(crop, (resized_width, 32), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = tensor[None, :, :]

    padded = np.zeros((1, 32, max_width), dtype=np.float32)
    padded[:, :, :resized_width] = tensor
    return padded, resized_width


class TritonOCRClient:
    def __init__(
        self,
        url: str = "localhost:8000",
        model_name: str = "svtr_ocr_onnx",
        model_version: str = "1",
        alphabet: str = DEFAULT_OCR_ALPHABET,
        max_width: int = 512,
    ) -> None:
        try:
            import tritonclient.http as httpclient
        except ImportError as exc:
            raise ImportError(
                "Install Triton HTTP client with: pip install tritonclient[http]"
            ) from exc

        self.httpclient = httpclient
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name
        self.model_version = model_version
        self.alphabet = alphabet
        self.max_width = max_width

    def is_ready(self) -> bool:
        return self.client.is_server_ready() and self.client.is_model_ready(
            self.model_name,
            self.model_version,
        )

    def infer_batch(self, images: np.ndarray) -> np.ndarray:
        infer_input = self.httpclient.InferInput("images", images.shape, "FP32")
        infer_input.set_data_from_numpy(images)
        infer_output = self.httpclient.InferRequestedOutput("log_probs")

        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=[infer_output],
        )
        return response.as_numpy("log_probs")

    def predict_line_crops(
        self,
        crops: Sequence[np.ndarray],
        *,
        batch_size: int = 2,
    ) -> list[tuple[str, float, int]]:
        results: list[tuple[str, float, int]] = []

        for start in range(0, len(crops), batch_size):
            chunk = crops[start : start + batch_size]
            prepared = [_resize_and_pad_crop(crop, max_width=self.max_width) for crop in chunk]
            images = np.stack([item[0] for item in prepared], axis=0)
            widths = [item[1] for item in prepared]
            input_lengths = [math.ceil(width / 4) for width in widths]

            log_probs = self.infer_batch(images)
            texts = ctc_greedy_decode_numpy(
                log_probs,
                input_lengths=input_lengths,
                alphabet=self.alphabet,
                blank_index=0,
            )
            probs = np.exp(log_probs)
            confidences = []
            for idx, length in enumerate(input_lengths):
                confidences.append(float(np.max(probs[idx, :length], axis=-1).mean()))
            results.extend(zip(texts, confidences, widths))

        return results

    def predict_page(
        self,
        page_image: np.ndarray,
        *,
        overlap: int = 64,
        batch_size: int = 2,
    ) -> OCRPageResult:
        if page_image.ndim == 3:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

        boxes = extract_line_boxes_horizontal_projection(page_image)
        all_crops: list[np.ndarray] = []
        chunk_meta: list[tuple[int, tuple[int, int, int, int]]] = []

        for line_idx, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            crop = page_image[y1:y2, x1:x2]
            for chunk_crop, chunk_bbox in split_long_line_crop(
                crop,
                page_bbox=bbox,
                img_height=32,
                max_width=self.max_width,
                overlap=overlap,
            ):
                all_crops.append(chunk_crop)
                chunk_meta.append((line_idx, chunk_bbox))

        predictions = self.predict_line_crops(all_crops, batch_size=batch_size)
        grouped_parts: list[list[OCRPartResult]] = [[] for _ in boxes]
        for (line_idx, chunk_bbox), (text, confidence, width) in zip(chunk_meta, predictions):
            grouped_parts[line_idx].append(
                OCRPartResult(text=text, confidence=confidence, bbox=chunk_bbox, width=width)
            )

        line_results: list[OCRLineResult] = []
        for bbox, parts in zip(boxes, grouped_parts):
            text = merge_chunk_texts([part.text for part in parts])
            confidence = float(np.mean([part.confidence for part in parts])) if parts else 0.0
            line_results.append(
                OCRLineResult(text=text, confidence=confidence, bbox=bbox, parts=parts)
            )

        page_text = "\n".join(line.text for line in line_results if line.text)
        return OCRPageResult(text=page_text, lines=line_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR through Triton Inference Server.")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model-name", default="svtr_ocr_onnx")
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TritonOCRClient(
        url=args.url,
        model_name=args.model_name,
        model_version=args.model_version,
    )
    image = read_image_grayscale(args.image)
    result = client.predict_page(image, overlap=args.overlap, batch_size=args.batch_size)
    payload = result_to_dict(result)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(result.text)


if __name__ == "__main__":
    main()
