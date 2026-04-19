"""Build an EasyOCR + Levenshtein aligned line-level OCR manifest.

The source dataset stores page images and page-level transcripts. This script
cuts page images into line crops, recognizes every crop with EasyOCR, then
aligns the rough OCR lines to the page transcript in reading order.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

try:
    import Levenshtein
except ImportError:  # pragma: no cover - optional speedup
    Levenshtein = None

from Datasets.Dataclass import (
    DEFAULT_OCR_ALPHABET,
    OldBooksDataclass,
    extract_line_boxes_horizontal_projection,
    normalize_ocr_text,
    resize_line_image,
    split_text_into_line_count,
)




@dataclass
class LineOCRResult:
    line_idx: int
    image_path: str
    box: tuple[int, int, int, int]
    rough_text: str
    aligned_text: str = ""


def edit_distance(left: Sequence, right: Sequence) -> int:
    if isinstance(left, str) and isinstance(right, str) and Levenshtein is not None:
        return int(Levenshtein.distance(left, right))

    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def get_easyocr_reader(gpu: bool | None = None):
    try:
        import easyocr
    except ImportError as exc:
        raise ImportError(
            "EasyOCR is required for aligned manifest generation. "
            "Install it with: pip install easyocr"
        ) from exc

    if gpu is None:
        gpu = torch.cuda.is_available()
    return easyocr.Reader(["en"], gpu=gpu, verbose=False)


def recognize_line(reader, image: np.ndarray, alphabet: str = DEFAULT_OCR_ALPHABET) -> str:
    """Recognize a cropped line with EasyOCR and normalize to model alphabet."""

    try:
        pieces = reader.readtext(
            image,
            detail=0,
            paragraph=False,
            decoder="greedy",
            batch_size=1,
            workers=0,
        )
    except RuntimeError:
        pieces = []

    text = " ".join(str(piece) for piece in pieces)
    return normalize_ocr_text(text, alphabet=alphabet)


def _segment_cost(rough_line: str, candidate: str, target_avg_chars: float) -> float:
    if rough_line:
        distance = edit_distance(rough_line, candidate)
        return distance / max(1, len(rough_line), len(candidate))

    # EasyOCR can miss very degraded short lines. Keep these lines near the
    # page-average text length instead of letting them absorb the whole page.
    return 0.50 + abs(len(candidate) - target_avg_chars) / max(1.0, target_avg_chars)


def align_rough_lines_to_text(
    rough_lines: Sequence[str],
    page_text: str,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    max_segment_words: int = 42,
) -> list[str]:
    """Monotonic DP alignment from rough OCR lines to page transcript words."""

    normalized_text = normalize_ocr_text(page_text, alphabet=alphabet)
    words = normalized_text.split()
    line_count = len(rough_lines)

    if line_count == 0:
        return []
    if not words:
        return [""] * line_count

    word_count = len(words)
    target_avg_chars = max(1.0, len(normalized_text) / max(1, line_count))
    inf = float("inf")

    dp = [[inf] * (word_count + 1) for _ in range(line_count + 1)]
    parent: list[list[tuple[int, int] | None]] = [
        [None] * (word_count + 1) for _ in range(line_count + 1)
    ]
    dp[0][0] = 0.0

    for line_idx, rough_line in enumerate(rough_lines):
        remaining_lines_after = line_count - line_idx - 1
        rough_words = max(1, len(rough_line.split()))

        for start_word in range(word_count + 1):
            if not math.isfinite(dp[line_idx][start_word]):
                continue

            words_left = word_count - start_word
            min_words = 0 if words_left <= remaining_lines_after else 1
            max_words_by_remainder = words_left
            if remaining_lines_after > 0:
                max_words_by_remainder = words_left

            adaptive_max = max(
                8,
                int(math.ceil(rough_words * 2.4)) + 6,
                int(math.ceil(words_left / max(1, remaining_lines_after + 1) * 2.2)) + 4,
            )
            max_words = min(max_segment_words, max_words_by_remainder, adaptive_max)

            for take_words in range(min_words, max_words + 1):
                end_word = start_word + take_words
                if end_word > word_count:
                    break
                if word_count - end_word < 0:
                    continue

                candidate = " ".join(words[start_word:end_word])
                cost = _segment_cost(rough_line, candidate, target_avg_chars)
                length_penalty = 0.02 * abs(len(candidate.split()) - rough_words)
                new_cost = dp[line_idx][start_word] + cost + length_penalty

                if new_cost < dp[line_idx + 1][end_word]:
                    dp[line_idx + 1][end_word] = new_cost
                    parent[line_idx + 1][end_word] = (start_word, end_word)

    best_end = word_count
    if not math.isfinite(dp[line_count][best_end]):
        return split_text_into_line_count(page_text, line_count, alphabet=alphabet)

    aligned = [""] * line_count
    cursor = best_end
    for line_pos in range(line_count, 0, -1):
        prev = parent[line_pos][cursor]
        if prev is None:
            return split_text_into_line_count(page_text, line_count, alphabet=alphabet)
        start_word, end_word = prev
        aligned[line_pos - 1] = " ".join(words[start_word:end_word])
        cursor = start_word

    if cursor != 0:
        prefix = " ".join(words[:cursor])
        aligned[0] = f"{prefix} {aligned[0]}".strip()

    return aligned


def _json_ready_result(result: LineOCRResult) -> dict:
    return {
        "line_idx": result.line_idx,
        "image_path": result.image_path,
        "box": list(result.box),
        "rough_text": result.rough_text,
        "aligned_text": result.aligned_text,
    }


def _load_cached_results(cache_path: Path) -> list[LineOCRResult] | None:
    if not cache_path.exists():
        return None

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [
        LineOCRResult(
            line_idx=int(item["line_idx"]),
            image_path=str(item["image_path"]),
            box=tuple(int(v) for v in item["box"]),
            rough_text=str(item.get("rough_text", "")),
            aligned_text=str(item.get("aligned_text", "")),
        )
        for item in data
    ]


def _save_cached_results(cache_path: Path, results: Sequence[LineOCRResult]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_json_ready_result(result) for result in results]
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_manifest_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resize_width_for_crop(crop: np.ndarray, img_height: int, max_width: int) -> int:
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return 0
    return min(max_width, max(1, int(round(width * (img_height / height)))))


def _split_text_balanced_by_words(text: str, parts: int) -> list[str]:
    words = text.split()
    if parts <= 1 or not words:
        return [text]

    total_chars = sum(len(word) for word in words) + max(0, len(words) - 1)
    target_chars = max(1, math.ceil(total_chars / parts))
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current else 0)
        should_flush = (
            current
            and current_len + word_len > target_chars
            and len(chunks) < parts - 1
        )
        if should_flush:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += word_len

    if current:
        chunks.append(" ".join(current))

    while len(chunks) < parts:
        chunks.append("")
    return chunks[:parts]


def _text_chunks_for_ctc_limit(text: str, max_target_length: int) -> list[str]:
    """Split text into word-preserving chunks below the requested CTC target limit."""

    if len(text) <= max_target_length:
        return [text]

    parts = math.ceil(len(text) / max_target_length)
    chunks = _split_text_balanced_by_words(text, parts)

    # Very long OCR tokens are rare but possible after normalization. Split them
    # as a fallback so no target exceeds the configured limit.
    fixed: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_target_length:
            fixed.append(chunk)
            continue
        for start in range(0, len(chunk), max_target_length):
            fixed.append(chunk[start : start + max_target_length].strip())
    return [chunk for chunk in fixed if chunk]


def _split_crop_by_text_chunks(
    crop: np.ndarray,
    chunks: Sequence[str],
    img_height: int,
    max_width: int,
    min_chunk_width: int,
) -> list[np.ndarray] | None:
    if not chunks:
        return None

    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return None

    char_counts = [max(1, len(chunk)) for chunk in chunks]
    total_chars = sum(char_counts)
    boundaries = [0]
    acc = 0
    for count in char_counts[:-1]:
        acc += count
        boundaries.append(int(round(width * acc / total_chars)))
    boundaries.append(width)

    crop_chunks: list[np.ndarray] = []
    for idx, chunk_text in enumerate(chunks):
        x1 = max(0, min(width - 1, boundaries[idx]))
        x2 = max(x1 + 1, min(width, boundaries[idx + 1]))
        part = crop[:, x1:x2]
        resized_width = _resize_width_for_crop(part, img_height=img_height, max_width=max_width)
        input_length = math.ceil(resized_width / 4)
        if resized_width < min_chunk_width or len(chunk_text) > input_length:
            return None
        crop_chunks.append(part)

    return crop_chunks


def expand_result_to_manifest_rows(
    page: dict,
    result: LineOCRResult,
    line_dir: Path,
    project_root: Path,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    img_height: int = 32,
    max_width: int = 512,
    max_target_length: int = 112,
    min_chunk_width: int = 16,
    overwrite_images: bool = False,
) -> list[tuple[str, str]]:
    """Return one or more CTC-valid manifest rows for a line alignment result."""

    text = normalize_ocr_text(result.aligned_text, alphabet=alphabet)
    if not text:
        return []

    image = page["image"]
    x1, y1, x2, y2 = result.box
    crop = image[y1:y2, x1:x2]
    resized_width = _resize_width_for_crop(crop, img_height=img_height, max_width=max_width)
    input_length = math.ceil(resized_width / 4)

    if len(text) <= min(max_target_length, input_length):
        return [(result.image_path, text)]

    chunks = _text_chunks_for_ctc_limit(text, max_target_length=max_target_length)
    crop_chunks = _split_crop_by_text_chunks(
        crop=crop,
        chunks=chunks,
        img_height=img_height,
        max_width=max_width,
        min_chunk_width=min_chunk_width,
    )
    if crop_chunks is None:
        return []

    page_id = page["image_id"]
    page_line_dir = line_dir / page_id
    page_line_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for chunk_idx, (chunk_crop, chunk_text) in enumerate(zip(crop_chunks, chunks)):
        chunk_crop = resize_line_image(chunk_crop, img_height=img_height, max_width=max_width)
        chunk_path = page_line_dir / f"{page_id}_{result.line_idx:04d}_part{chunk_idx:02d}.png"
        if overwrite_images or not chunk_path.exists():
            cv2.imwrite(str(chunk_path), chunk_crop)
        rows.append((_relative_manifest_path(chunk_path, project_root), chunk_text))

    return rows


def process_page(
    page: dict,
    reader,
    line_dir: Path,
    cache_dir: Path,
    project_root: Path,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    img_height: int = 32,
    max_width: int = 512,
    overwrite_images: bool = False,
    force_ocr: bool = False,
) -> list[LineOCRResult]:
    page_id = page["image_id"]
    cache_path = cache_dir / f"{page_id}.json"

    cached = None if force_ocr else _load_cached_results(cache_path)
    if cached is not None:
        rough_results = cached
    else:
        image = page["image"]
        boxes = extract_line_boxes_horizontal_projection(image)
        rough_results = []
        page_line_dir = line_dir / page_id
        page_line_dir.mkdir(parents=True, exist_ok=True)

        for line_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            crop = resize_line_image(crop, img_height=img_height, max_width=max_width)

            line_path = page_line_dir / f"{page_id}_{line_idx:04d}.png"
            if overwrite_images or not line_path.exists():
                cv2.imwrite(str(line_path), crop)

            rough_text = recognize_line(reader, crop, alphabet=alphabet)
            rough_results.append(
                LineOCRResult(
                    line_idx=line_idx,
                    image_path=_relative_manifest_path(line_path, project_root),
                    box=tuple(int(v) for v in box),
                    rough_text=rough_text,
                )
            )

    aligned_lines = align_rough_lines_to_text(
        [result.rough_text for result in rough_results],
        page["text"],
        alphabet=alphabet,
    )
    for result, aligned_text in zip(rough_results, aligned_lines):
        result.aligned_text = normalize_ocr_text(aligned_text, alphabet=alphabet)

    _save_cached_results(cache_path, rough_results)
    return rough_results


def build_aligned_manifest(
    data_root: str | Path = "Datasets",
    images_folder: str = "300dpi/tiff",
    groundtruth_folder: str = "groundtruth",
    output_manifest: str | Path = "Datasets/manifests/lines_aligned.tsv",
    line_dir: str | Path = "Datasets/lines_aligned",
    cache_dir: str | Path = "Datasets/easyocr_cache",
    alphabet: str = DEFAULT_OCR_ALPHABET,
    img_height: int = 32,
    max_width: int = 512,
    max_target_length: int = 112,
    min_chunk_width: int = 16,
    gpu: bool | None = None,
    limit_pages: int | None = None,
    overwrite_images: bool = False,
    force_ocr: bool = False,
) -> Path:
    project_root = Path.cwd().resolve()
    data_root = Path(data_root)
    output_manifest = Path(output_manifest)
    line_dir = Path(line_dir)
    cache_dir = Path(cache_dir)

    dataset = OldBooksDataclass(
        root_dir=data_root,
        images_folder=images_folder,
        ground_truths_folder=groundtruth_folder,
        split=None,
        grayscale=True,
    )
    if limit_pages is not None:
        dataset.image_ids = dataset.image_ids[:limit_pages]

    reader = get_easyocr_reader(gpu=gpu)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for page_idx in range(len(dataset)):
        page = dataset[page_idx]
        results = process_page(
            page=page,
            reader=reader,
            line_dir=line_dir,
            cache_dir=cache_dir,
            project_root=project_root,
            alphabet=alphabet,
            img_height=img_height,
            max_width=max_width,
            overwrite_images=overwrite_images,
            force_ocr=force_ocr,
        )

        kept = 0
        skipped = 0
        split_parts = 0
        for result in results:
            manifest_rows = expand_result_to_manifest_rows(
                page=page,
                result=result,
                line_dir=line_dir,
                project_root=project_root,
                alphabet=alphabet,
                img_height=img_height,
                max_width=max_width,
                max_target_length=max_target_length,
                min_chunk_width=min_chunk_width,
                overwrite_images=overwrite_images,
            )
            if not manifest_rows:
                skipped += 1
                continue

            for image_path, text in manifest_rows:
                rows.append(f"{image_path}\t{text}")
            kept += len(manifest_rows)
            split_parts += max(0, len(manifest_rows) - 1)

        print(
            f"[{page_idx + 1:04d}/{len(dataset):04d}] "
            f"{page['image_id']}: lines={len(results)}, kept={kept}, "
            f"split_parts={split_parts}, skipped={skipped}"
        )

    output_manifest.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"Saved aligned manifest: {output_manifest} ({len(rows)} lines)")
    return output_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Datasets")
    parser.add_argument("--images-folder", default="300dpi/tiff")
    parser.add_argument("--groundtruth-folder", default="groundtruth")
    parser.add_argument("--output-manifest", default="Datasets/manifests/lines_aligned.tsv")
    parser.add_argument("--line-dir", default="Datasets/lines_aligned")
    parser.add_argument("--cache-dir", default="Datasets/easyocr_cache")
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=112)
    parser.add_argument("--min-chunk-width", type=int, default=16)
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force EasyOCR to run on CPU.")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--force-ocr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_aligned_manifest(
        data_root=args.data_root,
        images_folder=args.images_folder,
        groundtruth_folder=args.groundtruth_folder,
        output_manifest=args.output_manifest,
        line_dir=args.line_dir,
        cache_dir=args.cache_dir,
        img_height=args.img_height,
        max_width=args.max_width,
        max_target_length=args.max_target_length,
        min_chunk_width=args.min_chunk_width,
        gpu=False if args.cpu else None,
        limit_pages=args.limit_pages,
        overwrite_images=args.overwrite_images,
        force_ocr=args.force_ocr,
    )


if __name__ == "__main__":
    main()
