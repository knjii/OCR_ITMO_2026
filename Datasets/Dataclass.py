import math
import os
import random as _random
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


DEFAULT_OCR_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,;:!?\"'()-[]/&"
)


def normalize_ocr_text(
    text: str,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    lowercase: bool = True,
) -> str:
    """Normalize page or line text to the alphabet used by the CTC model."""

    if lowercase:
        text = text.lower()

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\t": " ",
        "\n": " ",
        "\r": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    allowed = set(alphabet)
    return "".join(ch for ch in text if ch in allowed)


def encode_text(text: str, alphabet: str = DEFAULT_OCR_ALPHABET) -> list[int]:
    """Encode text for CTC targets. Index 0 is reserved for blank."""

    char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}
    return [char_to_idx[char] for char in text if char in char_to_idx]


def decode_ctc_indices(
    indices: Sequence[int],
    alphabet: str = DEFAULT_OCR_ALPHABET,
    blank_index: int = 0,
    collapse_repeats: bool = True,
) -> str:
    """Decode a sequence of CTC token ids into text."""

    chars: list[str] = []
    prev_idx: int | None = None

    for idx in indices:
        idx = int(idx)
        if idx == blank_index:
            prev_idx = idx
            continue
        if collapse_repeats and idx == prev_idx:
            continue
        if 1 <= idx <= len(alphabet):
            chars.append(alphabet[idx - 1])
        prev_idx = idx

    return "".join(chars)


def _apply_transform(transform: Callable | None, image: np.ndarray) -> np.ndarray:
    if transform is None:
        return image

    result = transform(image=image)
    if isinstance(result, dict):
        return result["image"]
    return result


def resize_line_image(
    image: np.ndarray,
    img_height: int = 32,
    max_width: int = 512,
) -> np.ndarray:
    """Resize a grayscale line image to a fixed height, preserving width ratio."""

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image")

    new_width = int(round(width * (img_height / height)))
    new_width = max(1, min(max_width, new_width))
    return cv2.resize(image, (new_width, img_height), interpolation=cv2.INTER_AREA)


def image_to_tensor(image: np.ndarray, normalize: str = "zero_one") -> torch.Tensor:
    """Convert a grayscale image to [1, H, W] float tensor."""

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
    if normalize == "zero_one":
        tensor = tensor / 255.0
    elif normalize == "minus_one_one":
        tensor = tensor / 127.5 - 1.0
    else:
        raise ValueError("normalize must be 'zero_one' or 'minus_one_one'")
    return tensor


def _split_ids(
    ids: Sequence[str],
    split: str | None,
    val_size: float,
    test_size: float,
    seed: int,
) -> list[str]:
    ids = sorted(ids)
    if split is None:
        return ids

    rng = _random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)

    n_items = len(ids)
    n_test = math.ceil(n_items * test_size)
    n_val = math.ceil(n_items * val_size)

    if split == "test":
        return ids[:n_test]
    if split == "val":
        return ids[n_test : n_test + n_val]
    if split == "train":
        return ids[n_test + n_val :]

    raise ValueError(f"Unknown split: {split!r}. Expected 'train', 'val' or 'test'.")


class OldBooksDataclass(Dataset):
    """Page-level dataset for the source old-books-dataset files."""

    def __init__(
        self,
        root_dir: str | os.PathLike,
        images_folder: str = "300dpi/tiff",
        ground_truths_folder: str = "groundtruth",
        split: str | None = None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
        augmentation: Callable | None = None,
        preprocessing: Callable | None = None,
        grayscale: bool = True,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / images_folder
        self.ground_truths_dir = self.root_dir / ground_truths_folder
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.grayscale = grayscale

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")
        if not self.ground_truths_dir.exists():
            raise FileNotFoundError(f"Groundtruth folder not found: {self.ground_truths_dir}")

        image_ids = [path.stem for path in self.images_dir.glob("*.tiff")]
        self.image_ids = _split_ids(image_ids, split, val_size, test_size, seed)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image_path = self.images_dir / f"{image_id}.tiff"
        text_path = self.ground_truths_dir / f"{image_id}.txt"

        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), flag)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        with open(text_path, "r", encoding="utf-8", errors="replace") as file:
            text = file.read()

        image = _apply_transform(self.augmentation, image)
        image = _apply_transform(self.preprocessing, image)

        return {
            "image": image,
            "text": text,
            "image_id": image_id,
            "image_path": str(image_path),
            "text_path": str(text_path),
        }


class OCRLineDataset(Dataset):
    """Line-level OCR dataset from a TSV manifest: image_path<TAB>text."""

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        root_dir: str | os.PathLike | None = None,
        alphabet: str = DEFAULT_OCR_ALPHABET,
        img_height: int = 32,
        max_width: int = 512,
        augmentation: Callable | None = None,
        preprocessing: Callable | None = None,
        normalize: str = "zero_one",
        lowercase: bool = True,
        drop_empty: bool = True,
    ) -> None:
        super().__init__()

        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir is not None else self.manifest_path.parent
        self.alphabet = alphabet
        self.img_height = img_height
        self.max_width = max_width
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.normalize = normalize
        self.lowercase = lowercase

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.samples: list[tuple[Path, str]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Bad manifest line {line_no}: expected image_path<TAB>text"
                    )

                rel_or_abs_path, text = parts
                text = normalize_ocr_text(text, alphabet=alphabet, lowercase=lowercase)
                if drop_empty and not text:
                    continue

                image_path = Path(rel_or_abs_path)
                if not image_path.is_absolute():
                    image_path = self.root_dir / image_path
                self.samples.append((image_path, text))

        if not self.samples:
            raise ValueError(f"No usable samples found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, text = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read line image: {image_path}")

        image = resize_line_image(image, img_height=self.img_height, max_width=self.max_width)
        image = _apply_transform(self.augmentation, image)
        image = _apply_transform(self.preprocessing, image)
        image = resize_line_image(image, img_height=self.img_height, max_width=self.max_width)

        target = torch.tensor(encode_text(text, self.alphabet), dtype=torch.long)
        tensor = image_to_tensor(image, normalize=self.normalize)

        return {
            "image": tensor,
            "target": target,
            "target_length": len(target),
            "text": text,
            "image_path": str(image_path),
            "width": tensor.shape[-1],
        }


def ocr_collate_fn(
    batch: Sequence[dict],
    pad_value: float = 0.0,
    max_width: int = 512,
    patch_width_stride: int = 4,
) -> dict:
    """Pad line images to the longest width in the batch and pack CTC targets."""

    if not batch:
        raise ValueError("Empty batch is not supported")

    heights = {sample["image"].shape[-2] for sample in batch}
    if len(heights) != 1:
        raise ValueError(f"All images must have the same height, got {sorted(heights)}")

    batch_width = min(max(sample["image"].shape[-1] for sample in batch), max_width)
    channels = batch[0]["image"].shape[0]
    height = batch[0]["image"].shape[-2]

    images = torch.full(
        (len(batch), channels, height, batch_width),
        fill_value=pad_value,
        dtype=batch[0]["image"].dtype,
    )
    widths: list[int] = []
    targets: list[torch.Tensor] = []
    texts: list[str] = []
    image_paths: list[str] = []

    for idx, sample in enumerate(batch):
        image = sample["image"][..., :batch_width]
        width = image.shape[-1]
        images[idx, :, :, :width] = image
        widths.append(width)
        targets.append(sample["target"])
        texts.append(sample["text"])
        image_paths.append(sample["image_path"])

    target_lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
    packed_targets = torch.cat(targets).long()
    input_lengths = torch.tensor(
        [math.ceil(width / patch_width_stride) for width in widths],
        dtype=torch.long,
    )

    return {
        "images": images,
        "targets": packed_targets,
        "target_lengths": target_lengths,
        "input_lengths": input_lengths,
        "texts": texts,
        "image_paths": image_paths,
        "widths": torch.tensor(widths, dtype=torch.long),
    }


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

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

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
    return x1, y1, x2, y2


def split_text_into_line_count(text: str, line_count: int, alphabet: str = DEFAULT_OCR_ALPHABET) -> list[str]:
    """Fallback page-text splitter used when no aligned line transcript exists."""

    text = normalize_ocr_text(text, alphabet=alphabet)
    words = text.split()
    if line_count <= 0:
        return []
    if not words:
        return [""] * line_count

    total_chars = sum(len(word) for word in words) + max(0, len(words) - 1)
    target_chars = max(1, math.ceil(total_chars / line_count))

    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > target_chars and len(lines) < line_count - 1:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra

    lines.append(" ".join(current))
    while len(lines) < line_count:
        lines.append("")
    return lines[:line_count]


def build_line_manifest_from_pages(
    pages_dataset: OldBooksDataclass,
    output_dir: str | os.PathLike,
    manifest_path: str | os.PathLike,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    img_height: int = 32,
    max_width: int = 512,
    overwrite: bool = False,
) -> Path:
    """Create a bootstrap line-level manifest from page images.

    This uses projection-based line extraction and proportional page-text
    splitting. For final experiments, replace this bootstrap manifest with a
    manually or OCR-aligned line manifest.
    """

    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []

    for page in pages_dataset:
        image = page["image"]
        boxes = extract_line_boxes_horizontal_projection(image)
        line_texts = split_text_into_line_count(page["text"], len(boxes), alphabet=alphabet)

        for line_idx, (box, text) in enumerate(zip(boxes, line_texts)):
            text = normalize_ocr_text(text, alphabet=alphabet)
            if not text:
                continue

            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            crop = resize_line_image(crop, img_height=img_height, max_width=max_width)
            line_name = f"{page['image_id']}_{line_idx:04d}.png"
            line_path = output_dir / line_name

            if overwrite or not line_path.exists():
                cv2.imwrite(str(line_path), crop)

            rows.append(f"{line_path.as_posix()}\t{text}")

    with open(manifest_path, "w", encoding="utf-8") as file:
        file.write("\n".join(rows))
        if rows:
            file.write("\n")

    return manifest_path
