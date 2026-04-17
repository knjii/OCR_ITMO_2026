import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm


DEFAULT_OCR_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,;:!?\"'()-[]/&"
)


try:
    import Levenshtein
except ImportError:  # pragma: no cover - optional speedup
    Levenshtein = None


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_grad_scaler(enabled: bool = True):
    enabled = enabled and torch.cuda.is_available()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda")
        return torch.cuda.amp.autocast()
    return nullcontext()


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=False)
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _as_batch_first(log_probs: torch.Tensor, batch_size: int) -> torch.Tensor:
    if log_probs.ndim != 3:
        raise ValueError(f"Expected 3D CTC log_probs, got shape {tuple(log_probs.shape)}")
    if log_probs.shape[0] == batch_size:
        return log_probs
    if log_probs.shape[1] == batch_size:
        return log_probs.transpose(0, 1).contiguous()
    raise ValueError(
        f"Cannot infer batch dimension for shape {tuple(log_probs.shape)} and batch={batch_size}"
    )


def _as_time_first(log_probs: torch.Tensor, batch_size: int) -> torch.Tensor:
    if log_probs.ndim != 3:
        raise ValueError(f"Expected 3D CTC log_probs, got shape {tuple(log_probs.shape)}")
    if log_probs.shape[1] == batch_size:
        return log_probs
    if log_probs.shape[0] == batch_size:
        return log_probs.transpose(0, 1).contiguous()
    raise ValueError(
        f"Cannot infer batch dimension for shape {tuple(log_probs.shape)} and batch={batch_size}"
    )


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor | Sequence[int] | None = None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    blank_index: int = 0,
) -> list[str]:
    """Greedy CTC decoder: argmax, remove duplicates, remove blank."""

    if input_lengths is None:
        batch_size = log_probs.shape[0]
        lengths = None
    else:
        lengths = [int(x) for x in torch.as_tensor(input_lengths).cpu().tolist()]
        batch_size = len(lengths)

    batch_first = _as_batch_first(log_probs.detach(), batch_size)
    predictions = batch_first.argmax(dim=-1).cpu()

    if lengths is None:
        lengths = [predictions.shape[1]] * predictions.shape[0]

    decoded: list[str] = []
    for indices, length in zip(predictions, lengths):
        chars: list[str] = []
        prev_idx: int | None = None
        for idx in indices[:length].tolist():
            idx = int(idx)
            if idx == blank_index:
                prev_idx = idx
                continue
            if idx == prev_idx:
                continue
            if 1 <= idx <= len(alphabet):
                chars.append(alphabet[idx - 1])
            prev_idx = idx
        decoded.append("".join(chars))
    return decoded


def _edit_distance(a: Sequence, b: Sequence) -> int:
    if isinstance(a, str) and isinstance(b, str) and Levenshtein is not None:
        return int(Levenshtein.distance(a, b))

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (item_a != item_b)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


class OCRMetricTracker:
    """Accumulates OCR metrics without batch-size bias."""

    def __init__(self) -> None:
        self.char_distance = 0
        self.char_total = 0
        self.word_distance = 0
        self.word_total = 0
        self.exact_matches = 0
        self.total = 0

    def update(self, predictions: Sequence[str], targets: Sequence[str]) -> None:
        for pred, target in zip(predictions, targets):
            self.char_distance += _edit_distance(pred, target)
            self.char_total += max(1, len(target))

            pred_words = pred.split()
            target_words = target.split()
            self.word_distance += _edit_distance(pred_words, target_words)
            self.word_total += max(1, len(target_words))

            self.exact_matches += int(pred == target)
            self.total += 1

    def compute(self) -> dict[str, float]:
        if self.total == 0:
            return {"cer": 0.0, "wer": 0.0, "accuracy": 0.0}
        return {
            "cer": self.char_distance / self.char_total,
            "wer": self.word_distance / self.word_total,
            "accuracy": self.exact_matches / self.total,
        }


def calculate_ocr_metrics(
    log_probs: torch.Tensor,
    texts: Sequence[str],
    input_lengths: torch.Tensor | Sequence[int] | None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    blank_index: int = 0,
) -> dict[str, float]:
    predictions = ctc_greedy_decode(
        log_probs=log_probs,
        input_lengths=input_lengths,
        alphabet=alphabet,
        blank_index=blank_index,
    )
    tracker = OCRMetricTracker()
    tracker.update(predictions, texts)
    return tracker.compute()


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _ctc_loss_from_batch(
    model: torch.nn.Module,
    batch: dict,
    criterion: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images"]
    batch_size = images.shape[0]
    log_probs = model(images)
    time_first = _as_time_first(log_probs, batch_size=batch_size)

    input_lengths = batch["input_lengths"].to("cpu")
    target_lengths = batch["target_lengths"].to("cpu")
    targets = batch["targets"]

    if not torch.isfinite(time_first).all():
        finite_mask = torch.isfinite(time_first)
        finite_ratio = finite_mask.float().mean().detach().cpu().item()
        raise FloatingPointError(
            "Model produced non-finite log probabilities. "
            f"log_probs_shape={tuple(time_first.shape)}, "
            f"dtype={time_first.dtype}, finite_ratio={finite_ratio:.6f}, "
            f"image_range=({float(images.min().detach().cpu()):.4f}, "
            f"{float(images.max().detach().cpu()):.4f})"
        )

    if (target_lengths > input_lengths).any():
        bad_idx = torch.nonzero(target_lengths > input_lengths, as_tuple=False).flatten()
        preview = [
            {
                "batch_idx": int(idx),
                "input_length": int(input_lengths[idx]),
                "target_length": int(target_lengths[idx]),
                "text": batch.get("texts", [""] * batch_size)[int(idx)][:120],
                "image_path": batch.get("image_paths", [""] * batch_size)[int(idx)],
            }
            for idx in bad_idx[:5]
        ]
        raise ValueError(f"CTC target is longer than input sequence: {preview}")

    # CTCLoss is numerically fragile in fp16. Keep model activations under AMP,
    # but compute the CTC objective in fp32.
    with _autocast_disabled(time_first.device):
        loss = criterion(
            time_first.float(),
            targets,
            input_lengths,
            target_lengths,
        )

    if not torch.isfinite(loss):
        per_sample_loss = torch.nn.functional.ctc_loss(
            time_first.float(),
            targets,
            input_lengths,
            target_lengths,
            blank=getattr(criterion, "blank", 0),
            reduction="none",
            zero_infinity=True,
        )
        bad_loss_idx = torch.nonzero(~torch.isfinite(per_sample_loss), as_tuple=False).flatten()
        preview = [
            {
                "batch_idx": int(idx),
                "input_length": int(input_lengths[idx]),
                "target_length": int(target_lengths[idx]),
                "loss": float(per_sample_loss[idx].detach().cpu()),
                "text": batch.get("texts", [""] * batch_size)[int(idx)][:120],
                "image_path": batch.get("image_paths", [""] * batch_size)[int(idx)],
            }
            for idx in bad_loss_idx[:5]
        ]
        raise FloatingPointError(
            "CTCLoss became non-finite. "
            f"log_probs_shape={tuple(time_first.shape)}, "
            f"input_lengths=({int(input_lengths.min())}, {int(input_lengths.max())}), "
            f"target_lengths=({int(target_lengths.min())}, {int(target_lengths.max())}), "
            f"bad_loss_samples={preview}"
        )
    return loss, log_probs


def train_epoch(
    model: torch.nn.Module,
    train_loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler=None,
    device: torch.device | str | None = None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    blank_index: int = 0,
    accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    use_amp: bool = True,
    compute_metrics: bool = True,
) -> tuple[float, dict[str, float]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scaler = scaler or create_grad_scaler(enabled=use_amp)
    accumulation_steps = max(1, int(accumulation_steps))

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    sample_count = 0
    tracker = OCRMetricTracker()

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = _move_batch_to_device(batch, device)
        batch_size = batch["images"].shape[0]

        with _autocast(device, enabled=use_amp):
            loss, log_probs = _ctc_loss_from_batch(model, batch, criterion)
            loss_for_backward = loss / accumulation_steps

        scaler.scale(loss_for_backward).backward()

        if step % accumulation_steps == 0 or step == len(train_loader):
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size

        if compute_metrics:
            predictions = ctc_greedy_decode(
                log_probs.detach(),
                input_lengths=batch["input_lengths"].detach().cpu(),
                alphabet=alphabet,
                blank_index=blank_index,
            )
            tracker.update(predictions, batch["texts"])

        metrics = tracker.compute() if compute_metrics else {}
        pbar.set_postfix({"loss": running_loss / sample_count, **metrics})

    avg_loss = running_loss / max(1, sample_count)
    return avg_loss, tracker.compute() if compute_metrics else {}


def validate_epoch(
    model: torch.nn.Module,
    val_loader,
    criterion: torch.nn.Module,
    *,
    device: torch.device | str | None = None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    blank_index: int = 0,
    use_amp: bool = True,
) -> tuple[float, dict[str, float]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model.eval()
    running_loss = 0.0
    sample_count = 0
    tracker = OCRMetricTracker()

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = _move_batch_to_device(batch, device)
            batch_size = batch["images"].shape[0]

            with _autocast(device, enabled=use_amp):
                loss, log_probs = _ctc_loss_from_batch(model, batch, criterion)

            running_loss += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size

            predictions = ctc_greedy_decode(
                log_probs.detach(),
                input_lengths=batch["input_lengths"].detach().cpu(),
                alphabet=alphabet,
                blank_index=blank_index,
            )
            tracker.update(predictions, batch["texts"])
            pbar.set_postfix({"loss": running_loss / sample_count, **tracker.compute()})

    avg_loss = running_loss / max(1, sample_count)
    return avg_loss, tracker.compute()


def _scheduler_step(scheduler, metric: float | None = None) -> None:
    if scheduler is None:
        return

    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(metric)
    else:
        scheduler.step()


def _is_better(value: float, best_value: float, mode: str) -> bool:
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError("mode must be 'min' or 'max'")


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_value: float,
    best_metric: str,
    history: dict,
) -> dict:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_value": best_value,
        "best_metric": best_metric,
        "history": history,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def train_model(
    model: torch.nn.Module,
    num_epochs: int,
    train_loader,
    val_loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    *,
    scaler=None,
    device: torch.device | str | None = None,
    alphabet: str = DEFAULT_OCR_ALPHABET,
    blank_index: int = 0,
    accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    use_amp: bool = True,
    best_metric: str = "cer",
    best_mode: str = "min",
    checkpoint_dir: str | os.PathLike = "checkpoints",
    resume_path: str | os.PathLike | None = None,
    history_path: str | os.PathLike = "training_history.json",
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    scaler = scaler or create_grad_scaler(enabled=use_amp)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "train_cer": [],
        "train_wer": [],
        "train_accuracy": [],
        "val_cer": [],
        "val_wer": [],
        "val_accuracy": [],
    }
    start_epoch = 0
    best_value = math.inf if best_mode == "min" else -math.inf

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint.get("best_value", best_value))
        history = checkpoint.get("history", history)
        print(f"Resumed from epoch {start_epoch}, best {best_metric}: {best_value:.6f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler=scaler,
            device=device,
            alphabet=alphabet,
            blank_index=blank_index,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            compute_metrics=True,
        )
        val_loss, val_metrics = validate_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            alphabet=alphabet,
            blank_index=blank_index,
            use_amp=use_amp,
        )

        metric_for_scheduler = val_loss if best_metric == "loss" else val_metrics.get(best_metric)
        _scheduler_step(scheduler, metric_for_scheduler)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        for metric_name in ("cer", "wer", "accuracy"):
            history[f"train_{metric_name}"].append(train_metrics.get(metric_name, 0.0))
            history[f"val_{metric_name}"].append(val_metrics.get(metric_name, 0.0))

        if best_metric == "loss":
            current_value = val_loss
        else:
            current_value = val_metrics[best_metric]

        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_value,
            best_metric,
            history,
        )
        torch.save(payload, last_path)

        if _is_better(current_value, best_value, best_mode):
            best_value = current_value
            payload["best_value"] = best_value
            torch.save(payload, best_path)
            print(f"Saved best checkpoint: {best_metric}={best_value:.6f}")

        print(
            "train_loss={:.5f} val_loss={:.5f} "
            "val_cer={:.5f} val_wer={:.5f} val_acc={:.5f} lr={:.2e}".format(
                train_loss,
                val_loss,
                val_metrics["cer"],
                val_metrics["wer"],
                val_metrics["accuracy"],
                current_lr,
            )
        )

        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    final_results = {
        "num_parameters_m": count_parameters(model) / 1e6,
        "best_metric": best_metric,
        "best_value": best_value,
    }
    final_results.update({key: values[-1] for key, values in history.items() if values})

    with open("final_metrics.json", "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2)

    return history


def plot_metrics(history: dict) -> None:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["train_cer"], label="train CER")
    plt.plot(history["val_cer"], label="val CER")
    plt.plot(history["train_wer"], label="train WER")
    plt.plot(history["val_wer"], label="val WER")
    plt.title("Error rates")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["train_accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="val")
    plt.title("Exact match accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
