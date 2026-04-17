"""Export the trained SVTR OCR model to ONNX and validate parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from Datasets.Dataclass import DEFAULT_OCR_ALPHABET
from ocr_model import build_svtr_tiny_ocr


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def load_model(checkpoint_path: str | Path, alphabet: str = DEFAULT_OCR_ALPHABET) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_svtr_tiny_ocr(alphabet=alphabet, output_batch_first=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset: int = 18,
    batch_size: int = 1,
    width: int = 512,
    dynamic_batch: bool = False,
    dynamic_width: bool = False,
) -> Path:
    """Export model to ONNX.

    Static width is the default because the SVTR local-window attention uses
    shape-dependent padding and reshapes. For Triton production, prefer dynamic
    batch with fixed-width buckets, e.g. 128/256/384/512.
    """

    model = load_model(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(batch_size, 1, 32, width, dtype=torch.float32)
    dynamic_axes = None
    dynamic_shapes = None
    if dynamic_batch or dynamic_width:
        image_axes = {}
        output_axes = {}
        shape_spec = {}

        if dynamic_batch:
            image_axes[0] = "batch"
            output_axes[0] = "batch"
            shape_spec[0] = torch.export.Dim("batch", min=1, max=max(1, batch_size))

        if dynamic_width:
            image_axes[3] = "width"
            output_axes[1] = "time"
            shape_spec[3] = torch.export.Dim("width", min=32, max=width)

        dynamic_axes = {"images": image_axes, "log_probs": output_axes}
        # torch.export uses the Python forward argument name ("x"), while ONNX
        # input_names uses the exported tensor name ("images").
        dynamic_shapes = {"x": shape_spec}

    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        input_names=["images"],
        output_names=["log_probs"],
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        do_constant_folding=True,
    )
    return output_path


def check_onnx_model(path: str | Path) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def validate_onnx_parity(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    *,
    batch_size: int = 1,
    width: int = 512,
    atol: float = 2e-3,
    rtol: float = 2e-3,
) -> dict:
    import onnxruntime as ort

    torch.manual_seed(42)
    model = load_model(checkpoint_path)
    sample = torch.randn(batch_size, 1, 32, width, dtype=torch.float32)

    with torch.inference_mode():
        torch_output = model(sample).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["log_probs"], {"images": sample.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_output - ort_output)))
    mean_abs_diff = float(np.mean(np.abs(torch_output - ort_output)))
    ok = bool(np.allclose(torch_output, ort_output, atol=atol, rtol=rtol))
    return {
        "ok": ok,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "torch_shape": list(torch_output.shape),
        "onnx_shape": list(ort_output.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints_2/best_model.pth")
    parser.add_argument("--output", default="onnx_models/svtr_ocr_w512.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch and width.")
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--dynamic-width", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = export_onnx(
        args.checkpoint,
        args.output,
        opset=args.opset,
        batch_size=args.batch_size,
        width=args.width,
        dynamic_batch=args.dynamic or args.dynamic_batch,
        dynamic_width=args.dynamic or args.dynamic_width,
    )
    print(f"Exported: {onnx_path}")

    if not args.skip_check:
        check_onnx_model(onnx_path)
        parity = validate_onnx_parity(
            args.checkpoint,
            onnx_path,
            batch_size=args.batch_size,
            width=args.width,
        )
        print("Parity:", parity)
        if not parity["ok"]:
            raise SystemExit("ONNX parity check failed")


if __name__ == "__main__":
    main()
