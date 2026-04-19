"""Build a TensorRT engine (.plan) from the exported OCR ONNX model.

Requires NVIDIA TensorRT `trtexec` on PATH. Build engines on the same GPU
family where they will be served.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_plan(
    onnx_path: str | Path = "triton_model_repository/svtr_ocr_onnx/1/model.onnx",
    output_plan: str | Path = "triton_model_repository/svtr_ocr_trt/1/model.plan",
    *,
    max_batch_size: int = 2,
    width: int = 512,
    fp16: bool = True,
    workspace_mb: int = 2048,
) -> Path:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise FileNotFoundError(
            "trtexec was not found on PATH. Install TensorRT locally or run this "
            "script inside a TensorRT/NVIDIA container."
        )

    onnx_path = Path(onnx_path)
    output_plan = Path(output_plan)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_plan.parent.mkdir(parents=True, exist_ok=True)

    min_shape = f"images:1x1x32x{width}"
    opt_shape = f"images:{max_batch_size}x1x32x{width}"
    max_shape = f"images:{max_batch_size}x1x32x{width}"

    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_plan}",
        f"--minShapes={min_shape}",
        f"--optShapes={opt_shape}",
        f"--maxShapes={max_shape}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--verbose",
    ]
    if fp16:
        command.append("--fp16")

    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)
    return output_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="triton_model_repository/svtr_ocr_onnx/1/model.onnx")
    parser.add_argument("--output", default="triton_model_repository/svtr_ocr_trt/1/model.plan")
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 engine build.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(
        onnx_path=args.onnx,
        output_plan=args.output,
        max_batch_size=args.max_batch_size,
        width=args.width,
        fp16=not args.fp32,
        workspace_mb=args.workspace_mb,
    )
    print(f"Built TensorRT plan: {plan}")


if __name__ == "__main__":
    main()
