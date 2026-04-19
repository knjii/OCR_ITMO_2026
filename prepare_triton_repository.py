"""Prepare a Triton model repository for ONNX and TensorRT OCR models."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ONNX_CONFIG = """name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [1, 32, 512]
  }}
]

output [
  {{
    name: "log_probs"
    data_type: TYPE_FP32
    dims: [128, 53]
  }}
]

dynamic_batching {{
  preferred_batch_size: [{max_batch_size}]
  max_queue_delay_microseconds: {queue_delay_us}
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""


TRT_CONFIG = """name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [1, 32, 512]
  }}
]

output [
  {{
    name: "log_probs"
    data_type: TYPE_FP32
    dims: [128, 53]
  }}
]

dynamic_batching {{
  preferred_batch_size: [{max_batch_size}]
  max_queue_delay_microseconds: {queue_delay_us}
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""


def copy_onnx_with_external_data(source_onnx: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_onnx, target_dir / "model.onnx")

    external_data = source_onnx.with_suffix(source_onnx.suffix + ".data")
    if external_data.exists():
        # The ONNX file references the original external-data filename. Keep it
        # in the same version directory so ONNX Runtime/Triton can resolve it.
        shutil.copy2(external_data, target_dir / external_data.name)


def prepare_repository(
    source_onnx: str | Path,
    repository_dir: str | Path = "triton_model_repository",
    *,
    onnx_model_name: str = "svtr_ocr_onnx",
    trt_model_name: str = "svtr_ocr_trt",
    max_batch_size: int = 2,
    queue_delay_us: int = 1000,
    plan_path: str | Path | None = None,
    write_trt_placeholder: bool = False,
) -> Path:
    source_onnx = Path(source_onnx)
    repository_dir = Path(repository_dir)

    if not source_onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {source_onnx}")

    onnx_model_dir = repository_dir / onnx_model_name
    onnx_version_dir = onnx_model_dir / "1"
    copy_onnx_with_external_data(source_onnx, onnx_version_dir)
    (onnx_model_dir / "config.pbtxt").write_text(
        ONNX_CONFIG.format(
            model_name=onnx_model_name,
            max_batch_size=max_batch_size,
            queue_delay_us=queue_delay_us,
        ),
        encoding="utf-8",
    )

    if plan_path is not None or write_trt_placeholder:
        trt_model_dir = repository_dir / trt_model_name
        trt_version_dir = trt_model_dir / "1"
        trt_version_dir.mkdir(parents=True, exist_ok=True)
        (trt_model_dir / "config.pbtxt").write_text(
            TRT_CONFIG.format(
                model_name=trt_model_name,
                max_batch_size=max_batch_size,
                queue_delay_us=queue_delay_us,
            ),
            encoding="utf-8",
        )

    if plan_path is not None:
        plan_path = Path(plan_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"TensorRT plan not found: {plan_path}")
        shutil.copy2(plan_path, trt_version_dir / "model.plan")

    return repository_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="onnx_models/svtr_ocr_w512_b2_dynamic.onnx")
    parser.add_argument("--repository", default="triton_model_repository")
    parser.add_argument("--onnx-model-name", default="svtr_ocr_onnx")
    parser.add_argument("--trt-model-name", default="svtr_ocr_trt")
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--queue-delay-us", type=int, default=1000)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--write-trt-placeholder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = prepare_repository(
        args.onnx,
        args.repository,
        onnx_model_name=args.onnx_model_name,
        trt_model_name=args.trt_model_name,
        max_batch_size=args.max_batch_size,
        queue_delay_us=args.queue_delay_us,
        plan_path=args.plan,
        write_trt_placeholder=args.write_trt_placeholder,
    )
    print(f"Prepared Triton repository: {repo}")
    print(f"ONNX model: {repo / args.onnx_model_name}")
    if args.plan or args.write_trt_placeholder:
        print(f"TensorRT model config: {repo / args.trt_model_name}")
    else:
        print("TensorRT entry was not created because --plan was not provided.")
        print("Build model.plan first, then rerun with --plan path/to/model.plan.")


if __name__ == "__main__":
    main()
