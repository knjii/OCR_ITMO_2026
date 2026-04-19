"""Build a TensorRT engine (.plan) from the exported OCR ONNX model.

Requires NVIDIA TensorRT `trtexec` on PATH. Build engines on the same GPU
family where they will be served.
"""

from __future__ import annotations

import argparse
import shlex
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
    use_docker: bool = False,
    docker_image: str = "nvcr.io/nvidia/tritonserver:24.12-py3",
    trtexec_path: str = "trtexec",
) -> Path:
    trtexec = shutil.which(trtexec_path)
    if trtexec is None and not use_docker:
        raise FileNotFoundError(
            "trtexec was not found on PATH. Install TensorRT locally or run this "
            "script with --use-docker."
        )

    onnx_path = Path(onnx_path)
    output_plan = Path(output_plan)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_plan.parent.mkdir(parents=True, exist_ok=True)

    min_shape = f"images:1x1x32x{width}"
    opt_shape = f"images:{max_batch_size}x1x32x{width}"
    max_shape = f"images:{max_batch_size}x1x32x{width}"

    trtexec_args = [
        f"--onnx={onnx_path}",
        f"--saveEngine={output_plan}",
        f"--minShapes={min_shape}",
        f"--optShapes={opt_shape}",
        f"--maxShapes={max_shape}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--verbose",
    ]
    if fp16:
        trtexec_args.append("--fp16")

    if use_docker:
        command = _docker_trtexec_command(
            trtexec_args,
            docker_image=docker_image,
            trtexec_path=trtexec_path,
        )
    else:
        command = [trtexec, *trtexec_args]

    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)
    return output_plan


def _docker_trtexec_command(
    trtexec_args: list[str],
    *,
    docker_image: str,
    trtexec_path: str,
) -> list[str]:
    cwd = Path.cwd().resolve()
    container_args: list[str] = []

    for arg in trtexec_args:
        if arg.startswith("--onnx=") or arg.startswith("--saveEngine="):
            name, value = arg.split("=", 1)
            path = Path(value).resolve()
            try:
                relative = path.relative_to(cwd)
            except ValueError as exc:
                raise ValueError(
                    f"Docker mode expects model paths inside the current directory: {cwd}"
                ) from exc
            container_args.append(f"{name}=/workspace/{relative.as_posix()}")
        else:
            container_args.append(arg)

    command_string = " ".join(
        [shlex.quote(trtexec_path), *[shlex.quote(arg) for arg in container_args]]
    )
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{cwd.as_posix()}:/workspace",
        "-w",
        "/workspace",
        "--entrypoint",
        "/bin/bash",
        docker_image,
        "-lc",
        command_string,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="triton_model_repository/svtr_ocr_onnx/1/model.onnx")
    parser.add_argument("--output", default="triton_model_repository/svtr_ocr_trt/1/model.plan")
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 engine build.")
    parser.add_argument("--use-docker", action="store_true", help="Run trtexec in Docker.")
    parser.add_argument("--docker-image", default="nvcr.io/nvidia/tritonserver:24.12-py3")
    parser.add_argument(
        "--trtexec-path",
        default="trtexec",
        help="Path to trtexec locally or inside the Docker image.",
    )
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
        use_docker=args.use_docker,
        docker_image=args.docker_image,
        trtexec_path=args.trtexec_path,
    )
    print(f"Built TensorRT plan: {plan}")


if __name__ == "__main__":
    main()
