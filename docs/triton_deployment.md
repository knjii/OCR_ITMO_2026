# Triton Deployment: SVTR OCR

## Model Repository Layout

ONNX model:

```text
triton_model_repository/
  svtr_ocr_onnx/
    config.pbtxt
    1/
      model.onnx
      svtr_ocr_w512_b2_dynamic.onnx.data
```

TensorRT model after engine build:

```text
triton_model_repository/
  svtr_ocr_trt/
    config.pbtxt
    1/
      model.plan
```

ONNX and TensorRT are separate Triton models because they use different Triton backends:

- `svtr_ocr_onnx`: `onnxruntime_onnx`
- `svtr_ocr_trt`: `tensorrt_plan`

Do not put `.onnx` and `.plan` as versions of the same Triton model name.

## Prepare ONNX Repository

```powershell
python prepare_triton_repository.py `
  --onnx onnx_models\svtr_ocr_w512_b2_dynamic.onnx `
  --repository triton_model_repository `
  --max-batch-size 2
```

## Build TensorRT Plan

Build the TensorRT engine on the same GPU family where it will be served.

```powershell
python build_tensorrt_plan.py `
  --onnx triton_model_repository\svtr_ocr_onnx\1\model.onnx `
  --output triton_model_repository\svtr_ocr_trt\1\model.plan `
  --max-batch-size 2 `
  --width 512
```

If `trtexec` is not available locally, run the command inside a TensorRT/NVIDIA container with the repository mounted.

After `model.plan` is built, add the TensorRT model config:

```powershell
python prepare_triton_repository.py `
  --onnx onnx_models\svtr_ocr_w512_b2_dynamic.onnx `
  --repository triton_model_repository `
  --max-batch-size 2 `
  --plan triton_model_repository\svtr_ocr_trt\1\model.plan
```

## Start Triton

Linux/Docker example:

```bash
docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/triton_model_repository:/models" \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models
```

Windows PowerShell path example:

```powershell
docker run --rm --gpus all `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}\triton_model_repository:/models" `
  nvcr.io/nvidia/tritonserver:latest `
  tritonserver --model-repository=/models
```

## Check Model Readiness

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/svtr_ocr_onnx/ready
curl http://localhost:8000/v2/models/svtr_ocr_trt/ready
```

## Run Triton OCR Client

Install client dependency:

```powershell
pip install "tritonclient[http]"
```

ONNX backend:

```powershell
python triton_ocr_client.py `
  --url localhost:8000 `
  --model-name svtr_ocr_onnx `
  --image Datasets\300dpi\tiff\a013.tiff `
  --output-json triton_ocr_result.json
```

TensorRT backend:

```powershell
python triton_ocr_client.py `
  --url localhost:8000 `
  --model-name svtr_ocr_trt `
  --image Datasets\300dpi\tiff\a013.tiff `
  --output-json triton_ocr_result_trt.json
```

## Shape Contract

FastAPI/Triton client sends:

```text
images: [B, 1, 32, 512], FP32
```

Triton returns:

```text
log_probs: [B, 128, 53], FP32
```

Postprocessing outside Triton:

```text
argmax -> remove repeated labels -> remove blank=0 -> text
```

## Batching

Current config:

```text
max_batch_size: 2
preferred_batch_size: [2]
max_queue_delay_microseconds: 1000
```

This enables Triton dynamic batching for independent line/chunk requests. For this stateless OCR model, use Triton dynamic batching, not LLM-style continuous batching.
