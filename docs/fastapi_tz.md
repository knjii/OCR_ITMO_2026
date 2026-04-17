# ТЗ: FastAPI OCR Backend

## Назначение

FastAPI-сервис принимает изображение страницы или строки машинопечатного английского текста, выполняет preprocessing/page segmentation/chunking, вызывает OCR-модель через Triton Inference Server и возвращает распознанный текст с координатами строк.

FastAPI не хранит PyTorch-модель в памяти. Взаимодействие с нейросетью идет через Triton.

## Triton Contract

Модель: `svtr_ocr`

ONNX input:

```text
name: images
shape: [1, 1, 32, 512]
dtype: FP32
```

ONNX output:

```text
name: log_probs
shape: [1, 128, 53]
dtype: FP32
```

FastAPI отвечает за:

- чтение изображения;
- grayscale conversion;
- line detection;
- resize line/chunk to `H=32`;
- padding to `W=512`;
- Triton request;
- CTC greedy decode;
- merge chunks;
- сбор page-level ответа.

## Pydantic Models

```python
from pydantic import BaseModel, Field


class OCRRequestOptions(BaseModel):
    return_lines: bool = True
    return_parts: bool = False
    max_width: int = Field(default=512, ge=128, le=512)
    overlap: int = Field(default=64, ge=0, le=256)
    binarize: bool = False


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class OCRPart(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox
    width: int


class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox
    parts: list[OCRPart] = []


class OCRPageResponse(BaseModel):
    text: str
    lines: list[OCRLine] = []
    processing_ms: float
    model_name: str = "svtr_ocr"
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    triton_ready: bool
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    alphabet_size: int
    input_shape: list[int]
    output_shape: list[int]
    max_width: int
    image_height: int


class ErrorResponse(BaseModel):
    detail: str
    code: str
```

## Методы API

### `GET /health`

Проверяет FastAPI, Triton и доступность модели.

Response `200`:

```json
{
  "status": "ok",
  "triton_ready": true,
  "model_ready": true
}
```

### `GET /v1/model/info`

Возвращает информацию о модели и shape-контракте.

Response `200`:

```json
{
  "model_name": "svtr_ocr",
  "alphabet_size": 53,
  "input_shape": [1, 1, 32, 512],
  "output_shape": [1, 128, 53],
  "max_width": 512,
  "image_height": 32
}
```

### `POST /v1/ocr/page`

Основной endpoint для фронтенда. Принимает изображение страницы.

Request:

```text
multipart/form-data
file: UploadFile (.png, .jpg, .jpeg, .tif, .tiff)
options: optional JSON string matching OCRRequestOptions
```

Response `200`: `OCRPageResponse`

```json
{
  "text": "why and wherefore.\nin making a study ...",
  "lines": [
    {
      "text": "why and wherefore.",
      "confidence": 0.98,
      "bbox": {"x1": 460, "y1": 584, "x2": 1279, "y2": 630},
      "parts": []
    }
  ],
  "processing_ms": 842.1,
  "model_name": "svtr_ocr",
  "model_version": "1"
}
```

### `POST /v1/ocr/line`

Отладочный endpoint для распознавания одной уже вырезанной строки.

Request:

```text
multipart/form-data
file: UploadFile
options: optional JSON string matching OCRRequestOptions
```

Response `200`: `OCRPageResponse`, где `lines` содержит одну строку.

### `POST /v1/ocr/page/preview`

Отладочный endpoint без вызова Triton. Возвращает найденные line bounding boxes, чтобы фронтенд мог показать качество сегментации.

Response:

```json
{
  "lines": [
    {"x1": 460, "y1": 584, "x2": 1279, "y2": 630}
  ],
  "processing_ms": 120.4
}
```

## Ошибки

- `400`: файл не является изображением или не может быть прочитан.
- `413`: изображение слишком большое.
- `422`: некорректные options.
- `503`: Triton или модель недоступны.
- `500`: внутренняя ошибка inference pipeline.

Формат:

```json
{
  "detail": "Triton model svtr_ocr is not ready",
  "code": "MODEL_NOT_READY"
}
```

## Нефункциональные Требования

- Максимальный размер входного файла: 20 MB.
- Поддерживаемые форматы: PNG, JPEG, TIFF.
- Таймаут OCR-запроса: 60 секунд.
- Логировать: request id, filename, image size, number of lines, latency, Triton latency.
- Не логировать содержимое распознанного текста по умолчанию.
- CORS настроить под адрес Streamlit-фронтенда.

## Definition Of Done

- `/health` возвращает `model_ready=true`.
- `/v1/ocr/page` обрабатывает страницу из old-books-dataset.
- Ответ содержит page text и bbox строк.
- Ошибки Triton корректно превращаются в HTTP `503`.
- Есть unit-тест для CTC decode и integration smoke-test на одной странице.
