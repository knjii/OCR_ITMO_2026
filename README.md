# OCR ITMO 2026

Проект для распознавания английского машинопечатного текста на сканах старопечатных книг.

Основной inference pipeline:

1. Пользователь загружает изображение страницы через Streamlit или Swagger.
2. FastAPI принимает файл, валидирует его и вызывает OCR-клиент.
3. OCR-клиент находит строки текста на странице, режет слишком длинные строки на чанки и отправляет батчи в Triton Inference Server.
4. Triton выполняет модель SVTR OCR в формате ONNX или TensorRT.
5. Backend декодирует CTC-выход, склеивает чанки обратно в строки и возвращает текст, bbox строк и confidence.

## Состав проекта

```text
OCR_ITMO_2026/
├── backend/                         # FastAPI backend
├── frontend/                        # вспомогательные модули Streamlit UI
├── triton_model_repository/         # модели для Triton
│   ├── svtr_ocr_onnx/               # ONNX Runtime модель
│   └── svtr_ocr_trt/                # TensorRT plan, локально/опционально
├── ocr_model.py                     # PyTorch SVTR-Tiny OCR архитектура
├── ocr_runtime.py                   # preprocessing/postprocessing без PyTorch
├── triton_ocr_client.py             # клиент для Triton inference
├── inference.py                     # локальный PyTorch inference
├── export_onnx.py                   # экспорт checkpoint -> ONNX
├── prepare_triton_repository.py     # подготовка model repository для Triton
├── build_tensorrt_plan.py           # сборка TensorRT .plan из ONNX
├── prepare_aligned_manifest.py      # подготовка line manifest для обучения
├── Train.py                         # функции обучения, валидации и метрик
├── train_ocr.model.ipynb            # notebook обучения
├── streamlit_app.py                 # Streamlit frontend
├── docker-compose.yml               # запуск Triton + FastAPI + Streamlit
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

## Требования

Для запуска полного сервиса:

- Windows/Linux с Docker.
- NVIDIA GPU.
- Установленный NVIDIA Driver.
- Установленный NVIDIA Container Toolkit.
- Docker Compose v2.
- Доступ к образу `nvcr.io/nvidia/tritonserver:24.12-py3`.

Для разработки и локальных скриптов:

- Python 3.10+.
- PyTorch 2.0+ для обучения, локального inference и ONNX export.
- OpenCV, NumPy, ONNX/ONNX Runtime, Triton HTTP client.

Проверка GPU в Docker:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если команда не видит GPU, сначала нужно исправить Docker/NVIDIA runtime. Triton без GPU в текущей конфигурации проекта не поднимется корректно.

## Быстрый запуск всего проекта

Из корня проекта:

```powershell
cd C:\python\ocr_itmo\OCR_ITMO_2026
docker compose up --build
```

После запуска будут доступны:

- Streamlit UI: `http://127.0.0.1:8501`
- FastAPI Swagger: `http://127.0.0.1:8080/docs`
- FastAPI health: `http://127.0.0.1:8080/health`
- Triton HTTP: `http://127.0.0.1:8000`
- Triton gRPC: `127.0.0.1:8001`
- Triton metrics: `http://127.0.0.1:8002/metrics`

Остановить сервисы:

```powershell
docker compose down
```

Пересобрать только backend после изменений preprocessing/API:

```powershell
docker compose build backend
docker compose up -d backend
```

Пересобрать только frontend:

```powershell
docker compose build frontend
docker compose up -d frontend
```

Посмотреть логи:

```powershell
docker compose logs -f triton
docker compose logs -f backend
docker compose logs -f frontend
```

## Проверка работоспособности

Health check backend:

```powershell
curl.exe http://127.0.0.1:8080/health
```

Ожидаемый ответ при рабочем Triton:

```json
{
  "status": "ok",
  "triton_url": "triton:8000",
  "model_name": "svtr_ocr_onnx",
  "model_version": "1",
  "triton_ready": true,
  "model_ready": true,
  "message": null
}
```

Информация о модели:

```powershell
curl.exe http://127.0.0.1:8080/v1/model/info
```

Preview разбиения страницы на строки:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/v1/ocr/page/preview" `
  -F "file=@Datasets/300dpi/tiff/a027.tiff"
```

OCR одной страницы:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/v1/ocr/page" `
  -F "file=@Datasets/300dpi/tiff/a027.tiff" `
  -F "return_lines=true" `
  -F "return_parts=false" `
  -F "overlap=64" `
  -F "batch_size=2"
```

OCR нескольких страниц:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/v1/ocr/pages" `
  -F "files=@Datasets/300dpi/tiff/a027.tiff" `
  -F "files=@Datasets/300dpi/tiff/a028.tiff" `
  -F "return_lines=true" `
  -F "return_parts=false" `
  -F "overlap=64" `
  -F "batch_size=2"
```

## Streamlit frontend

Запускается контейнером `frontend` и доступен по адресу:

```text
http://127.0.0.1:8501
```

В UI можно:

- загрузить одну или несколько страниц;
- посмотреть статус backend/Triton;
- отправить страницы на OCR;
- получить текст;
- посмотреть line boxes через preview;
- скачать результат.

В Docker Compose frontend ходит в backend по внутреннему адресу:

```text
API_BASE_URL=http://backend:8080
```

При локальном запуске Streamlit без Docker можно задать:

```powershell
$env:API_BASE_URL="http://127.0.0.1:8080"
python -m streamlit run streamlit_app.py
```

## FastAPI endpoints

### `GET /health`

Проверяет доступность backend, Triton и выбранной модели.

### `GET /v1/model/info`

Возвращает параметры текущей модели:

- `model_name`
- `model_version`
- `max_width`
- `default_batch_size`
- `max_batch_size`
- `alphabet_size`
- `blank_index`

### `POST /v1/ocr/page/preview`

Вход:

- `file`: изображение страницы.

Выход:

- размер изображения;
- количество найденных строк;
- bbox каждой строки;
- время обработки.

Эта ручка не вызывает модель. Она нужна для отладки line segmentation.

### `POST /v1/ocr/page`

Вход form-data:

- `file`: изображение страницы;
- `return_lines`: вернуть OCR по строкам;
- `return_parts`: вернуть чанки длинных строк;
- `overlap`: overlap при нарезке длинной строки, по умолчанию `64`;
- `batch_size`: размер батча для запросов в Triton, по умолчанию `2`.

Выход:

- полный текст страницы;
- список строк с bbox и confidence;
- имя и версия модели;
- время обработки.

### `POST /v1/ocr/pages`

То же, что `/v1/ocr/page`, но принимает несколько файлов в поле `files`.

Используется frontend для пакетной обработки нескольких страниц.

### `POST /v1/ocr/line`

Распознает уже вырезанную строку. Используется для тестов и точечной отладки модели.

## Как работает OCR страницы

Модель обучена как line recognizer, поэтому страница обрабатывается в несколько этапов:

1. `ocr_runtime.extract_line_boxes_horizontal_projection` строит горизонтальный projection profile.
2. По активным строкам пикселей находятся bbox текстовых строк.
3. Слишком высокие bbox дополнительно делятся, если внутри есть горизонтальные паузы.
4. Каждая строка приводится к высоте `32 px`.
5. Если после resize ширина строки превышает `512 px`, строка режется на чанки с overlap.
6. Чанки отправляются в Triton батчами.
7. CTC output декодируется greedy-декодером.
8. Текст чанков склеивается обратно в строку.
9. Строки собираются в текст страницы через `\n`.

Важно: ошибки вида склейки нескольких строк обычно относятся к preprocessing. Ошибки вида `blody` вместо `bloody` относятся уже к качеству модели/датасета.

## Triton model repository

Triton ожидает такую структуру:

```text
triton_model_repository/
├── svtr_ocr_onnx/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.onnx
│       └── svtr_ocr_w512_b2_dynamic.onnx.data
└── svtr_ocr_trt/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

ONNX-модель должна быть доступна в репозитории, иначе Triton не сможет загрузить `svtr_ocr_onnx`.

TensorRT `.plan` является hardware-specific артефактом. Его лучше собирать на той же архитектуре GPU, где будет запускаться inference. Поэтому `svtr_ocr_trt/` может не храниться в Git и пересобираться на сервере.

## Выбор ONNX или TensorRT модели

По умолчанию backend использует:

```text
TRITON_MODEL_NAME=svtr_ocr_onnx
```

Запуск с TensorRT:

```powershell
$env:TRITON_MODEL_NAME="svtr_ocr_trt"
docker compose up --build
```

Или одной командой:

```powershell
docker compose --env-file .env up --build
```

Пример `.env`:

```text
TRITON_MODEL_NAME=svtr_ocr_trt
```

Если выбранная модель не готова, `/health` вернет `status=degraded`, а OCR endpoints вернут ошибку `MODEL_NOT_READY`.

## Экспорт checkpoint в ONNX

Установить зависимости для inference/export:

```powershell
python -m pip install -r requirements_inference.txt
```

Экспортировать checkpoint:

```powershell
python export_onnx.py `
  --checkpoint checkpoints_2/best_model.pth `
  --output onnx_models/svtr_ocr_w512_b2_dynamic.onnx `
  --batch-size 2 `
  --width 512 `
  --dynamic-batch
```

Скрипт:

- загружает PyTorch checkpoint;
- экспортирует модель в ONNX;
- проверяет ONNX checker;
- сравнивает выход PyTorch и ONNX Runtime.

Подготовить Triton repository:

```powershell
python prepare_triton_repository.py `
  --onnx onnx_models/svtr_ocr_w512_b2_dynamic.onnx `
  --repository triton_model_repository `
  --max-batch-size 2 `
  --write-trt-placeholder
```

После этого можно запускать Triton через Docker Compose.

## Сборка TensorRT `.plan`

TensorRT engine нужно собирать на целевой GPU или на совместимой GPU той же архитектуры.

Вариант через Docker:

```powershell
python build_tensorrt_plan.py --use-docker
```

Явно указать вход и выход:

```powershell
python build_tensorrt_plan.py `
  --use-docker `
  --onnx triton_model_repository/svtr_ocr_onnx/1/model.onnx `
  --output triton_model_repository/svtr_ocr_trt/1/model.plan `
  --max-batch-size 2 `
  --width 512 `
  --workspace-mb 2048
```

По умолчанию используется FP16. Для FP32:

```powershell
python build_tensorrt_plan.py --use-docker --fp32
```

После сборки проверить, что есть:

```text
triton_model_repository/svtr_ocr_trt/1/model.plan
triton_model_repository/svtr_ocr_trt/config.pbtxt
```

Затем запустить:

```powershell
$env:TRITON_MODEL_NAME="svtr_ocr_trt"
docker compose up --build
```

## Локальный inference без FastAPI

Через Triton client:

```powershell
python triton_ocr_client.py `
  --url localhost:8000 `
  --model-name svtr_ocr_onnx `
  --image Datasets/300dpi/tiff/a027.tiff `
  --batch-size 2 `
  --output-json ocr_result.json
```

Через PyTorch checkpoint:

```powershell
python inference.py `
  --checkpoint checkpoints_2/best_model.pth `
  --image Datasets/300dpi/tiff/a027.tiff `
  --output-json ocr_result.json
```

PyTorch-вариант полезен для сравнения качества до экспорта в ONNX/TensorRT.

## Обучение

Основной notebook:

```text
train_ocr.model.ipynb
```

Основные модули:

- `ocr_model.py`: архитектура SVTR-Tiny OCR;
- `Train.py`: train loop, validation, CTC loss, CER/WER/accuracy;
- `prepare_aligned_manifest.py`: подготовка aligned line manifest;
- `Datasets/Dataclass.py`: dataset/collate/augmentations для обучения.

Данные ожидаются в:

```text
Datasets/300dpi/tiff/
Datasets/groundtruth/
```

Подготовка aligned manifest:

```powershell
python prepare_aligned_manifest.py
```

После обучения checkpoint обычно лежит в:

```text
checkpoints_2/best_model.pth
```

Далее его нужно экспортировать в ONNX и обновить `triton_model_repository`.

## Переменные окружения backend

Backend читает параметры из env:

| Переменная | Значение по умолчанию | Назначение |
| --- | --- | --- |
| `TRITON_URL` | `localhost:8000` | адрес Triton HTTP API |
| `TRITON_MODEL_NAME` | `svtr_ocr_onnx` | имя модели в Triton |
| `TRITON_MODEL_VERSION` | `1` | версия модели |
| `TRITON_BATCH_SIZE` | `2` | batch size по умолчанию |
| `TRITON_MAX_BATCH_SIZE` | `2` | верхний лимит batch size в API |
| `OCR_MAX_WIDTH` | `512` | максимальная ширина входа модели |
| `OCR_OVERLAP` | `64` | overlap для чанков длинных строк |
| `OCR_MAX_UPLOAD_MB` | `20` | лимит размера загружаемого файла |
| `CORS_ORIGINS` | `http://localhost:8501,http://localhost:3000` | разрешенные frontend origins |

В Docker Compose часть переменных уже задана в `docker-compose.yml`.

## Что хранить в Git

Рекомендуется хранить:

- исходный код;
- Dockerfile и `docker-compose.yml`;
- `triton_model_repository/svtr_ocr_onnx/config.pbtxt`;
- `triton_model_repository/svtr_ocr_onnx/1/model.onnx`;
- внешний ONNX data-файл рядом с `model.onnx`, если он создан экспортом.

Рекомендуется не хранить:

- исходный датасет;
- нарезанные строки;
- EasyOCR cache;
- checkpoints `.pth`, если нет Git LFS;
- training history;
- TensorRT `.plan`, если серверная GPU отличается от локальной.

Если нужно, чтобы проект после `git clone` запускался сразу без дополнительных артефактов, ONNX-модель и ее external data-файл должны быть закоммичены или доставлены отдельным release artifact.

## Частые проблемы

### Triton image не скачивается

Ошибка вида:

```text
lookup nvcr.io.nvidia: no such host
```

Обычно это DNS/proxy/network issue. Проверить:

- доступность `nvcr.io`;
- настройки DNS в Docker Desktop;
- корпоративный proxy/VPN;
- ручной `docker pull nvcr.io/nvidia/tritonserver:24.12-py3`.

### Triton образ очень большой

Это нормально. Triton image содержит runtime для разных backend-ов, CUDA/NVIDIA зависимости и Python-окружение. Образ может занимать десятки гигабайт.

### `MODEL_NOT_READY`

Проверить:

```powershell
docker compose logs triton
curl.exe http://127.0.0.1:8080/health
```

Частые причины:

- нет модели в `triton_model_repository`;
- выбран `TRITON_MODEL_NAME=svtr_ocr_trt`, но нет `model.plan`;
- `config.pbtxt` не соответствует форме модели;
- Triton не видит GPU.

### `batch-size must be <= 2`

Текущий ONNX/TensorRT config рассчитан на `max_batch_size: 2`.

Используйте:

```text
batch_size=1
```

или:

```text
batch_size=2
```

Чтобы увеличить batch size, нужно:

1. переэкспортировать ONNX с нужным dynamic batch;
2. обновить `config.pbtxt`;
3. пересобрать TensorRT `.plan`, если используется TRT;
4. выставить `TRITON_BATCH_SIZE` и `TRITON_MAX_BATCH_SIZE`.

### Preview склеивает несколько строк

Это preprocessing. Проверять через:

```text
POST /v1/ocr/page/preview
```

Основная логика находится в:

```text
ocr_runtime.py
```

### OCR ошибается в отдельных буквах

Это уже качество модели, а не backend. Возможные действия:

- дообучить модель на похожих строках;
- улучшить aligned manifest;
- добавить больше примеров старой типографики;
- проверить, что inference использует тот же алфавит и нормализацию, что обучение.

## Рекомендуемый порядок работы с новой моделью

1. Обучить модель в `train_ocr.model.ipynb`.
2. Получить `checkpoints_2/best_model.pth`.
3. Экспортировать ONNX через `export_onnx.py`.
4. Подготовить `triton_model_repository` через `prepare_triton_repository.py`.
5. Запустить `docker compose up --build`.
6. Проверить `/health`.
7. Проверить `/v1/ocr/page/preview` на нескольких страницах.
8. Проверить `/v1/ocr/page`.
9. Если нужен TensorRT, собрать `.plan` через `build_tensorrt_plan.py`.
10. Переключить `TRITON_MODEL_NAME=svtr_ocr_trt` и повторить проверки.

