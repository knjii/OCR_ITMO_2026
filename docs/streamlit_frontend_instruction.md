# Инструкция: Streamlit Frontend Для OCR

## Назначение

Streamlit-приложение дает пользователю простой интерфейс:

1. загрузить изображение страницы;
2. отправить его в FastAPI;
3. показать распознанный текст;
4. показать найденные строки и confidence;
5. дать возможность скачать результат.

## Backend URL

Вынести URL FastAPI в настройку:

```python
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
```

## Основной Сценарий

UI:

- `st.file_uploader("Upload page image", type=["png", "jpg", "jpeg", "tif", "tiff"])`
- checkbox `Return lines`
- checkbox `Return parts`
- slider `Overlap`, default `64`
- button `Run OCR`

Запрос:

```python
import json
import requests

options = {
    "return_lines": return_lines,
    "return_parts": return_parts,
    "max_width": 512,
    "overlap": overlap,
    "binarize": False,
}

files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
data = {"options": json.dumps(options)}

response = requests.post(
    f"{API_BASE_URL}/v1/ocr/page",
    files=files,
    data=data,
    timeout=90,
)
response.raise_for_status()
result = response.json()
```

## Отображение Результата

Основной текст:

```python
st.subheader("Recognized text")
st.text_area("OCR result", value=result["text"], height=400)
```

Метаданные:

```python
st.caption(f"Processing time: {result['processing_ms']:.1f} ms")
st.caption(f"Lines: {len(result.get('lines', []))}")
```

Таблица строк:

```python
import pandas as pd

rows = []
for idx, line in enumerate(result.get("lines", []), start=1):
    bbox = line["bbox"]
    rows.append({
        "line": idx,
        "text": line["text"],
        "confidence": line["confidence"],
        "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)
```

Скачивание:

```python
st.download_button(
    "Download TXT",
    data=result["text"],
    file_name="ocr_result.txt",
    mime="text/plain",
)

st.download_button(
    "Download JSON",
    data=json.dumps(result, ensure_ascii=False, indent=2),
    file_name="ocr_result.json",
    mime="application/json",
)
```

## Preview Сегментации Строк

Для отладки добавить кнопку `Preview line boxes`, которая вызывает:

```text
POST /v1/ocr/page/preview
```

и рисует bounding boxes поверх изображения.

Минимальная визуализация:

```python
from PIL import Image, ImageDraw
import io

image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
draw = ImageDraw.Draw(image)

for line in result.get("lines", []):
    bbox = line["bbox"]
    draw.rectangle(
        [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
        outline="red",
        width=2,
    )

st.image(image, caption="Detected text lines", use_container_width=True)
```

## Обработка Ошибок

Показывать понятные сообщения:

```python
try:
    response = requests.post(...)
    response.raise_for_status()
except requests.Timeout:
    st.error("OCR request timed out. Try a smaller image.")
except requests.HTTPError:
    try:
        payload = response.json()
        st.error(f"{payload.get('code', 'ERROR')}: {payload.get('detail', response.text)}")
    except Exception:
        st.error(response.text)
except requests.RequestException as exc:
    st.error(f"Backend is unavailable: {exc}")
```

## Рекомендуемая Структура Streamlit App

```text
streamlit_app.py
frontend/
  api.py              # requests wrapper
  drawing.py          # bbox rendering helpers
  settings.py         # API_BASE_URL
```

## Acceptance Criteria

- Пользователь может загрузить PNG/JPEG/TIFF.
- После `Run OCR` видит полный распознанный текст.
- Можно скачать TXT и JSON.
- Можно увидеть line-level таблицу с confidence.
- Ошибки backend показываются без traceback.
- URL backend задается через `st.secrets` или env.
