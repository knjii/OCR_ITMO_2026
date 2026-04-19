import requests

from frontend.settings import API_BASE_URL


def _multipart_file(uploaded_file):
    return {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            getattr(uploaded_file, "type", "application/octet-stream"),
        )
    }


def _multipart_files(uploaded_files):
    return [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                getattr(uploaded_file, "type", "application/octet-stream"),
            ),
        )
        for uploaded_file in uploaded_files
    ]


def _raise_for_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = response.text
        try:
            payload = response.json()
            message = payload.get("message") or payload.get("detail") or message
        except ValueError:
            pass
        raise requests.HTTPError(message, response=response) from exc


def get_health(timeout: int = 10) -> dict:
    response = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
    _raise_for_status(response)
    return response.json()


def post_ocr_page(
    uploaded_file,
    return_lines: bool,
    return_parts: bool,
    overlap: int,
    batch_size: int = 2,
    timeout: int = 120,
) -> dict:
    data = {
        "return_lines": str(bool(return_lines)).lower(),
        "return_parts": str(bool(return_parts)).lower(),
        "overlap": str(int(overlap)),
        "batch_size": str(int(batch_size)),
    }

    response = requests.post(
        f"{API_BASE_URL}/v1/ocr/page",
        files=_multipart_file(uploaded_file),
        data=data,
        timeout=timeout,
    )
    _raise_for_status(response)
    return response.json()


def post_ocr_pages(
    uploaded_files,
    return_lines: bool,
    return_parts: bool,
    overlap: int,
    batch_size: int = 2,
    timeout: int = 180,
) -> dict:
    data = {
        "return_lines": str(bool(return_lines)).lower(),
        "return_parts": str(bool(return_parts)).lower(),
        "overlap": str(int(overlap)),
        "batch_size": str(int(batch_size)),
    }

    response = requests.post(
        f"{API_BASE_URL}/v1/ocr/pages",
        files=_multipart_files(uploaded_files),
        data=data,
        timeout=timeout,
    )
    _raise_for_status(response)
    return response.json()


def post_preview(uploaded_file, timeout: int = 60) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/v1/ocr/page/preview",
        files=_multipart_file(uploaded_file),
        timeout=timeout,
    )
    _raise_for_status(response)
    return response.json()
