"""Focused smoke tests for the HTTP DOCX export endpoint.

These tests avoid re-implementing the full FastAPI application logic and
instead validate that the `/extract` endpoint orchestrates the processor and
DOCX exporter correctly for both uploaded files and direct file paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from pdf_mcp_server import http_app
from pdf_mcp_server.models import ProcessingMode, ProcessingRequest


@dataclass
class _FakeResult:
    payload: Dict[str, object]

    def model_dump(self) -> Dict[str, object]:
        return self.payload


@pytest.fixture()
def client_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Tuple[TestClient, AsyncMock, List[Tuple[Dict[str, object], Path]]]:
    """Provide a TestClient with fake processor/config/exporter wiring.

    The FastAPI module keeps global state, so we monkeypatch it for the duration
    of the test and clean up afterwards to avoid cross-test contamination.
    """

    calls: List[Tuple[Dict[str, object], Path]] = []

    fake_processor = AsyncMock()
    fake_processor.process.return_value = _FakeResult({"doc": "ok"})
    http_app._processor = fake_processor
    http_app._config = object()

    def fake_export(data: Dict[str, object], destination: Path) -> None:
        calls.append((data, destination))
        destination.write_bytes(b"DOCX")

    monkeypatch.setattr(http_app, "build_docx_from_pipeline", fake_export)

    client = TestClient(http_app.app)
    try:
        yield client, fake_processor, calls
    finally:
        client.close()
        http_app._processor = None
        http_app._config = None


def test_extract_docx_upload(tmp_path: Path, client_with_fakes):
    client, processor, exporter_calls = client_with_fakes

    upload = tmp_path / "sample.pdf"
    upload.write_bytes(b"%PDF-sample")

    with upload.open("rb") as handle:
        response = client.post(
            "/extract",
            params={"output_format": "docx", "mode": ProcessingMode.TEXT.value},
            files={"file": (upload.name, handle, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert response.content == b"DOCX"

    assert processor.process.await_count == 1
    await_call = processor.process.await_args
    request_arg = await_call.args[0]
    assert isinstance(request_arg, ProcessingRequest)
    assert request_arg.mode is ProcessingMode.TEXT
    request_path = Path(request_arg.file_path)
    assert request_path.suffix == ".pdf"

    assert len(exporter_calls) == 1
    exported_payload, exported_path = exporter_calls[0]
    assert exported_payload == {"doc": "ok"}
    assert exported_path.exists()


def test_extract_docx_from_local_path(tmp_path: Path, client_with_fakes):
    client, processor, exporter_calls = client_with_fakes

    source = tmp_path / "existing.pdf"
    source.write_bytes(b"PDF")

    response = client.post(
        "/extract",
        params={"output_format": "docx", "file_path": str(source)},
    )

    assert response.status_code == 200
    assert response.content == b"DOCX"

    assert processor.process.await_count == 1
    await_call = processor.process.await_args
    request_arg = await_call.args[0]
    assert request_arg.file_path == str(source)
    assert request_arg.mode is ProcessingMode.FULL

    assert exporter_calls and exporter_calls[0][1].exists()
