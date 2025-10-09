"""Tests covering CLI flags for layout reconstruction."""

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import types

from pdf_mcp_server.models import ProcessingRequest

stub_main = types.ModuleType("pdf_mcp_server.main")

def _noop_main(*_args, **_kwargs):  # pragma: no cover - unused in tests
    return None

stub_main.main = _noop_main
sys.modules.setdefault("pdf_mcp_server.main", stub_main)

from pdf_mcp_server import cli as cli_module


class DummyProcessor:
    instances = []
    requests = []

    def __init__(self, config):
        self.config = config
        self._requests = []
        DummyProcessor.instances.append(self)

    async def initialize(self):  # pragma: no cover - trivial
        return None

    async def process(self, request):
        self._requests.append(request)
        DummyProcessor.requests.append(request)

        class DummyResult:
            def __init__(self, req):
                self._req = req

            def model_dump(self, *_, **__):
                return {
                    "file": self._req.file_path,
                    "reconstruct_layout": self._req.reconstruct_layout,
                }

            def __str__(self):  # pragma: no cover - CLI fallback
                return json.dumps(self.model_dump())

        return DummyResult(request)

    async def cleanup(self):  # pragma: no cover - trivial
        return None


@pytest.fixture(autouse=True)
def reset_dummy_processor(monkeypatch):
    DummyProcessor.instances.clear()
    DummyProcessor.requests.clear()

    monkeypatch.setattr(cli_module, "PDFProcessor", DummyProcessor)

    def _load_config(cls, *_args, **_kwargs):
        return cli_module.Config()

    monkeypatch.setattr(
        cli_module.Config,
        "load_config",
        classmethod(_load_config),
        raising=False,
    )


def _create_pdf(tmp_path: Path, name: str = "sample.pdf") -> Path:
    pdf_path = tmp_path / name
    pdf_path.write_bytes(b"%PDF-1.4\n%\xff\xff\xff\xff\n")
    return pdf_path


def test_process_command_sets_reconstruct_layout(tmp_path):
    runner = CliRunner()
    pdf_path = _create_pdf(tmp_path)

    result = runner.invoke(
        cli_module.cli,
        [
            "--quiet",
            "process",
            "--file",
            str(pdf_path),
            "--mode",
            "full_pipeline",
            "--reconstruct-layout",
        ],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output
    assert DummyProcessor.requests, "process() should be invoked"
    assert DummyProcessor.requests[0].reconstruct_layout is True


def test_batch_command_passes_reconstruct_layout(tmp_path):
    runner = CliRunner()
    pdf_one = _create_pdf(tmp_path, "one.pdf")
    pdf_two = _create_pdf(tmp_path, "two.pdf")

    result = runner.invoke(
        cli_module.cli,
        [
            "--quiet",
            "batch",
            "--files",
            str(pdf_one),
            "--files",
            str(pdf_two),
            "--reconstruct-layout",
        ],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output
    assert len(DummyProcessor.requests) == 2
    assert all(req.reconstruct_layout for req in DummyProcessor.requests)

    # When quiet mode is enabled the CLI prints a JSON summary; ensure it is valid.
    summary = json.loads(result.output)
    assert summary["successful"] == 2
