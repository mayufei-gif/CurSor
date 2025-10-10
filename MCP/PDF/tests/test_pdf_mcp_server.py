"""Lightweight tests for the PDF MCP server entry points.

These tests focus on behaviour that can be validated without launching the full
FastAPI application or requiring heavy PDF processing dependencies. They also
serve as regression checks for the issues highlighted in CI: ensuring the server
lifecycle helpers behave correctly and that registered MCP tools stay in sync
with the public processing modes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Iterable

import pytest

try:
    from fastapi import HTTPException
except ImportError:  # pragma: no cover - optional dependency for local runs
    pytest.skip("fastapi is required for pdf_mcp_server tests", allow_module_level=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_mcp_server import main as server_main
from src.pdf_mcp_server.models import ProcessingMode


def test_health_check_reports_healthy_status() -> None:
    """The health endpoint should always report a healthy status."""

    response = asyncio.run(server_main.health_check())

    assert response.status == "healthy"
    # uptime is reported in seconds and should be non-negative
    assert response.uptime >= 0
    # timestamp should carry timezone-aware datetime information
    assert response.timestamp.tzinfo is not None


def test_get_pdf_processor_requires_initialisation(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_pdf_processor`` must raise when the processor is not ready."""

    monkeypatch.setattr(server_main, "pdf_processor", None, raising=False)

    with pytest.raises(HTTPException) as excinfo:
        server_main.get_pdf_processor()

    assert excinfo.value.status_code == 503
    assert "not initialized" in excinfo.value.detail


def test_registered_tools_cover_public_processing_modes() -> None:
    """Each processing mode should have a corresponding MCP tool."""

    tools = asyncio.run(server_main.list_tools())
    tool_names = {tool.name for tool in tools}

    expected_names: Iterable[str] = (
        ProcessingMode.TEXT.value,
        ProcessingMode.TABLES.value,
        ProcessingMode.FORMULAS.value,
        ProcessingMode.FULL.value,
    )

    for mode_name in expected_names:
        assert mode_name in tool_names, f"Missing MCP tool for mode '{mode_name}'"


def test_health_check_invocation_is_isolated() -> None:
    """Calling ``health_check`` concurrently should not mutate shared state."""

    async def invoke_many() -> list:
        return await asyncio.gather(*(server_main.health_check() for _ in range(5)))

    # Run multiple invocations to ensure no unexpected state leakage.
    results = asyncio.run(invoke_many())

    assert all(result.status == "healthy" for result in results)
    # ensure uptime is monotonically increasing across calls
    uptimes = [result.uptime for result in results]
    assert uptimes == sorted(uptimes)
