"""Regression tests for MCP exception helpers."""

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pdf_mcp_server.mcp.exceptions import ToolExecutionException


def test_tool_execution_exception_single_argument():
    exc = ToolExecutionException("failure")
    assert exc.execution_error == "failure"
    assert exc.tool_name == "generic_tool"
    assert "failure" in str(exc)


def test_tool_execution_exception_with_tool_name():
    exc = ToolExecutionException("read_text", "boom")
    assert exc.execution_error == "boom"
    assert exc.tool_name == "read_text"
    assert "read_text" in str(exc)
