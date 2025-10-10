"""Integration-oriented tests for the PDF MCP server building blocks."""

from __future__ import annotations

# The tests intentionally mirror production request payloads and setup logic to
# ensure end-to-end coverage, which results in large shared blocks of code with
# the implementation modules.  Disable Pylint's duplicate-code warning so the
# lint step focuses on actionable issues.
# pylint: disable=duplicate-code

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pdf_mcp_server import __version__  # noqa: E402  pylint: disable=wrong-import-position
from pdf_mcp_server.main import (  # noqa: E402  pylint: disable=wrong-import-position
    Config,
    lifespan,
)
from pdf_mcp_server.mcp.protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    MCPCapabilities,
    MCPClientInfo,
    MCPInitializeParams,
    MCPMethod,
    MCPProtocolHandler,
)
from pdf_mcp_server.models import (  # noqa: E402  pylint: disable=wrong-import-position
    ProcessingMode,
    ProcessingRequest,
)


def test_metadata_exposed() -> None:
    """The package should expose a semantic version for clients."""
    major, minor, patch = __version__.split(".")
    assert major.isdigit()
    assert minor.isdigit()
    assert patch.isdigit()


def test_protocol_round_trip() -> None:
    """Requests created by the protocol handler should survive serialization."""
    handler = MCPProtocolHandler()
    params = MCPInitializeParams(
        protocolVersion="2024-11-05",
        capabilities=MCPCapabilities(tools={"streaming": True}),
        clientInfo=MCPClientInfo(name="pytest", version="1.0.0"),
    )
    request = handler.create_request(method=MCPMethod.INITIALIZE, params=params.model_dump())

    parsed = handler.parse_message(request.model_dump())

    assert parsed.id == request.id
    assert parsed.method == request.method
    assert parsed.params == request.params


def test_lifespan_initializes_processor(monkeypatch, tmp_path) -> None:
    """The FastAPI lifespan hook should initialize and clean up the processor."""

    created = SimpleNamespace(instance=None, initialized=False, cleaned=False)

    async def exercise() -> None:
        def fake_config() -> Config:
            config = Config()
            config.temp_dir = str(tmp_path)
            config.log_file = None
            return config

        class DummyProcessor:
            def __init__(self, config: Config) -> None:
                created.instance = self
                self.config = config

            async def initialize(self) -> None:
                created.initialized = True

            async def cleanup(self) -> None:
                created.cleaned = True

        monkeypatch.setattr("pdf_mcp_server.main.Config", fake_config)
        monkeypatch.setattr("pdf_mcp_server.main.PDFProcessor", DummyProcessor)
        monkeypatch.setattr("pdf_mcp_server.main.setup_logging", lambda *_, **__: None)
        monkeypatch.setattr("pdf_mcp_server.main.pdf_processor", None)

        app = FastAPI()

        try:
            async with lifespan(app):
                assert created.instance is not None
                assert created.initialized is True
        finally:
            # Ensure the module-level singleton is reset for other tests.
            monkeypatch.setattr("pdf_mcp_server.main.pdf_processor", None)

    asyncio.run(exercise())

    assert created.cleaned is True


def test_processing_request_builds_from_primitives() -> None:
    """The request model should normalise primitive payloads into enums."""
    request = ProcessingRequest(
        file_path="sample.pdf",
        mode="read_text",
        table_output_format="json",
        include_ocr=True,
    )
codex/review-pull-request-logs-lkxc54

    assert request.mode is ProcessingMode.TEXT
    assert request.table_output_format.value == "json"
    assert request.include_ocr is True

    
    # Serialize the request
    serialized_request = protocol.serialize_message(request)
    
    # Process the request (in a real scenario, this would be sent to the server)
    # For testing, we'll directly call the tool handler
    tool_name = request.params["name"]
    tool_args = request.params["arguments"]
    
    # Get the tool
    tool = next((t for t in server.server.tools if t.name == tool_name), None)
    assert tool is not None, f"Tool '{tool_name}' not found"
    
    # Call the tool
    result = await tool.execute(tool_args)
    
    # Verify the result
    assert result is not None, "Tool execution returned None"
    assert "text" in result or "content" in result, "Text extraction result missing text content"
    
    logger.info("Text extraction test passed")


async def run_tests():
    """Run all tests."""
    logger.info("Starting PDF-MCP server tests")
    
    try:
        await test_server_initialization()
        await test_tool_registration()
        await test_protocol_handler()
        await test_text_extraction()
        
        logger.info("All tests passed successfully")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
main
