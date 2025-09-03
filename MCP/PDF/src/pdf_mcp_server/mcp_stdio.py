#!/usr/bin/env python3
"""
MCP stdio server for PDF-MCP.

Exposes a minimal MCP tool `pdf_to_docx` that:
- takes a `file_path` (PDF) and optional `out_dir` or `out_path`
- runs the full pipeline and exports a .docx next to the PDF if no out is given
- returns the output path and basic metadata

Usage (Windows example):
  .venv\Scripts\python.exe -m pdf_mcp_server.mcp_stdio

Then configure your MCP client to launch the above command.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .models import ProcessingRequest, ProcessingMode
from .processors.pdf_processor import PDFProcessor
from .utils.config import Config
from .utils.docx_exporter import build_docx_from_pipeline


server = Server("pdf-mcp-docx")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="pdf_to_docx",
            description="Process a PDF and export a DOCX next to it",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to PDF file"},
                    "out_dir": {"type": "string", "description": "Optional output directory"},
                    "out_path": {"type": "string", "description": "Optional explicit output .docx path"},
                    "include_ocr": {"type": "boolean", "description": "Apply OCR if needed"},
                    "include_formulas": {"type": "boolean", "description": "Extract formulas"}
                },
                "required": ["file_path"]
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    if name != "pdf_to_docx":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    file_path = Path(arguments.get("file_path", "")).expanduser()
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        return [TextContent(type="text", text=f"Invalid PDF path: {file_path}")]

    out_path_arg = arguments.get("out_path")
    out_dir_arg = arguments.get("out_dir")
    if out_path_arg:
        out_path = Path(out_path_arg)
    else:
        out_dir = Path(out_dir_arg) if out_dir_arg else file_path.parent
        out_path = out_dir / (file_path.stem + ".docx")

    include_ocr = bool(arguments.get("include_ocr", True))
    include_formulas = bool(arguments.get("include_formulas", False))

    config = Config()
    processor = PDFProcessor(config)
    await processor.initialize()
    try:
        request = ProcessingRequest(
            file_path=str(file_path),
            mode=ProcessingMode.FULL,
            include_ocr=include_ocr,
            include_formulas=include_formulas,
            include_grobid=False,
        )
        result = await processor.process(request)
    finally:
        await processor.cleanup()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_docx_from_pipeline(result, out_path)

    return [TextContent(type="text", text=str(out_path))]


async def _run_async():
    # Start stdio transport and hand streams to the MCP Server
    async with stdio_server() as (read, write):
        init = server.create_initialization_options()
        await server.run(read, write, init)


def main():
    asyncio.run(_run_async())


if __name__ == "__main__":
    main()
