"""Tools for reconstructing PDF layout."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_error_content, create_text_content
from ..mcp.exceptions import ToolExecutionException
from ..models import FormulaModel, ProcessingMode, ProcessingRequest, TableEngine
from ..processors.pdf_processor import PDFProcessor
from ..utils.config import Config


class ReconstructLayoutTool(PDFTool):
    """High-level MCP tool for PDF layout reconstruction."""

    def __init__(self) -> None:
        super().__init__(
            name="reconstruct_layout",
            description="Rebuild PDF layout using extracted text, tables, and formulas",
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file",
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to process (1-indexed)",
                },
                "include_ocr": {
                    "type": "boolean",
                    "default": True,
                    "description": "Allow OCR for scanned documents",
                },
                "include_tables": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract tables for layout placement",
                },
                "include_formulas": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract formulas for layout placement",
                },
                "table_engine": {
                    "type": "string",
                    "enum": [engine.value for engine in TableEngine],
                    "default": TableEngine.CAMELOT.value,
                    "description": "Primary table extraction engine",
                },
                "formula_model": {
                    "type": "string",
                    "enum": [model.value for model in FormulaModel],
                    "default": FormulaModel.LATEX_OCR.value,
                    "description": "Formula recognition model",
                },
                "output": {
                    "type": "string",
                    "enum": ["summary", "json", "html", "all"],
                    "default": "all",
                    "description": "Output format preference",
                },
            },
            "required": ["file_path"],
        }

    @property
    def input_schema(self):  # type: ignore[override]
        from pydantic import BaseModel, Field
        from typing import List, Optional

        class LayoutInput(BaseModel):
            file_path: str = Field(description="Path to the PDF file")
            pages: Optional[List[int]] = Field(default=None, description="Pages to process (1-indexed)")
            include_ocr: bool = Field(default=True, description="Allow OCR when necessary")
            include_tables: bool = Field(default=True, description="Extract tables for placement")
            include_formulas: bool = Field(default=True, description="Extract formulas for placement")
            table_engine: str = Field(default=TableEngine.CAMELOT.value, description="Table extraction engine")
            formula_model: str = Field(default=FormulaModel.LATEX_OCR.value, description="Formula recognition model")
            output: str = Field(default="all", description="Preferred output format")

        return LayoutInput

    @property
    def output_schema(self):  # type: ignore[override]
        from pydantic import BaseModel, Field
        from typing import Any, Dict, Optional

        class LayoutOutput(BaseModel):
            success: bool = Field(description="Whether reconstruction succeeded")
            message: Optional[str] = Field(default=None, description="Status message")
            result: Optional[Dict[str, Any]] = Field(default=None, description="Layout reconstruction payload")

        return LayoutOutput

    async def execute(self, **kwargs) -> MCPToolResult:
        try:
            pdf_path = self.validate_file_path(kwargs["file_path"])
        except Exception as exc:  # pragma: no cover - validation path
            return MCPToolResult(
                content=[create_error_content(f"Invalid file path: {exc}")],
                isError=True,
            )

        pages = kwargs.get("pages")
        include_ocr = kwargs.get("include_ocr", True)
        include_tables = kwargs.get("include_tables", True)
        include_formulas = kwargs.get("include_formulas", True)
        table_engine_value = kwargs.get("table_engine", TableEngine.CAMELOT.value)
        formula_model_value = kwargs.get("formula_model", FormulaModel.LATEX_OCR.value)
        output_mode = kwargs.get("output", "all")

        try:
            table_engine = TableEngine(table_engine_value)
        except Exception as exc:  # pragma: no cover - enum validation
            raise ToolExecutionException(
                self.name,
                f"Unsupported table engine: {table_engine_value}",
                exc,
            ) from exc

        try:
            formula_model = FormulaModel(formula_model_value)
        except Exception as exc:  # pragma: no cover - enum validation
            raise ToolExecutionException(
                self.name,
                f"Unsupported formula model: {formula_model_value}",
                exc,
            ) from exc

        config = Config.load()
        processor = PDFProcessor(config)
        await processor.initialize()
        try:
            mode = ProcessingMode.FULL if (include_tables or include_formulas) else ProcessingMode.TEXT
            request = ProcessingRequest(
                file_path=str(pdf_path),
                mode=mode,
                pages=pages,
                include_ocr=include_ocr,
                include_formulas=include_formulas,
                table_engine=table_engine,
                formula_model=formula_model,
                reconstruct_layout=True,
            )
            result = await processor.process(request)
        except ToolExecutionException:
            raise
        except Exception as exc:
            self.logger.error("Layout reconstruction failed: %s", exc, exc_info=True)
            return MCPToolResult(
                content=[create_error_content(f"Layout reconstruction failed: {exc}")],
                isError=True,
            )
        finally:
            await processor.cleanup()

        layout = result.content.layout if result and result.content else None
        if not layout:
            return MCPToolResult(
                content=[create_error_content("Layout reconstruction returned no result")],
                isError=True,
            )

        content: List[Dict[str, Any]] = []
        summary = (
            f"Reconstructed {len(layout.pages)} pages in {layout.processing_time:.2f}s using {layout.method}."
        )
        content.append(create_text_content(summary))

        if output_mode in ("json", "all"):
            content.append(
                create_text_content(
                    json.dumps(layout.model_dump(), ensure_ascii=False, indent=2)
                )
            )

        if output_mode in ("html", "all"):
            combined_html = "<style>" + layout.css + "</style>" + "".join(page.html for page in layout.pages)
            content.append(create_text_content(combined_html))

        return MCPToolResult(content=content)

