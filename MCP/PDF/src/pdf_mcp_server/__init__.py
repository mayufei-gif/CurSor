"""PDF-MCP Server - A comprehensive PDF processing MCP server.

This package provides a unified interface for PDF processing including:
- Text extraction using PyMuPDF and pdfplumber
- Table extraction using Camelot and Tabula
- OCR processing using OCRmyPDF and Tesseract
- Formula recognition using LaTeX-OCR and pix2tex
- Scientific document parsing using GROBID
"""

# Re-export metadata constants from a dedicated module so that other modules
# (e.g. :mod:`pdf_mcp_server.models`) can reference the package version without
# importing this package and triggering circular dependencies during import
# time.  Keeping metadata in a lightweight module allows both sides to read the
# values safely.
from ._metadata import __version__, __author__, __email__  # noqa: F401

# Note: avoid importing heavy modules at package import time to prevent side effects
# and circular import issues. Import consumers should import submodules directly.
from .models import (
    ProcessingRequest,
    ProcessingResponse,
    TextExtractionResult,
    TableExtractionResult,
    FormulaExtractionResult,
    LayoutReconstructionResult,
    PDFInfo,
)

__all__ = [
    "ProcessingRequest",
    "ProcessingResponse",
    "TextExtractionResult",
    "TableExtractionResult",
    "FormulaExtractionResult",
    "LayoutReconstructionResult",
    "PDFInfo",
]
