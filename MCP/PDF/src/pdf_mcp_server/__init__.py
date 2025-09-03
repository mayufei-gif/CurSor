"""PDF-MCP Server - A comprehensive PDF processing MCP server.

This package provides a unified interface for PDF processing including:
- Text extraction using PyMuPDF and pdfplumber
- Table extraction using Camelot and Tabula
- OCR processing using OCRmyPDF and Tesseract
- Formula recognition using LaTeX-OCR and pix2tex
- Scientific document parsing using GROBID
"""

__version__ = "0.1.0"
__author__ = "PDF-MCP Team"
__email__ = "team@pdf-mcp.com"

# Note: avoid importing heavy modules at package import time to prevent side effects
# and circular import issues. Import consumers should import submodules directly.
from .models import (
    ProcessingRequest,
    ProcessingResponse,
    TextExtractionResult,
    TableExtractionResult,
    FormulaExtractionResult,
    PDFInfo,
)

__all__ = [
    "ProcessingRequest",
    "ProcessingResponse",
    "TextExtractionResult",
    "TableExtractionResult",
    "FormulaExtractionResult",
    "PDFInfo",
]
