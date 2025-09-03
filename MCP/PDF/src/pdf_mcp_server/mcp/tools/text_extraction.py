#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Extraction Tools

This module provides tools for extracting text from PDF files using various methods.
"""

import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

from .base import PDFTool, PDFToolOutput

class ExtractTextInput(BaseModel):
    """Input schema for text extraction."""
    model_config = ConfigDict(extra="forbid")
    
    file_path: str = Field(
        description="Path to the PDF file",
        examples=["/path/to/document.pdf"]
    )
    method: str = Field(
        default="pymupdf",
        description="Extraction method: 'pymupdf' or 'pdfplumber'",
        examples=["pymupdf", "pdfplumber"]
    )
    pages: Optional[str] = Field(
        default=None,
        description="Page range (e.g., '1-5', '1,3,5', 'all')",
        examples=["1-5", "1,3,5", "all"]
    )
    output_format: str = Field(
        default="text",
        description="Output format: 'text', 'json', 'markdown'",
        examples=["text", "json", "markdown"]
    )

class ExtractTextOutput(PDFToolOutput):
    """Output schema for text extraction."""
    
    text: str = Field(
        description="Extracted text content"
    )
    page_count: int = Field(
        description="Total number of pages in the PDF"
    )
    pages_processed: List[int] = Field(
        description="List of page numbers that were processed"
    )
    method_used: str = Field(
        description="Extraction method that was used"
    )
    word_count: Optional[int] = Field(
        default=None,
        description="Number of words extracted"
    )
    character_count: Optional[int] = Field(
        default=None,
        description="Number of characters extracted"
    )

class ExtractTextAdvancedInput(BaseModel):
    """Input schema for advanced text extraction."""
    model_config = ConfigDict(extra="forbid")
    
    file_path: str = Field(
        description="Path to the PDF file",
        examples=["/path/to/document.pdf"]
    )
    method: str = Field(
        default="pymupdf",
        description="Extraction method: 'pymupdf' or 'pdfplumber'",
        examples=["pymupdf", "pdfplumber"]
    )
    pages: Optional[str] = Field(
        default=None,
        description="Page range (e.g., '1-5', '1,3,5', 'all')",
        examples=["1-5", "1,3,5", "all"]
    )
    output_format: str = Field(
        default="json",
        description="Output format: 'text', 'json', 'markdown', 'html'",
        examples=["text", "json", "markdown", "html"]
    )
    preserve_layout: bool = Field(
        default=False,
        description="Whether to preserve text layout and formatting"
    )
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract document metadata"
    )
    extract_annotations: bool = Field(
        default=False,
        description="Whether to extract annotations and comments"
    )
    extract_links: bool = Field(
        default=False,
        description="Whether to extract hyperlinks"
    )
    min_word_length: int = Field(
        default=1,
        description="Minimum word length to include",
        ge=1
    )
    remove_headers_footers: bool = Field(
        default=False,
        description="Whether to attempt to remove headers and footers"
    )

class ExtractTextAdvancedOutput(PDFToolOutput):
    """Output schema for advanced text extraction."""
    
    text: str = Field(
        description="Extracted text content"
    )
    page_count: int = Field(
        description="Total number of pages in the PDF"
    )
    pages_processed: List[int] = Field(
        description="List of page numbers that were processed"
    )
    method_used: str = Field(
        description="Extraction method that was used"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Document metadata"
    )
    annotations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Extracted annotations"
    )
    links: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Extracted hyperlinks"
    )
    statistics: Dict[str, Any] = Field(
        description="Text extraction statistics"
    )
    pages_data: List[Dict[str, Any]] = Field(
        description="Per-page extraction data"
    )

class ExtractTextTool(PDFTool):
    """Basic text extraction tool."""
    
    def __init__(self):
        super().__init__(
            name="extract_text",
            description="Extract text from PDF files using PyMuPDF or pdfplumber"
        )
    
    @property
    def input_schema(self) -> type[BaseModel]:
        return ExtractTextInput
    
    @property
    def output_schema(self) -> type[BaseModel]:
        return ExtractTextOutput
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text extraction."""
        start_time = time.time()
        
        # Validate input
        validated_input = self.validate_input(input_data)
        file_path = self.validate_file_path(validated_input.file_path)
        
        try:
            # Import extraction libraries
            if validated_input.method == "pymupdf":
                import fitz  # PyMuPDF
                text, page_data = await self._extract_with_pymupdf(
                    file_path, validated_input.pages
                )
            elif validated_input.method == "pdfplumber":
                import pdfplumber
                text, page_data = await self._extract_with_pdfplumber(
                    file_path, validated_input.pages
                )
            else:
                raise ValueError(f"Unsupported method: {validated_input.method}")
            
            # Format output
            if validated_input.output_format == "json":
                formatted_text = {
                    "pages": page_data,
                    "full_text": text
                }
                text = str(formatted_text)
            elif validated_input.output_format == "markdown":
                text = self._format_as_markdown(text, page_data)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "message": "Text extraction completed successfully",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "text": text,
                "page_count": len(page_data),
                "pages_processed": list(page_data.keys()),
                "method_used": validated_input.method,
                "word_count": len(text.split()),
                "character_count": len(text)
            }
            
            return self.validate_output(result).model_dump()
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {str(e)}")
            processing_time = time.time() - start_time
            
            result = {
                "success": False,
                "message": f"Text extraction failed: {str(e)}",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "text": "",
                "page_count": 0,
                "pages_processed": [],
                "method_used": validated_input.method,
                "word_count": 0,
                "character_count": 0
            }
            
            return self.validate_output(result).model_dump()
    
    async def _extract_with_pymupdf(self, file_path: Path, pages: Optional[str]) -> tuple[str, Dict[int, str]]:
        """Extract text using PyMuPDF."""
        import fitz
        
        doc = fitz.open(str(file_path))
        page_data = {}
        full_text = []
        
        page_numbers = self._parse_page_range(pages, len(doc))
        
        for page_num in page_numbers:
            page = doc[page_num - 1]  # fitz uses 0-based indexing
            text = page.get_text()
            page_data[page_num] = text
            full_text.append(text)
        
        doc.close()
        return "\n\n".join(full_text), page_data
    
    async def _extract_with_pdfplumber(self, file_path: Path, pages: Optional[str]) -> tuple[str, Dict[int, str]]:
        """Extract text using pdfplumber."""
        import pdfplumber
        
        page_data = {}
        full_text = []
        
        with pdfplumber.open(str(file_path)) as pdf:
            page_numbers = self._parse_page_range(pages, len(pdf.pages))
            
            for page_num in page_numbers:
                page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing
                text = page.extract_text() or ""
                page_data[page_num] = text
                full_text.append(text)
        
        return "\n\n".join(full_text), page_data
    
    def _parse_page_range(self, pages: Optional[str], total_pages: int) -> List[int]:
        """Parse page range specification."""
        if not pages or pages.lower() == "all":
            return list(range(1, total_pages + 1))
        
        page_numbers = []
        
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                page_numbers.extend(range(start, end + 1))
            else:
                page_numbers.append(int(part))
        
        # Filter valid page numbers
        return [p for p in page_numbers if 1 <= p <= total_pages]
    
    def _format_as_markdown(self, text: str, page_data: Dict[int, str]) -> str:
        """Format text as markdown."""
        markdown_parts = []
        
        for page_num, page_text in page_data.items():
            markdown_parts.append(f"## Page {page_num}\n\n{page_text}\n")
        
        return "\n".join(markdown_parts)

class ExtractTextAdvancedTool(PDFTool):
    """Advanced text extraction tool with additional features."""
    
    def __init__(self):
        super().__init__(
            name="extract_text_advanced",
            description="Advanced text extraction with metadata, annotations, and formatting options"
        )
    
    @property
    def input_schema(self) -> type[BaseModel]:
        return ExtractTextAdvancedInput
    
    @property
    def output_schema(self) -> type[BaseModel]:
        return ExtractTextAdvancedOutput
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced text extraction."""
        start_time = time.time()
        
        # Validate input
        validated_input = self.validate_input(input_data)
        file_path = self.validate_file_path(validated_input.file_path)
        
        try:
            # Extract text and additional data
            if validated_input.method == "pymupdf":
                result_data = await self._extract_advanced_pymupdf(file_path, validated_input)
            elif validated_input.method == "pdfplumber":
                result_data = await self._extract_advanced_pdfplumber(file_path, validated_input)
            else:
                raise ValueError(f"Unsupported method: {validated_input.method}")
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "message": "Advanced text extraction completed successfully",
                "file_path": str(file_path),
                "processing_time": processing_time,
                **result_data
            }
            
            return self.validate_output(result).model_dump()
            
        except Exception as e:
            self.logger.error(f"Advanced text extraction failed: {str(e)}")
            processing_time = time.time() - start_time
            
            result = {
                "success": False,
                "message": f"Advanced text extraction failed: {str(e)}",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "text": "",
                "page_count": 0,
                "pages_processed": [],
                "method_used": validated_input.method,
                "metadata": None,
                "annotations": None,
                "links": None,
                "statistics": {},
                "pages_data": []
            }
            
            return self.validate_output(result).model_dump()
    
    async def _extract_advanced_pymupdf(self, file_path: Path, config: ExtractTextAdvancedInput) -> Dict[str, Any]:
        """Advanced extraction using PyMuPDF."""
        import fitz
        
        doc = fitz.open(str(file_path))
        
        # Extract metadata
        metadata = doc.metadata if config.extract_metadata else None
        
        # Process pages
        page_numbers = self._parse_page_range(config.pages, len(doc))
        pages_data = []
        full_text_parts = []
        all_annotations = []
        all_links = []
        
        for page_num in page_numbers:
            page = doc[page_num - 1]
            
            # Extract text
            if config.preserve_layout:
                text = page.get_text("dict")
                page_text = self._extract_text_with_layout(text)
            else:
                page_text = page.get_text()
            
            # Filter by word length
            if config.min_word_length > 1:
                words = page_text.split()
                words = [w for w in words if len(w) >= config.min_word_length]
                page_text = " ".join(words)
            
            # Extract annotations
            page_annotations = []
            if config.extract_annotations:
                for annot in page.annots():
                    page_annotations.append({
                        "type": annot.type[1],
                        "content": annot.content,
                        "rect": list(annot.rect)
                    })
            
            # Extract links
            page_links = []
            if config.extract_links:
                for link in page.get_links():
                    page_links.append({
                        "uri": link.get("uri", ""),
                        "rect": link.get("from", [])
                    })
            
            page_data = {
                "page_number": page_num,
                "text": page_text,
                "word_count": len(page_text.split()),
                "character_count": len(page_text),
                "annotations": page_annotations,
                "links": page_links
            }
            
            pages_data.append(page_data)
            full_text_parts.append(page_text)
            all_annotations.extend(page_annotations)
            all_links.extend(page_links)
        
        doc.close()
        
        full_text = "\n\n".join(full_text_parts)
        
        # Calculate statistics
        statistics = {
            "total_words": len(full_text.split()),
            "total_characters": len(full_text),
            "average_words_per_page": len(full_text.split()) / len(pages_data) if pages_data else 0,
            "annotation_count": len(all_annotations),
            "link_count": len(all_links)
        }
        
        return {
            "text": full_text,
            "page_count": len(doc),
            "pages_processed": page_numbers,
            "method_used": config.method,
            "metadata": metadata,
            "annotations": all_annotations if config.extract_annotations else None,
            "links": all_links if config.extract_links else None,
            "statistics": statistics,
            "pages_data": pages_data
        }
    
    async def _extract_advanced_pdfplumber(self, file_path: Path, config: ExtractTextAdvancedInput) -> Dict[str, Any]:
        """Advanced extraction using pdfplumber."""
        import pdfplumber
        
        pages_data = []
        full_text_parts = []
        
        with pdfplumber.open(str(file_path)) as pdf:
            # Extract metadata
            metadata = pdf.metadata if config.extract_metadata else None
            
            page_numbers = self._parse_page_range(config.pages, len(pdf.pages))
            
            for page_num in page_numbers:
                page = pdf.pages[page_num - 1]
                
                # Extract text
                if config.preserve_layout:
                    page_text = page.extract_text(layout=True) or ""
                else:
                    page_text = page.extract_text() or ""
                
                # Filter by word length
                if config.min_word_length > 1:
                    words = page_text.split()
                    words = [w for w in words if len(w) >= config.min_word_length]
                    page_text = " ".join(words)
                
                page_data = {
                    "page_number": page_num,
                    "text": page_text,
                    "word_count": len(page_text.split()),
                    "character_count": len(page_text),
                    "annotations": [],  # pdfplumber doesn't extract annotations easily
                    "links": []  # pdfplumber doesn't extract links easily
                }
                
                pages_data.append(page_data)
                full_text_parts.append(page_text)
        
        full_text = "\n\n".join(full_text_parts)
        
        # Calculate statistics
        statistics = {
            "total_words": len(full_text.split()),
            "total_characters": len(full_text),
            "average_words_per_page": len(full_text.split()) / len(pages_data) if pages_data else 0,
            "annotation_count": 0,
            "link_count": 0
        }
        
        return {
            "text": full_text,
            "page_count": len(pdf.pages),
            "pages_processed": page_numbers,
            "method_used": config.method,
            "metadata": metadata,
            "annotations": None,
            "links": None,
            "statistics": statistics,
            "pages_data": pages_data
        }
    
    def _parse_page_range(self, pages: Optional[str], total_pages: int) -> List[int]:
        """Parse page range specification."""
        if not pages or pages.lower() == "all":
            return list(range(1, total_pages + 1))
        
        page_numbers = []
        
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                page_numbers.extend(range(start, end + 1))
            else:
                page_numbers.append(int(part))
        
        # Filter valid page numbers
        return [p for p in page_numbers if 1 <= p <= total_pages]
    
    def _extract_text_with_layout(self, text_dict: Dict) -> str:
        """Extract text while preserving layout from PyMuPDF text dict."""
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    if line_text.strip():
                        text_parts.append(line_text)
        
        return "\n".join(text_parts)