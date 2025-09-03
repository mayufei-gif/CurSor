"""Text extraction processor using PyMuPDF and pdfplumber.

This module provides text extraction functionality with support for
multiple engines and output formats.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..models import (
    ProcessingRequest,
    TextExtractionResult,
    PageText,
    BoundingBox,
    OutputFormat
)
from ..utils.config import Config
from ..utils.exceptions import PDFProcessingError


class TextExtractor:
    """Extracts text from PDF documents using multiple engines."""
    
    def __init__(self, config: Config):
        """Initialize the text extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not fitz and not pdfplumber:
            raise PDFProcessingError("Either PyMuPDF or pdfplumber is required")
        
        # Preferred engine order
        self.engines = []
        if fitz:
            self.engines.append("pymupdf")
        if pdfplumber:
            self.engines.append("pdfplumber")
        
        self.logger.info(f"Text extractor initialized with engines: {self.engines}")
    
    async def initialize(self):
        """Initialize the extractor."""
        self.logger.info("Text extractor initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Text extractor cleanup complete")
    
    async def health_check(self) -> bool:
        """Check if the extractor is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return len(self.engines) > 0
    
    async def extract(self, file_path: Path, request: ProcessingRequest) -> TextExtractionResult:
        """Extract text from PDF.
        
        Args:
            file_path: Path to PDF file
            request: Processing request with options
            
        Returns:
            Text extraction result
            
        Raises:
            PDFProcessingError: If extraction fails
        """
        self.logger.info(f"Extracting text from: {file_path}")
        
        try:
            # Try engines in order of preference
            for engine in self.engines:
                try:
                    if engine == "pymupdf" and fitz:
                        return await self._extract_with_pymupdf(file_path, request)
                    elif engine == "pdfplumber" and pdfplumber:
                        return await self._extract_with_pdfplumber(file_path, request)
                except Exception as e:
                    self.logger.warning(f"Engine {engine} failed: {e}")
                    continue

            raise PDFProcessingError("All text extraction engines failed")
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise PDFProcessingError(f"Text extraction failed: {e}")
    
    async def _extract_with_pymupdf(self, file_path: Path, request: ProcessingRequest) -> TextExtractionResult:
        """Extract text using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            request: Processing request
            
        Returns:
            Text extraction result
        """
        doc = fitz.open(str(file_path))
        
        try:
            pages = []
            full_text = ""
            include_bbox = getattr(request, "include_bbox", False)
            
            # Determine pages to process
            page_range = request.pages if request.pages else range(len(doc))
            
            for page_num in page_range:
                if page_num >= len(doc):
                    continue
                
                page = doc[page_num]
                
                # Extract text with different methods based on requirements
                if include_bbox:
                    page_result = await self._extract_with_coordinates_pymupdf(page, page_num)
                else:
                    page_result = await self._extract_simple_pymupdf(page, page_num)
                
                pages.append(page_result)
                full_text += page_result.text + "\n\n"
            
            # Post-process text
            if getattr(request, "clean_text", True):
                full_text = self._clean_text(full_text)
                for p in pages:
                    p.text = self._clean_text(p.text)

            # Build result to match models.TextExtractionResult
            from time import perf_counter
            # Note: processing_time for this segment is not precisely tracked per engine here
            word_count = sum(len(p.text.split()) for p in pages)
            char_count = len(full_text)
            return TextExtractionResult(
                full_text=full_text.strip(),
                pages=pages,
                word_count=word_count,
                character_count=char_count,
                extraction_method="pymupdf",
                processing_time=0.0,
            )
            
        finally:
            doc.close()
    
    async def _extract_with_coordinates_pymupdf(self, page, page_num: int) -> PageText:
        """Extract text with coordinates using PyMuPDF.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            Page text with coordinates
        """
        # Get text with detailed information
        text_dict = page.get_text("dict")
        
        text_blocks = []
        full_text = ""
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                block_text = ""
                block_bbox = block["bbox"]
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    
                    if line_text.strip():
                        block_text += line_text + "\n"
                
                if block_text.strip():
                    text_blocks.append({
                        "text": block_text.strip(),
                        "bbox": BoundingBox(
                            x0=block_bbox[0],
                            y0=block_bbox[1],
                            x1=block_bbox[2],
                            y1=block_bbox[3]
                        ),
                        "font_info": self._extract_font_info(block)
                    })
                    full_text += block_text
        
        # Map to models.PageText fields
        return PageText(
            page=page_num + 1,
            text=full_text.strip(),
            bbox=None,
            confidence=None,
            language=None,
        )
    
    async def _extract_simple_pymupdf(self, page, page_num: int) -> PageText:
        """Extract simple text using PyMuPDF.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            Page text without coordinates
        """
        text = page.get_text()
        
        return PageText(
            page=page_num + 1,
            text=text,
            bbox=None,
            confidence=None,
            language=None,
        )
    
    async def _extract_with_pdfplumber(self, file_path: Path, request: ProcessingRequest) -> TextExtractionResult:
        """Extract text using pdfplumber.
        
        Args:
            file_path: Path to PDF file
            request: Processing request
            
        Returns:
            Text extraction result
        """
        with pdfplumber.open(str(file_path)) as pdf:
            pages = []
            full_text = ""
            include_bbox = getattr(request, "include_bbox", False)
            
            # Determine pages to process
            page_range = request.pages if request.pages else range(len(pdf.pages))
            
            for page_num in page_range:
                if page_num >= len(pdf.pages):
                    continue
                
                page = pdf.pages[page_num]
                
                # Extract text with different methods
                if include_bbox:
                    page_result = await self._extract_with_coordinates_pdfplumber(page, page_num)
                else:
                    page_result = await self._extract_simple_pdfplumber(page, page_num)
                
                pages.append(page_result)
                full_text += page_result.text + "\n\n"
            
            # Post-process text
            if getattr(request, "clean_text", True):
                full_text = self._clean_text(full_text)
                for p in pages:
                    p.text = self._clean_text(p.text)

            word_count = sum(len(p.text.split()) for p in pages)
            char_count = len(full_text)
            return TextExtractionResult(
                full_text=full_text.strip(),
                pages=pages,
                word_count=word_count,
                character_count=char_count,
                extraction_method="pdfplumber",
                processing_time=0.0,
            )
    
    async def _extract_with_coordinates_pdfplumber(self, page, page_num: int) -> PageText:
        """Extract text with coordinates using pdfplumber.
        
        Args:
            page: pdfplumber page object
            page_num: Page number
            
        Returns:
            Page text with coordinates
        """
        # Get characters with positions
        chars = page.chars
        
        # Group characters into words and lines
        text_blocks = []
        full_text = ""
        
        if chars:
            # Simple grouping by proximity
            current_line = []
            current_y = None
            
            for char in chars:
                if current_y is None or abs(char['y0'] - current_y) < 5:  # Same line
                    current_line.append(char)
                    current_y = char['y0']
                else:
                    # Process current line
                    if current_line:
                        line_text = ''.join([c['text'] for c in current_line])
                        if line_text.strip():
                            bbox = self._get_line_bbox(current_line)
                            text_blocks.append({
                                "text": line_text.strip(),
                                "bbox": bbox,
                                "font_info": self._get_font_info_pdfplumber(current_line)
                            })
                            full_text += line_text.strip() + "\n"
                    
                    # Start new line
                    current_line = [char]
                    current_y = char['y0']
            
            # Process last line
            if current_line:
                line_text = ''.join([c['text'] for c in current_line])
                if line_text.strip():
                    bbox = self._get_line_bbox(current_line)
                    text_blocks.append({
                        "text": line_text.strip(),
                        "bbox": bbox,
                        "font_info": self._get_font_info_pdfplumber(current_line)
                    })
                    full_text += line_text.strip() + "\n"
        
        # Fallback to simple extraction if character-level fails
        if not full_text.strip():
            full_text = page.extract_text() or ""
        
        return PageText(
            page=page_num + 1,
            text=full_text.strip(),
            bbox=None,
            confidence=None,
            language=None,
        )
    
    async def _extract_simple_pdfplumber(self, page, page_num: int) -> PageText:
        """Extract simple text using pdfplumber.
        
        Args:
            page: pdfplumber page object
            page_num: Page number
            
        Returns:
            Page text without coordinates
        """
        text = page.extract_text() or ""
        
        return PageText(
            page=page_num + 1,
            text=text,
            bbox=None,
            confidence=None,
            language=None,
        )
    
    def _extract_font_info(self, block: Dict) -> Dict[str, Any]:
        """Extract font information from PyMuPDF block.
        
        Args:
            block: PyMuPDF text block
            
        Returns:
            Font information dictionary
        """
        font_info = {
            "fonts": [],
            "sizes": [],
            "flags": []
        }
        
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_info["fonts"].append(span.get("font", ""))
                font_info["sizes"].append(span.get("size", 0))
                font_info["flags"].append(span.get("flags", 0))
        
        return font_info
    
    def _get_line_bbox(self, chars: List[Dict]) -> BoundingBox:
        """Get bounding box for a line of characters.
        
        Args:
            chars: List of character dictionaries
            
        Returns:
            Bounding box
        """
        if not chars:
            return BoundingBox(x0=0, y0=0, x1=0, y1=0)
        
        x0 = min(c['x0'] for c in chars)
        y0 = min(c['y0'] for c in chars)
        x1 = max(c['x1'] for c in chars)
        y1 = max(c['y1'] for c in chars)
        
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    
    def _get_font_info_pdfplumber(self, chars: List[Dict]) -> Dict[str, Any]:
        """Get font information from pdfplumber characters.
        
        Args:
            chars: List of character dictionaries
            
        Returns:
            Font information dictionary
        """
        fonts = [c.get('fontname', '') for c in chars]
        sizes = [c.get('size', 0) for c in chars]
        
        return {
            "fonts": list(set(fonts)),
            "sizes": list(set(sizes)),
            "flags": []
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common OCR errors (basic)
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufeff', '')  # Remove BOM
        
        # Normalize quotes
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        
        return text.strip()
