"""Document analyzer for PDF type detection and basic information extraction.

This module provides functionality to analyze PDF documents and determine
their characteristics such as whether they are scanned or text-based.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from datetime import datetime
from ..models import PDFInfo, BoundingBox
from ..utils.config import Config
from ..utils.exceptions import PDFProcessingError


class DocumentAnalyzer:
    """Analyzes PDF documents to determine their characteristics."""
    
    def __init__(self, config: Config):
        """Initialize the document analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not fitz:
            self.logger.warning("PyMuPDF not available, some features may be limited")
        if not pdfplumber:
            self.logger.warning("pdfplumber not available, some features may be limited")
    
    async def initialize(self):
        """Initialize the analyzer."""
        self.logger.info("Document analyzer initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Document analyzer cleanup complete")
    
    async def health_check(self) -> bool:
        """Check if the analyzer is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return fitz is not None or pdfplumber is not None
    
    async def analyze(self, file_path: Path) -> PDFInfo:
        """Analyze a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDF information object
            
        Raises:
            PDFProcessingError: If analysis fails
        """
        self.logger.info(f"Analyzing document: {file_path}")
        
        try:
            # Basic file information
            file_size = file_path.stat().st_size
            
            # Use PyMuPDF for primary analysis
            if fitz:
                info = await self._analyze_with_pymupdf(file_path, file_size)
            elif pdfplumber:
                info = await self._analyze_with_pdfplumber(file_path, file_size)
            else:
                raise PDFProcessingError("No PDF processing library available")
            
            self.logger.info(
                f"Document analysis complete: {info.pages} pages, "
                f"text_layer={info.has_text_layer}, scanned={info.is_scanned}"
            )
            
            return info
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            raise PDFProcessingError(f"Analysis failed: {e}")
    
    async def _analyze_with_pymupdf(self, file_path: Path, file_size: int) -> PDFInfo:
        """Analyze document using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            file_size: File size in bytes
            
        Returns:
            PDF information
        """
        doc = fitz.open(str(file_path))
        
        try:
            # Basic information
            page_count = len(doc)
            metadata = doc.metadata
            
            # Check for text layer and determine if scanned
            has_text_layer = False
            is_scanned = True
            total_text_length = 0
            pages_with_text = 0
            
            # Sample pages for analysis (max 10 pages)
            sample_pages = min(10, page_count)
            page_indices = [i * page_count // sample_pages for i in range(sample_pages)]
            
            for page_num in page_indices:
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    has_text_layer = True
                    pages_with_text += 1
                    total_text_length += len(text)
            
            # Determine if document is scanned
            if has_text_layer:
                # If we have text but very little, it might be a scanned document with OCR
                avg_text_per_page = total_text_length / max(pages_with_text, 1)
                text_coverage = pages_with_text / sample_pages
                
                # Heuristics for scanned document detection
                if avg_text_per_page < 100 or text_coverage < 0.5:
                    is_scanned = True
                else:
                    is_scanned = False
            
            # Parse PDF date strings safely
            def _parse_pdf_date(s: Optional[str]) -> Optional[datetime]:
                if not s:
                    return None
                # Common PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
                try:
                    if s.startswith('D:'):
                        s = s[2:]
                    # Strip timezone if present
                    main = s.split("+", 1)[0].split("-", 1)[0]
                    # Pad if shorter
                    fmt = "%Y%m%d%H%M%S"
                    main = (main + "000000000000")[:14]
                    return datetime.strptime(main, fmt)
                except Exception:
                    return None

            return PDFInfo(
                filename=file_path.name,
                pages=page_count,
                file_size=file_size,
                is_scanned=is_scanned,
                has_text_layer=has_text_layer,
                title=metadata.get("title") or None,
                author=metadata.get("author") or None,
                subject=metadata.get("subject") or None,
                creation_date=_parse_pdf_date(metadata.get("creationDate")),
                modification_date=_parse_pdf_date(metadata.get("modDate")),
            )
            
        finally:
            doc.close()
    
    async def _analyze_with_pdfplumber(self, file_path: Path, file_size: int) -> PDFInfo:
        """Analyze document using pdfplumber.
        
        Args:
            file_path: Path to PDF file
            file_size: File size in bytes
            
        Returns:
            PDF information
        """
        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
            
            # Check for text layer
            has_text_layer = False
            is_scanned = True
            total_text_length = 0
            pages_with_text = 0
            
            # Sample pages for analysis
            sample_pages = min(10, page_count)
            page_indices = [i * page_count // sample_pages for i in range(sample_pages)]
            
            for page_num in page_indices:
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if text and text.strip():
                    has_text_layer = True
                    pages_with_text += 1
                    total_text_length += len(text)
            
            # Determine if scanned
            if has_text_layer:
                avg_text_per_page = total_text_length / max(pages_with_text, 1)
                text_coverage = pages_with_text / sample_pages
                
                if avg_text_per_page < 100 or text_coverage < 0.5:
                    is_scanned = True
                else:
                    is_scanned = False
            
            # Extract page dimensions
            page_dimensions = []
            for i in range(min(5, page_count)):
                page = pdf.pages[i]
                page_dimensions.append({
                    "width": float(page.width),
                    "height": float(page.height)
                })
            
            # Basic metadata (pdfplumber has limited metadata access)
            metadata = pdf.metadata or {}
            
            def _parse_pdf_date(s: Optional[str]) -> Optional[datetime]:
                if not s:
                    return None
                try:
                    if s.startswith('D:'):
                        s = s[2:]
                    main = s.split("+", 1)[0].split("-", 1)[0]
                    fmt = "%Y%m%d%H%M%S"
                    main = (main + "000000000000")[:14]
                    return datetime.strptime(main, fmt)
                except Exception:
                    return None

            return PDFInfo(
                filename=file_path.name,
                pages=page_count,
                file_size=file_size,
                is_scanned=is_scanned,
                has_text_layer=has_text_layer,
                title=metadata.get("Title") or None,
                author=metadata.get("Author") or None,
                subject=metadata.get("Subject") or None,
                creation_date=_parse_pdf_date(metadata.get("CreationDate")),
                modification_date=_parse_pdf_date(metadata.get("ModDate")),
            )
    
    async def detect_document_type(self, file_path: Path) -> str:
        """Detect the type of document (research paper, report, etc.).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document type string
        """
        try:
            info = await self.analyze(file_path)
            
            # Simple heuristics for document type detection
            if info.title:
                title_lower = info.title.lower()
                if any(keyword in title_lower for keyword in ["research", "paper", "study", "analysis"]):
                    return "research_paper"
                elif any(keyword in title_lower for keyword in ["report", "annual", "quarterly"]):
                    return "report"
                elif any(keyword in title_lower for keyword in ["manual", "guide", "documentation"]):
                    return "manual"
            
            # Fallback based on page count
            if info.page_count <= 5:
                return "short_document"
            elif info.page_count <= 20:
                return "medium_document"
            else:
                return "long_document"
                
        except Exception as e:
            self.logger.warning(f"Document type detection failed: {e}")
            return "unknown"
    
    async def extract_layout_info(self, file_path: Path, page_num: int = 0) -> Dict[str, Any]:
        """Extract layout information from a specific page.
        
        Args:
            file_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            Layout information dictionary
        """
        if not fitz:
            raise PDFProcessingError("PyMuPDF required for layout analysis")
        
        doc = fitz.open(str(file_path))
        
        try:
            if page_num >= len(doc):
                raise PDFProcessingError(f"Page {page_num} not found in document")
            
            page = doc[page_num]
            
            # Get text blocks with positions
            blocks = page.get_text("dict")
            
            # Extract images
            images = page.get_images()
            
            # Extract drawings/graphics
            drawings = page.get_drawings()
            
            layout_info = {
                "page_number": page_num,
                "page_size": {
                    "width": page.rect.width,
                    "height": page.rect.height
                },
                "text_blocks": len(blocks.get("blocks", [])),
                "images": len(images),
                "drawings": len(drawings),
                "has_tables": False,  # Will be determined by table extractor
                "has_formulas": False  # Will be determined by formula extractor
            }
            
            return layout_info
            
        finally:
            doc.close()
