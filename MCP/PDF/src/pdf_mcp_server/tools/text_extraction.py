#!/usr/bin/env python3
"""
PDF Text Extraction Tools

Implements text extraction functionality for PDF files using multiple libraries
including PyMuPDF, pdfplumber, and PyPDF2 for comprehensive text extraction.

Author: PDF-MCP Team
License: MIT
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class ReadTextTool(PDFTool):
    """Extract text content from PDF files with various extraction methods."""
    
    def __init__(self):
        super().__init__(
            name="read_text",
            description="Extract text content from PDF files using multiple extraction methods"
        )
    
    @property
    def input_schema(self) -> type[BaseModel]:
        """Return the input schema for this tool."""
        from pydantic import BaseModel, Field
        from typing import List, Optional
        
        class ReadTextInput(BaseModel):
            file_path: str = Field(description="Path to the PDF file")
            method: str = Field(default="auto", description="Text extraction method to use")
            pages: Optional[List[int]] = Field(default=None, description="Specific pages to extract (1-indexed)")
            include_metadata: bool = Field(default=False, description="Include page metadata in the output")
            preserve_layout: bool = Field(default=False, description="Attempt to preserve text layout and formatting")
            extract_annotations: bool = Field(default=False, description="Extract annotations and comments")
        
        return ReadTextInput
    
    @property
    def output_schema(self) -> type[BaseModel]:
        """Return the output schema for this tool."""
        from pydantic import BaseModel, Field
        from typing import List, Optional, Dict, Any
        
        class ReadTextOutput(BaseModel):
            success: bool = Field(description="Whether the operation was successful")
            content: List[Dict[str, Any]] = Field(description="Extracted text content")
            metadata: Optional[Dict[str, Any]] = Field(default=None, description="File metadata")
            error: Optional[str] = Field(default=None, description="Error message if operation failed")
        
        return ReadTextOutput
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "method": {
                    "type": "string",
                    "enum": ["auto", "pymupdf", "pdfplumber", "pypdf2"],
                    "default": "auto",
                    "description": "Text extraction method to use"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to extract (1-indexed). If not provided, extracts all pages"
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include page metadata in the output"
                },
                "preserve_layout": {
                    "type": "boolean",
                    "default": False,
                    "description": "Attempt to preserve text layout and formatting"
                },
                "extract_annotations": {
                    "type": "boolean",
                    "default": False,
                    "description": "Extract annotations and comments"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        method = kwargs.get("method", "auto")
        pages = kwargs.get("pages")
        include_metadata = kwargs.get("include_metadata", False)
        preserve_layout = kwargs.get("preserve_layout", False)
        extract_annotations = kwargs.get("extract_annotations", False)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Choose extraction method
            if method == "auto":
                method = self._choose_best_method()
            
            # Extract text based on method
            if method == "pymupdf":
                result = await self._extract_with_pymupdf(
                    pdf_path, pages, include_metadata, preserve_layout, extract_annotations
                )
            elif method == "pdfplumber":
                result = await self._extract_with_pdfplumber(
                    pdf_path, pages, include_metadata, preserve_layout
                )
            elif method == "pypdf2":
                result = await self._extract_with_pypdf2(
                    pdf_path, pages, include_metadata
                )
            else:
                raise ToolExecutionException(f"Unknown extraction method: {method}")
            
            # Format result
            content = []
            
            # Add summary
            summary = {
                "file": str(pdf_path),
                "method": method,
                "total_pages": result["total_pages"],
                "extracted_pages": len(result["pages"]),
                "total_characters": sum(len(page["text"]) for page in result["pages"]),
                "extraction_time": result.get("extraction_time", 0)
            }
            
            content.append(create_text_content(f"Text Extraction Summary:\n{json.dumps(summary, indent=2)}"))
            
            # Add extracted text
            for page_info in result["pages"]:
                page_num = page_info["page_number"]
                text = page_info["text"]
                
                if include_metadata and "metadata" in page_info:
                    metadata = page_info["metadata"]
                    content.append(create_text_content(
                        f"Page {page_num} Metadata:\n{json.dumps(metadata, indent=2)}"
                    ))
                
                content.append(create_text_content(f"Page {page_num} Text:\n{text}"))
                
                if extract_annotations and "annotations" in page_info:
                    annotations = page_info["annotations"]
                    if annotations:
                        content.append(create_text_content(
                            f"Page {page_num} Annotations:\n{json.dumps(annotations, indent=2)}"
                        ))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            content = [create_error_content(f"Text extraction failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _choose_best_method(self) -> str:
        """Choose the best available extraction method."""
        if fitz:
            return "pymupdf"
        elif pdfplumber:
            return "pdfplumber"
        elif PyPDF2:
            return "pypdf2"
        else:
            raise ToolExecutionException("No PDF processing library available")
    
    async def _extract_with_pymupdf(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        include_metadata: bool,
        preserve_layout: bool,
        extract_annotations: bool
    ) -> Dict[str, Any]:
        """Extract text using PyMuPDF."""
        if not fitz:
            raise ToolExecutionException("PyMuPDF not available")
        
        start_time = datetime.now()
        
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        # Determine pages to process
        if pages:
            page_numbers = [p - 1 for p in pages if 1 <= p <= total_pages]  # Convert to 0-indexed
        else:
            page_numbers = list(range(total_pages))
        
        extracted_pages = []
        
        for page_num in page_numbers:
            page = doc[page_num]
            
            # Extract text
            if preserve_layout:
                text = page.get_text("dict")
                # Convert dict format to readable text while preserving layout
                text = self._format_pymupdf_dict_text(text)
            else:
                text = page.get_text()
            
            page_info = {
                "page_number": page_num + 1,  # Convert back to 1-indexed
                "text": text
            }
            
            # Add metadata if requested
            if include_metadata:
                page_info["metadata"] = {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation,
                    "text_length": len(text)
                }
            
            # Extract annotations if requested
            if extract_annotations:
                annotations = []
                for annot in page.annots():
                    annot_info = {
                        "type": annot.type[1],  # Get annotation type name
                        "content": annot.info.get("content", ""),
                        "author": annot.info.get("title", ""),
                        "rect": list(annot.rect)
                    }
                    annotations.append(annot_info)
                page_info["annotations"] = annotations
            
            extracted_pages.append(page_info)
        
        doc.close()
        
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        return {
            "total_pages": total_pages,
            "pages": extracted_pages,
            "extraction_time": extraction_time
        }
    
    async def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        include_metadata: bool,
        preserve_layout: bool
    ) -> Dict[str, Any]:
        """Extract text using pdfplumber."""
        if not pdfplumber:
            raise ToolExecutionException("pdfplumber not available")
        
        start_time = datetime.now()
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            
            # Determine pages to process
            if pages:
                page_numbers = [p - 1 for p in pages if 1 <= p <= total_pages]  # Convert to 0-indexed
            else:
                page_numbers = list(range(total_pages))
            
            extracted_pages = []
            
            for page_num in page_numbers:
                page = pdf.pages[page_num]
                
                # Extract text
                if preserve_layout:
                    text = page.extract_text(layout=True)
                else:
                    text = page.extract_text()
                
                if text is None:
                    text = ""
                
                page_info = {
                    "page_number": page_num + 1,  # Convert back to 1-indexed
                    "text": text
                }
                
                # Add metadata if requested
                if include_metadata:
                    page_info["metadata"] = {
                        "width": page.width,
                        "height": page.height,
                        "rotation": getattr(page, 'rotation', 0),
                        "text_length": len(text),
                        "char_count": len([c for c in page.chars]),
                        "word_count": len(page.extract_words())
                    }
                
                extracted_pages.append(page_info)
        
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        return {
            "total_pages": total_pages,
            "pages": extracted_pages,
            "extraction_time": extraction_time
        }
    
    async def _extract_with_pypdf2(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Extract text using PyPDF2."""
        if not PyPDF2:
            raise ToolExecutionException("PyPDF2 not available")
        
        start_time = datetime.now()
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            # Determine pages to process
            if pages:
                page_numbers = [p - 1 for p in pages if 1 <= p <= total_pages]  # Convert to 0-indexed
            else:
                page_numbers = list(range(total_pages))
            
            extracted_pages = []
            
            for page_num in page_numbers:
                page = reader.pages[page_num]
                
                # Extract text
                text = page.extract_text()
                
                page_info = {
                    "page_number": page_num + 1,  # Convert back to 1-indexed
                    "text": text
                }
                
                # Add metadata if requested
                if include_metadata:
                    page_info["metadata"] = {
                        "rotation": page.rotation,
                        "text_length": len(text)
                    }
                    
                    # Try to get page dimensions
                    try:
                        mediabox = page.mediabox
                        page_info["metadata"].update({
                            "width": float(mediabox.width),
                            "height": float(mediabox.height)
                        })
                    except:
                        pass
                
                extracted_pages.append(page_info)
        
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        return {
            "total_pages": total_pages,
            "pages": extracted_pages,
            "extraction_time": extraction_time
        }
    
    def _format_pymupdf_dict_text(self, text_dict: Dict) -> str:
        """Format PyMuPDF dict text output to readable text."""
        lines = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    if line_text.strip():
                        lines.append(line_text)
        
        return "\n".join(lines)


class ExtractMetadataTool(PDFTool):
    """Extract metadata and document information from PDF files."""
    
    def __init__(self):
        super().__init__(
            name="extract_metadata",
            description="Extract metadata and document information from PDF files"
        )
    
    @property
    def input_schema(self) -> type[BaseModel]:
        """Return the input schema for this tool."""
        from pydantic import BaseModel, Field
        
        class ExtractMetadataInput(BaseModel):
            file_path: str = Field(description="Path to the PDF file")
            include_structure: bool = Field(default=False, description="Include document structure information")
            include_fonts: bool = Field(default=False, description="Include font information")
            include_images: bool = Field(default=False, description="Include image information")
        
        return ExtractMetadataInput
    
    @property
    def output_schema(self) -> type[BaseModel]:
        """Return the output schema for this tool."""
        from pydantic import BaseModel, Field
        from typing import Dict, Any, Optional
        
        class ExtractMetadataOutput(BaseModel):
            success: bool = Field(description="Whether the operation was successful")
            metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extracted metadata")
            error: Optional[str] = Field(default=None, description="Error message if operation failed")
        
        return ExtractMetadataOutput
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "include_structure": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include document structure information"
                },
                "include_fonts": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include font information"
                },
                "include_images": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include image information"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        include_structure = kwargs.get("include_structure", False)
        include_fonts = kwargs.get("include_fonts", False)
        include_images = kwargs.get("include_images", False)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Extract metadata using the best available method
            if fitz:
                metadata = await self._extract_metadata_pymupdf(
                    pdf_path, include_structure, include_fonts, include_images
                )
            elif PyPDF2:
                metadata = await self._extract_metadata_pypdf2(pdf_path)
            else:
                raise ToolExecutionException("No PDF processing library available for metadata extraction")
            
            # Format result
            content = [
                create_text_content(f"PDF Metadata:\n{json.dumps(metadata, indent=2, default=str)}")
            ]
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            content = [create_error_content(f"Metadata extraction failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    async def _extract_metadata_pymupdf(
        self,
        pdf_path: Path,
        include_structure: bool,
        include_fonts: bool,
        include_images: bool
    ) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF."""
        doc = fitz.open(str(pdf_path))
        
        # Basic metadata
        metadata = {
            "file_info": {
                "path": str(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
                "page_count": len(doc),
                "is_pdf": doc.is_pdf,
                "is_encrypted": doc.needs_pass,
                "can_copy": not doc.is_encrypted or doc.permissions & fitz.PDF_PERM_COPY,
                "can_print": not doc.is_encrypted or doc.permissions & fitz.PDF_PERM_PRINT
            },
            "document_metadata": doc.metadata
        }
        
        # Page information
        pages_info = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_info = {
                "page_number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "rotation": page.rotation
            }
            pages_info.append(page_info)
        
        metadata["pages"] = pages_info
        
        # Structure information
        if include_structure:
            toc = doc.get_toc()
            metadata["table_of_contents"] = toc
            
            # Form fields
            try:
                form_fields = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    widgets = page.widgets()
                    for widget in widgets:
                        field_info = {
                            "page": page_num + 1,
                            "field_name": widget.field_name,
                            "field_type": widget.field_type,
                            "rect": list(widget.rect)
                        }
                        form_fields.append(field_info)
                metadata["form_fields"] = form_fields
            except:
                pass
        
        # Font information
        if include_fonts:
            fonts = set()
            for page_num in range(len(doc)):
                page = doc[page_num]
                font_list = page.get_fonts()
                for font in font_list:
                    fonts.add(font[3])  # Font name
            metadata["fonts"] = list(fonts)
        
        # Image information
        if include_images:
            images_info = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    img_info = {
                        "page": page_num + 1,
                        "index": img_index,
                        "xref": img[0],
                        "width": img[2],
                        "height": img[3]
                    }
                    images_info.append(img_info)
            metadata["images"] = images_info
        
        doc.close()
        return metadata
    
    async def _extract_metadata_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata using PyPDF2."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            metadata = {
                "file_info": {
                    "path": str(pdf_path),
                    "size_bytes": pdf_path.stat().st_size,
                    "page_count": len(reader.pages),
                    "is_encrypted": reader.is_encrypted
                },
                "document_metadata": dict(reader.metadata) if reader.metadata else {}
            }
            
            # Page information
            pages_info = []
            for page_num, page in enumerate(reader.pages):
                page_info = {
                    "page_number": page_num + 1,
                    "rotation": page.rotation
                }
                
                # Try to get page dimensions
                try:
                    mediabox = page.mediabox
                    page_info.update({
                        "width": float(mediabox.width),
                        "height": float(mediabox.height)
                    })
                except:
                    pass
                
                pages_info.append(page_info)
            
            metadata["pages"] = pages_info
            
            return metadata