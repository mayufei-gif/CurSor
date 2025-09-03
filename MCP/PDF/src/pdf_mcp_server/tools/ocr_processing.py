#!/usr/bin/env python3
"""
PDF OCR Processing Tools

Implements OCR functionality for PDF files using multiple OCR engines
including Tesseract, EasyOCR, and OCRmyPDF for comprehensive text recognition.

Author: PDF-MCP Team
License: MIT
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
import subprocess

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class ProcessOCRTool(PDFTool):
    """Process OCR on PDF files using multiple OCR engines."""
    
    def __init__(self):
        super().__init__(
            name="process_ocr",
            description="Process OCR on PDF files using multiple OCR engines",
            version="1.0.0"
        )
        self._easyocr_reader = None
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "engine": {
                    "type": "string",
                    "enum": ["auto", "tesseract", "easyocr", "ocrmypdf"],
                    "default": "auto",
                    "description": "OCR engine to use"
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["eng"],
                    "description": "Languages for OCR recognition (ISO 639-1 codes)"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to process (1-indexed). If not provided, processes all pages"
                },
                "dpi": {
                    "type": "integer",
                    "default": 300,
                    "description": "DPI for image conversion (higher = better quality, slower processing)"
                },
                "preprocess": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply image preprocessing to improve OCR accuracy"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.6,
                    "description": "Minimum confidence threshold for OCR results (0.0-1.0)"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["text", "json", "hocr", "pdf"],
                    "default": "text",
                    "description": "Output format for OCR results"
                },
                "preserve_layout": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preserve original layout and formatting"
                },
                "extract_tables": {
                    "type": "boolean",
                    "default": False,
                    "description": "Attempt to extract table structures from OCR"
                },
                "denoise": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply denoising to improve OCR accuracy"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        engine = kwargs.get("engine", "auto")
        languages = kwargs.get("languages", ["eng"])
        pages = kwargs.get("pages")
        dpi = kwargs.get("dpi", 300)
        preprocess = kwargs.get("preprocess", True)
        confidence_threshold = kwargs.get("confidence_threshold", 0.6)
        output_format = kwargs.get("output_format", "text")
        preserve_layout = kwargs.get("preserve_layout", True)
        extract_tables = kwargs.get("extract_tables", False)
        denoise = kwargs.get("denoise", True)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Choose OCR engine
            if engine == "auto":
                engine = self._choose_best_engine()
            
            # Process OCR based on engine
            if engine == "tesseract":
                result = await self._process_with_tesseract(
                    pdf_path, languages, pages, dpi, preprocess, 
                    confidence_threshold, preserve_layout, denoise
                )
            elif engine == "easyocr":
                result = await self._process_with_easyocr(
                    pdf_path, languages, pages, dpi, preprocess, 
                    confidence_threshold, preserve_layout
                )
            elif engine == "ocrmypdf":
                result = await self._process_with_ocrmypdf(
                    pdf_path, languages, pages, output_format
                )
            else:
                raise ToolExecutionException(f"Unknown OCR engine: {engine}")
            
            # Extract tables if requested
            if extract_tables and result.get("pages"):
                result["tables"] = await self._extract_tables_from_ocr(result["pages"])
            
            # Format output
            content = []
            
            # Add summary
            summary = {
                "file": str(pdf_path),
                "engine": engine,
                "languages": languages,
                "total_pages": len(result.get("pages", [])),
                "processing_time": result.get("processing_time", 0),
                "average_confidence": result.get("average_confidence", 0),
                "total_words": result.get("total_words", 0)
            }
            
            content.append(create_text_content(f"OCR Processing Summary:\n{json.dumps(summary, indent=2)}"))
            
            # Add OCR results based on format
            if output_format == "text":
                full_text = "\n\n".join([page["text"] for page in result.get("pages", [])])
                content.append(create_text_content(f"Extracted Text:\n{full_text}"))
            
            elif output_format == "json":
                content.append(create_text_content(
                    f"OCR Results (JSON):\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```"
                ))
            
            elif output_format == "hocr":
                if "hocr" in result:
                    content.append(create_text_content(
                        f"HOCR Output:\n```html\n{result['hocr']}\n```"
                    ))
            
            # Add page-by-page results for detailed analysis
            for page_info in result.get("pages", []):
                page_num = page_info["page_number"]
                page_text = page_info["text"]
                confidence = page_info.get("confidence", 0)
                word_count = page_info.get("word_count", 0)
                
                content.append(create_text_content(
                    f"Page {page_num} (Confidence: {confidence:.2f}, Words: {word_count}):\n{page_text[:500]}{'...' if len(page_text) > 500 else ''}"
                ))
            
            # Add table extraction results if available
            if extract_tables and result.get("tables"):
                content.append(create_text_content(
                    f"Extracted Tables:\n```json\n{json.dumps(result['tables'], indent=2)}\n```"
                ))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            content = [create_error_content(f"OCR processing failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _choose_best_engine(self) -> str:
        """Choose the best available OCR engine."""
        if pytesseract:
            return "tesseract"
        elif easyocr:
            return "easyocr"
        else:
            # Try OCRmyPDF as fallback
            try:
                subprocess.run(["ocrmypdf", "--version"], capture_output=True, check=True)
                return "ocrmypdf"
            except:
                pass
        
        raise ToolExecutionException("No OCR engine available")
    
    async def _process_with_tesseract(
        self,
        pdf_path: Path,
        languages: List[str],
        pages: Optional[List[int]],
        dpi: int,
        preprocess: bool,
        confidence_threshold: float,
        preserve_layout: bool,
        denoise: bool
    ) -> Dict[str, Any]:
        """Process OCR using Tesseract."""
        if not pytesseract or not fitz:
            raise ToolExecutionException("Tesseract or PyMuPDF not available")
        
        start_time = datetime.now()
        
        # Convert language codes for Tesseract
        tesseract_langs = "+".join(languages)
        
        # Configure Tesseract
        config = "--oem 3 --psm 6"
        if preserve_layout:
            config += " -c preserve_interword_spaces=1"
        
        results = {
            "pages": [],
            "engine": "tesseract",
            "languages": languages
        }
        
        total_words = 0
        total_confidence = 0
        confidence_count = 0
        
        # Open PDF
        pdf_doc = fitz.open(str(pdf_path))
        
        try:
            # Determine pages to process
            if pages:
                page_numbers = [p - 1 for p in pages if 1 <= p <= len(pdf_doc)]  # Convert to 0-indexed
            else:
                page_numbers = list(range(len(pdf_doc)))
            
            for page_num in page_numbers:
                page = pdf_doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor for DPI
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Preprocess image if requested
                if preprocess:
                    image = await self._preprocess_image(image, denoise)
                
                # Perform OCR
                try:
                    # Get text with confidence data
                    ocr_data = pytesseract.image_to_data(
                        image, lang=tesseract_langs, config=config, output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract text and calculate confidence
                    page_text = ""
                    page_words = []
                    page_confidence_sum = 0
                    page_confidence_count = 0
                    
                    for i in range(len(ocr_data['text'])):
                        word = ocr_data['text'][i].strip()
                        confidence = int(ocr_data['conf'][i])
                        
                        if word and confidence > confidence_threshold * 100:
                            page_text += word + " "
                            page_words.append({
                                "text": word,
                                "confidence": confidence / 100,
                                "bbox": [
                                    ocr_data['left'][i],
                                    ocr_data['top'][i],
                                    ocr_data['left'][i] + ocr_data['width'][i],
                                    ocr_data['top'][i] + ocr_data['height'][i]
                                ]
                            })
                            
                            page_confidence_sum += confidence
                            page_confidence_count += 1
                    
                    page_confidence = page_confidence_sum / page_confidence_count if page_confidence_count > 0 else 0
                    
                    page_info = {
                        "page_number": page_num + 1,
                        "text": page_text.strip(),
                        "confidence": page_confidence / 100,
                        "word_count": len(page_words),
                        "words": page_words if preserve_layout else None
                    }
                    
                    results["pages"].append(page_info)
                    
                    total_words += len(page_words)
                    total_confidence += page_confidence
                    confidence_count += 1
                
                except Exception as e:
                    self.logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    results["pages"].append({
                        "page_number": page_num + 1,
                        "text": "",
                        "confidence": 0,
                        "word_count": 0,
                        "error": str(e)
                    })
        
        finally:
            pdf_doc.close()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results.update({
            "processing_time": processing_time,
            "total_words": total_words,
            "average_confidence": total_confidence / confidence_count if confidence_count > 0 else 0
        })
        
        return results
    
    async def _process_with_easyocr(
        self,
        pdf_path: Path,
        languages: List[str],
        pages: Optional[List[int]],
        dpi: int,
        preprocess: bool,
        confidence_threshold: float,
        preserve_layout: bool
    ) -> Dict[str, Any]:
        """Process OCR using EasyOCR."""
        if not easyocr or not fitz:
            raise ToolExecutionException("EasyOCR or PyMuPDF not available")
        
        start_time = datetime.now()
        
        # Initialize EasyOCR reader if not already done
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(languages)
        
        results = {
            "pages": [],
            "engine": "easyocr",
            "languages": languages
        }
        
        total_words = 0
        total_confidence = 0
        confidence_count = 0
        
        # Open PDF
        pdf_doc = fitz.open(str(pdf_path))
        
        try:
            # Determine pages to process
            if pages:
                page_numbers = [p - 1 for p in pages if 1 <= p <= len(pdf_doc)]  # Convert to 0-indexed
            else:
                page_numbers = list(range(len(pdf_doc)))
            
            for page_num in page_numbers:
                page = pdf_doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor for DPI
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array for EasyOCR
                if cv2 and np:
                    nparr = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Preprocess image if requested
                    if preprocess:
                        image = await self._preprocess_image_cv2(image)
                    
                    # Perform OCR
                    try:
                        results_list = self._easyocr_reader.readtext(image)
                        
                        page_text = ""
                        page_words = []
                        page_confidence_sum = 0
                        page_confidence_count = 0
                        
                        for (bbox, text, confidence) in results_list:
                            if confidence > confidence_threshold:
                                page_text += text + " "
                                
                                if preserve_layout:
                                    page_words.append({
                                        "text": text,
                                        "confidence": confidence,
                                        "bbox": bbox
                                    })
                                
                                page_confidence_sum += confidence
                                page_confidence_count += 1
                        
                        page_confidence = page_confidence_sum / page_confidence_count if page_confidence_count > 0 else 0
                        
                        page_info = {
                            "page_number": page_num + 1,
                            "text": page_text.strip(),
                            "confidence": page_confidence,
                            "word_count": len(page_words) if preserve_layout else len(page_text.split()),
                            "words": page_words if preserve_layout else None
                        }
                        
                        results["pages"].append(page_info)
                        
                        total_words += page_info["word_count"]
                        total_confidence += page_confidence
                        confidence_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"EasyOCR failed for page {page_num + 1}: {e}")
                        results["pages"].append({
                            "page_number": page_num + 1,
                            "text": "",
                            "confidence": 0,
                            "word_count": 0,
                            "error": str(e)
                        })
                else:
                    raise ToolExecutionException("OpenCV not available for EasyOCR")
        
        finally:
            pdf_doc.close()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results.update({
            "processing_time": processing_time,
            "total_words": total_words,
            "average_confidence": total_confidence / confidence_count if confidence_count > 0 else 0
        })
        
        return results
    
    async def _process_with_ocrmypdf(
        self,
        pdf_path: Path,
        languages: List[str],
        pages: Optional[List[int]],
        output_format: str
    ) -> Dict[str, Any]:
        """Process OCR using OCRmyPDF."""
        start_time = datetime.now()
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_output = Path(temp_file.name)
        
        try:
            # Build OCRmyPDF command
            cmd = ["ocrmypdf"]
            
            # Add language options
            if languages:
                cmd.extend(["-l", "+".join(languages)])
            
            # Add page selection if specified
            if pages:
                page_range = ",".join(map(str, pages))
                cmd.extend(["--pages", page_range])
            
            # Add other options
            cmd.extend([
                "--force-ocr",  # Force OCR even if text already exists
                "--optimize", "1",  # Basic optimization
                "--output-type", "pdf"
            ])
            
            cmd.extend([str(pdf_path), str(temp_output)])
            
            # Run OCRmyPDF
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise ToolExecutionException(f"OCRmyPDF failed: {stderr.decode()}")
            
            # Extract text from the OCR'd PDF
            if fitz:
                ocr_doc = fitz.open(str(temp_output))
                
                results = {
                    "pages": [],
                    "engine": "ocrmypdf",
                    "languages": languages
                }
                
                total_words = 0
                
                for page_num in range(len(ocr_doc)):
                    page = ocr_doc[page_num]
                    text = page.get_text()
                    word_count = len(text.split())
                    
                    page_info = {
                        "page_number": page_num + 1,
                        "text": text,
                        "confidence": 1.0,  # OCRmyPDF doesn't provide confidence scores
                        "word_count": word_count
                    }
                    
                    results["pages"].append(page_info)
                    total_words += word_count
                
                ocr_doc.close()
            else:
                raise ToolExecutionException("PyMuPDF not available for text extraction")
        
        finally:
            # Clean up temporary file
            if temp_output.exists():
                temp_output.unlink()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results.update({
            "processing_time": processing_time,
            "total_words": total_words,
            "average_confidence": 1.0
        })
        
        return results
    
    async def _preprocess_image(self, image: Any, denoise: bool = True) -> Any:
        """Preprocess PIL image to improve OCR accuracy."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        if cv2 and np:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Apply denoising
            if denoise:
                img_array = cv2.fastNlMeansDenoising(img_array)
            
            # Apply adaptive thresholding
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL
            image = Image.fromarray(img_array)
        
        return image
    
    async def _preprocess_image_cv2(self, image: Any) -> Any:
        """Preprocess OpenCV image to improve OCR accuracy."""
        if not cv2:
            return image
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    async def _extract_tables_from_ocr(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract table structures from OCR results."""
        tables = []
        
        for page_info in pages:
            if "words" in page_info and page_info["words"]:
                # Simple table detection based on word positions
                words = page_info["words"]
                page_tables = self._detect_tables_from_words(words)
                
                for table in page_tables:
                    table["page"] = page_info["page_number"]
                    tables.append(table)
        
        return tables
    
    def _detect_tables_from_words(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect table structures from word positions."""
        # This is a simplified table detection algorithm
        # In practice, you might want to use more sophisticated methods
        
        tables = []
        
        # Group words by approximate Y position (rows)
        rows = {}
        for word in words:
            bbox = word["bbox"]
            y_pos = bbox[1]  # Top Y coordinate
            
            # Find the closest row
            closest_row = None
            min_distance = float('inf')
            
            for row_y in rows.keys():
                distance = abs(y_pos - row_y)
                if distance < min_distance and distance < 10:  # Threshold for same row
                    min_distance = distance
                    closest_row = row_y
            
            if closest_row is not None:
                rows[closest_row].append(word)
            else:
                rows[y_pos] = [word]
        
        # Sort rows by Y position
        sorted_rows = sorted(rows.items())
        
        # Check if this looks like a table (multiple rows with similar column structure)
        if len(sorted_rows) >= 3:  # At least 3 rows
            # Analyze column structure
            column_positions = []
            for _, row_words in sorted_rows:
                row_words.sort(key=lambda w: w["bbox"][0])  # Sort by X position
                for word in row_words:
                    x_pos = word["bbox"][0]
                    
                    # Find closest column
                    closest_col = None
                    min_distance = float('inf')
                    
                    for col_x in column_positions:
                        distance = abs(x_pos - col_x)
                        if distance < min_distance and distance < 50:  # Threshold for same column
                            min_distance = distance
                            closest_col = col_x
                    
                    if closest_col is None:
                        column_positions.append(x_pos)
            
            # If we have consistent columns, create a table
            if len(column_positions) >= 2:
                table_data = []
                
                for _, row_words in sorted_rows:
                    row_words.sort(key=lambda w: w["bbox"][0])
                    row_data = []
                    
                    for col_x in sorted(column_positions):
                        # Find word closest to this column position
                        closest_word = None
                        min_distance = float('inf')
                        
                        for word in row_words:
                            distance = abs(word["bbox"][0] - col_x)
                            if distance < min_distance and distance < 50:
                                min_distance = distance
                                closest_word = word
                        
                        if closest_word:
                            row_data.append(closest_word["text"])
                        else:
                            row_data.append("")
                    
                    table_data.append(row_data)
                
                tables.append({
                    "data": table_data,
                    "rows": len(table_data),
                    "columns": len(column_positions),
                    "confidence": 0.7  # Estimated confidence for table detection
                })
        
        return tables


class OCRWithLayoutTool(PDFTool):
    """OCR processing with advanced layout analysis and preservation."""
    
    def __init__(self):
        super().__init__(
            name="ocr_with_layout",
            description="OCR processing with advanced layout analysis and preservation",
            version="1.0.0"
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "engine": {
                    "type": "string",
                    "enum": ["auto", "tesseract", "easyocr"],
                    "default": "auto",
                    "description": "OCR engine to use"
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["eng"],
                    "description": "Languages for OCR recognition"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to process (1-indexed)"
                },
                "layout_analysis": {
                    "type": "boolean",
                    "default": True,
                    "description": "Perform detailed layout analysis"
                },
                "detect_columns": {
                    "type": "boolean",
                    "default": True,
                    "description": "Detect and preserve column layout"
                },
                "detect_reading_order": {
                    "type": "boolean",
                    "default": True,
                    "description": "Detect and preserve reading order"
                },
                "extract_regions": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract different text regions (headers, paragraphs, etc.)"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["structured", "markdown", "html", "json"],
                    "default": "structured",
                    "description": "Output format preserving layout information"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        engine = kwargs.get("engine", "auto")
        languages = kwargs.get("languages", ["eng"])
        pages = kwargs.get("pages")
        layout_analysis = kwargs.get("layout_analysis", True)
        detect_columns = kwargs.get("detect_columns", True)
        detect_reading_order = kwargs.get("detect_reading_order", True)
        extract_regions = kwargs.get("extract_regions", True)
        output_format = kwargs.get("output_format", "structured")
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # First, perform basic OCR
            basic_ocr_tool = ProcessOCRTool()
            basic_result = await basic_ocr_tool.execute(
                file_path=file_path,
                engine=engine,
                languages=languages,
                pages=pages,
                preserve_layout=True,
                output_format="json"
            )
            
            if basic_result.isError:
                return basic_result
            
            # Parse basic OCR results
            ocr_data = self._parse_ocr_result(basic_result)
            
            # Perform layout analysis
            if layout_analysis:
                for page_data in ocr_data.get("pages", []):
                    if "words" in page_data and page_data["words"]:
                        # Analyze layout
                        layout_info = await self._analyze_page_layout(
                            page_data["words"], detect_columns, detect_reading_order, extract_regions
                        )
                        page_data["layout"] = layout_info
            
            # Format output based on requested format
            content = []
            
            # Add summary
            summary = {
                "file": str(pdf_path),
                "engine": engine,
                "total_pages": len(ocr_data.get("pages", [])),
                "layout_analysis": layout_analysis,
                "features": {
                    "column_detection": detect_columns,
                    "reading_order": detect_reading_order,
                    "region_extraction": extract_regions
                }
            }
            
            content.append(create_text_content(f"OCR with Layout Analysis Summary:\n{json.dumps(summary, indent=2)}"))
            
            # Format results based on output format
            if output_format == "structured":
                content.append(create_text_content(
                    f"Structured OCR Results:\n```json\n{json.dumps(ocr_data, indent=2, ensure_ascii=False)}\n```"
                ))
            
            elif output_format == "markdown":
                markdown_content = await self._format_as_markdown(ocr_data)
                content.append(create_text_content(f"Markdown Output:\n{markdown_content}"))
            
            elif output_format == "html":
                html_content = await self._format_as_html(ocr_data)
                content.append(create_text_content(f"HTML Output:\n```html\n{html_content}\n```"))
            
            elif output_format == "json":
                content.append(create_text_content(
                    f"JSON Output:\n```json\n{json.dumps(ocr_data, indent=2, ensure_ascii=False)}\n```"
                ))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"OCR with layout analysis failed: {e}")
            content = [create_error_content(f"OCR with layout analysis failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _parse_ocr_result(self, result: MCPToolResult) -> Dict[str, Any]:
        """Parse OCR result from the basic OCR tool."""
        for content_item in result.content:
            if content_item.get("type") == "text":
                text = content_item.get("text", "")
                if "```json" in text:
                    try:
                        start = text.find("```json") + 7
                        end = text.find("```", start)
                        if start > 6 and end > start:
                            json_str = text[start:end].strip()
                            return json.loads(json_str)
                    except:
                        continue
        
        return {"pages": []}
    
    async def _analyze_page_layout(
        self,
        words: List[Dict[str, Any]],
        detect_columns: bool,
        detect_reading_order: bool,
        extract_regions: bool
    ) -> Dict[str, Any]:
        """Analyze page layout from word positions."""
        layout_info = {
            "columns": [],
            "reading_order": [],
            "regions": []
        }
        
        if detect_columns:
            layout_info["columns"] = self._detect_columns(words)
        
        if detect_reading_order:
            layout_info["reading_order"] = self._detect_reading_order(words)
        
        if extract_regions:
            layout_info["regions"] = self._extract_text_regions(words)
        
        return layout_info
    
    def _detect_columns(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect column layout from word positions."""
        if not words:
            return []
        
        # Group words by X position to detect columns
        x_positions = [word["bbox"][0] for word in words]
        x_positions.sort()
        
        # Find gaps that might indicate column boundaries
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Threshold for column gap
                gaps.append((x_positions[i-1], x_positions[i], gap))
        
        # Create column definitions
        columns = []
        if gaps:
            # First column
            columns.append({
                "index": 0,
                "left": min(x_positions),
                "right": gaps[0][0],
                "width": gaps[0][0] - min(x_positions)
            })
            
            # Middle columns
            for i in range(len(gaps) - 1):
                columns.append({
                    "index": i + 1,
                    "left": gaps[i][1],
                    "right": gaps[i+1][0],
                    "width": gaps[i+1][0] - gaps[i][1]
                })
            
            # Last column
            columns.append({
                "index": len(gaps),
                "left": gaps[-1][1],
                "right": max(x_positions),
                "width": max(x_positions) - gaps[-1][1]
            })
        else:
            # Single column
            columns.append({
                "index": 0,
                "left": min(x_positions),
                "right": max(x_positions),
                "width": max(x_positions) - min(x_positions)
            })
        
        return columns
    
    def _detect_reading_order(self, words: List[Dict[str, Any]]) -> List[int]:
        """Detect reading order of text elements."""
        if not words:
            return []
        
        # Sort words by position (top to bottom, left to right)
        sorted_words = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))
        
        # Return indices in reading order
        reading_order = []
        for word in sorted_words:
            original_index = words.index(word)
            reading_order.append(original_index)
        
        return reading_order
    
    def _extract_text_regions(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract different text regions (headers, paragraphs, etc.)."""
        if not words:
            return []
        
        regions = []
        
        # Group words by approximate Y position (lines)
        lines = {}
        for i, word in enumerate(words):
            bbox = word["bbox"]
            y_pos = bbox[1]
            
            # Find the closest line
            closest_line = None
            min_distance = float('inf')
            
            for line_y in lines.keys():
                distance = abs(y_pos - line_y)
                if distance < min_distance and distance < 10:
                    min_distance = distance
                    closest_line = line_y
            
            if closest_line is not None:
                lines[closest_line].append((i, word))
            else:
                lines[y_pos] = [(i, word)]
        
        # Analyze each line to determine region type
        sorted_lines = sorted(lines.items())
        
        for line_y, line_words in sorted_lines:
            line_words.sort(key=lambda x: x[1]["bbox"][0])  # Sort by X position
            
            # Combine words into text
            line_text = " ".join([word[1]["text"] for word in line_words])
            
            # Determine region type based on characteristics
            region_type = self._classify_text_region(line_text, line_words)
            
            # Calculate bounding box for the entire line
            left = min([word[1]["bbox"][0] for word in line_words])
            top = min([word[1]["bbox"][1] for word in line_words])
            right = max([word[1]["bbox"][2] for word in line_words])
            bottom = max([word[1]["bbox"][3] for word in line_words])
            
            regions.append({
                "type": region_type,
                "text": line_text,
                "bbox": [left, top, right, bottom],
                "word_indices": [word[0] for word in line_words],
                "confidence": sum([word[1]["confidence"] for word in line_words]) / len(line_words)
            })
        
        return regions
    
    def _classify_text_region(self, text: str, words: List[Tuple[int, Dict[str, Any]]]) -> str:
        """Classify text region type based on content and formatting."""
        text_lower = text.lower().strip()
        
        # Check for headers (short, potentially bold, at top)
        if len(text.split()) <= 10 and len(text) < 100:
            # Check if it looks like a title or header
            if any(word in text_lower for word in ['chapter', 'section', 'introduction', 'conclusion']):
                return "header"
            if text.isupper() or text.istitle():
                return "header"
        
        # Check for lists
        if text_lower.startswith(('•', '-', '*', '1.', '2.', '3.', 'a)', 'b)', 'c)')):
            return "list_item"
        
        # Check for tables (short lines with potential separators)
        if len(text.split()) <= 5 and any(char in text for char in ['|', '\t']):
            return "table_cell"
        
        # Check for captions
        if text_lower.startswith(('figure', 'table', 'chart', 'image')):
            return "caption"
        
        # Default to paragraph
        return "paragraph"
    
    async def _format_as_markdown(self, ocr_data: Dict[str, Any]) -> str:
        """Format OCR results as markdown preserving layout."""
        markdown_lines = []
        
        for page_data in ocr_data.get("pages", []):
            page_num = page_data.get("page_number", 1)
            markdown_lines.append(f"# Page {page_num}\n")
            
            if "layout" in page_data and "regions" in page_data["layout"]:
                # Use layout information to format text
                for region in page_data["layout"]["regions"]:
                    region_type = region.get("type", "paragraph")
                    text = region.get("text", "")
                    
                    if region_type == "header":
                        markdown_lines.append(f"## {text}\n")
                    elif region_type == "list_item":
                        markdown_lines.append(f"- {text}\n")
                    elif region_type == "caption":
                        markdown_lines.append(f"*{text}*\n")
                    else:
                        markdown_lines.append(f"{text}\n")
            else:
                # Fallback to simple text
                text = page_data.get("text", "")
                markdown_lines.append(f"{text}\n")
            
            markdown_lines.append("\n---\n\n")  # Page separator
        
        return "".join(markdown_lines)
    
    async def _format_as_html(self, ocr_data: Dict[str, Any]) -> str:
        """Format OCR results as HTML preserving layout."""
        html_lines = ["<html><body>"]
        
        for page_data in ocr_data.get("pages", []):
            page_num = page_data.get("page_number", 1)
            html_lines.append(f"<div class='page' id='page-{page_num}'>")
            html_lines.append(f"<h1>Page {page_num}</h1>")
            
            if "layout" in page_data and "regions" in page_data["layout"]:
                # Use layout information to format text
                for region in page_data["layout"]["regions"]:
                    region_type = region.get("type", "paragraph")
                    text = region.get("text", "")
                    bbox = region.get("bbox", [0, 0, 0, 0])
                    
                    style = f"position: absolute; left: {bbox[0]}px; top: {bbox[1]}px; width: {bbox[2]-bbox[0]}px;"
                    
                    if region_type == "header":
                        html_lines.append(f"<h2 style='{style}'>{text}</h2>")
                    elif region_type == "list_item":
                        html_lines.append(f"<li style='{style}'>{text}</li>")
                    elif region_type == "caption":
                        html_lines.append(f"<em style='{style}'>{text}</em>")
                    else:
                        html_lines.append(f"<p style='{style}'>{text}</p>")
            else:
                # Fallback to simple text
                text = page_data.get("text", "")
                html_lines.append(f"<p>{text}</p>")
            
            html_lines.append("</div>")
        
        html_lines.append("</body></html>")
        return "".join(html_lines)


# Import io for image processing
import io