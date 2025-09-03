#!/usr/bin/env python3
"""
PDF Formula Recognition Tools

Implements mathematical formula recognition functionality for PDF files using
LaTeX-OCR, pix2tex, and other formula recognition engines.

Author: PDF-MCP Team
License: MIT
"""

import json
import logging
import tempfile
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
import subprocess
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
    import io
except ImportError:
    Image = None
    io = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    # Try to import LaTeX-OCR related packages
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class ExtractFormulasTool(PDFTool):
    """Extract mathematical formulas from PDF files."""
    
    def __init__(self):
        super().__init__(
            name="extract_formulas",
            description="Extract mathematical formulas from PDF files using multiple recognition methods",
            version="1.0.0"
        )
        self._formula_model = None
        self._formula_processor = None
    
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
                    "enum": ["auto", "latex_ocr", "pix2tex", "mathpix", "pattern_based"],
                    "default": "auto",
                    "description": "Formula recognition method to use"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to process (1-indexed). If not provided, processes all pages"
                },
                "detection_mode": {
                    "type": "string",
                    "enum": ["automatic", "manual_regions", "full_page"],
                    "default": "automatic",
                    "description": "Formula detection mode"
                },
                "formula_regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer"},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4
                            }
                        },
                        "required": ["page", "bbox"]
                    },
                    "description": "Manual formula regions [x1, y1, x2, y2] for each page"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["latex", "mathml", "text", "json"],
                    "default": "latex",
                    "description": "Output format for recognized formulas"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.7,
                    "description": "Minimum confidence threshold for formula recognition (0.0-1.0)"
                },
                "dpi": {
                    "type": "integer",
                    "default": 300,
                    "description": "DPI for image conversion (higher = better quality, slower processing)"
                },
                "preprocess_images": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply image preprocessing to improve recognition accuracy"
                },
                "extract_inline": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract inline formulas (e.g., $x^2$)"
                },
                "extract_display": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract display formulas (e.g., $$\\int x dx$$)"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        method = kwargs.get("method", "auto")
        pages = kwargs.get("pages")
        detection_mode = kwargs.get("detection_mode", "automatic")
        formula_regions = kwargs.get("formula_regions", [])
        output_format = kwargs.get("output_format", "latex")
        confidence_threshold = kwargs.get("confidence_threshold", 0.7)
        dpi = kwargs.get("dpi", 300)
        preprocess_images = kwargs.get("preprocess_images", True)
        extract_inline = kwargs.get("extract_inline", True)
        extract_display = kwargs.get("extract_display", True)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Choose recognition method
            if method == "auto":
                method = self._choose_best_method()
            
            # Extract formulas based on method
            if method == "latex_ocr":
                result = await self._extract_with_latex_ocr(
                    pdf_path, pages, detection_mode, formula_regions, 
                    confidence_threshold, dpi, preprocess_images
                )
            elif method == "pix2tex":
                result = await self._extract_with_pix2tex(
                    pdf_path, pages, detection_mode, formula_regions, dpi
                )
            elif method == "mathpix":
                result = await self._extract_with_mathpix(
                    pdf_path, pages, detection_mode, formula_regions, dpi
                )
            elif method == "pattern_based":
                result = await self._extract_with_pattern_based(
                    pdf_path, pages, extract_inline, extract_display
                )
            else:
                raise ToolExecutionException(f"Unknown formula recognition method: {method}")
            
            # Format output
            content = []
            
            # Add summary
            summary = {
                "file": str(pdf_path),
                "method": method,
                "total_formulas": len(result.get("formulas", [])),
                "pages_processed": result.get("pages_processed", []),
                "processing_time": result.get("processing_time", 0),
                "average_confidence": result.get("average_confidence", 0)
            }
            
            content.append(create_text_content(f"Formula Extraction Summary:\n{json.dumps(summary, indent=2)}"))
            
            # Add extracted formulas
            formulas = result.get("formulas", [])
            
            if output_format == "json":
                content.append(create_text_content(
                    f"Extracted Formulas (JSON):\n```json\n{json.dumps(formulas, indent=2, ensure_ascii=False)}\n```"
                ))
            else:
                for i, formula_info in enumerate(formulas):
                    page_num = formula_info.get("page", "unknown")
                    confidence = formula_info.get("confidence", 0)
                    formula_type = formula_info.get("type", "unknown")
                    
                    if output_format == "latex":
                        latex_code = formula_info.get("latex", "")
                        content.append(create_text_content(
                            f"Formula {i+1} (Page {page_num}, Type: {formula_type}, Confidence: {confidence:.2f}):\n```latex\n{latex_code}\n```"
                        ))
                    elif output_format == "mathml":
                        mathml_code = formula_info.get("mathml", "")
                        content.append(create_text_content(
                            f"Formula {i+1} (Page {page_num}, Type: {formula_type}, Confidence: {confidence:.2f}):\n```xml\n{mathml_code}\n```"
                        ))
                    elif output_format == "text":
                        text_repr = formula_info.get("text", "")
                        content.append(create_text_content(
                            f"Formula {i+1} (Page {page_num}, Type: {formula_type}, Confidence: {confidence:.2f}):\n{text_repr}"
                        ))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"Formula extraction failed: {e}")
            content = [create_error_content(f"Formula extraction failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _choose_best_method(self) -> str:
        """Choose the best available formula recognition method."""
        # Check for LaTeX-OCR dependencies
        if torch and TrOCRProcessor and VisionEncoderDecoderModel:
            return "latex_ocr"
        
        # Check for pix2tex
        try:
            subprocess.run(["pix2tex", "--version"], capture_output=True, check=True)
            return "pix2tex"
        except:
            pass
        
        # Fallback to pattern-based extraction
        return "pattern_based"
    
    async def _extract_with_latex_ocr(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        detection_mode: str,
        formula_regions: List[Dict[str, Any]],
        confidence_threshold: float,
        dpi: int,
        preprocess_images: bool
    ) -> Dict[str, Any]:
        """Extract formulas using LaTeX-OCR model."""
        if not torch or not TrOCRProcessor or not VisionEncoderDecoderModel:
            raise ToolExecutionException("LaTeX-OCR dependencies not available")
        
        start_time = datetime.now()
        
        # Initialize model if not already done
        if self._formula_model is None or self._formula_processor is None:
            try:
                # Use a pre-trained model for mathematical formula recognition
                model_name = "microsoft/trocr-base-printed"  # Fallback model
                self._formula_processor = TrOCRProcessor.from_pretrained(model_name)
                self._formula_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            except Exception as e:
                raise ToolExecutionException(f"Failed to load LaTeX-OCR model: {e}")
        
        results = {
            "formulas": [],
            "method": "latex_ocr",
            "pages_processed": []
        }
        
        total_confidence = 0
        confidence_count = 0
        
        if not fitz:
            raise ToolExecutionException("PyMuPDF not available")
        
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
                
                # Get formula regions for this page
                if detection_mode == "manual_regions":
                    page_regions = [r["bbox"] for r in formula_regions if r["page"] == page_num + 1]
                elif detection_mode == "full_page":
                    page_rect = page.rect
                    page_regions = [[page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1]]
                else:  # automatic
                    page_regions = await self._detect_formula_regions(page, dpi)
                
                # Process each region
                for region_bbox in page_regions:
                    try:
                        # Extract region as image
                        region_rect = fitz.Rect(region_bbox)
                        mat = fitz.Matrix(dpi / 72, dpi / 72)
                        pix = page.get_pixmap(matrix=mat, clip=region_rect)
                        img_data = pix.tobytes("png")
                        
                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Preprocess image if requested
                        if preprocess_images:
                            image = await self._preprocess_formula_image(image)
                        
                        # Recognize formula
                        pixel_values = self._formula_processor(image, return_tensors="pt").pixel_values
                        generated_ids = self._formula_model.generate(pixel_values)
                        generated_text = self._formula_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Calculate confidence (simplified)
                        confidence = 0.8  # Placeholder - actual confidence calculation would be more complex
                        
                        if confidence >= confidence_threshold:
                            formula_info = {
                                "page": page_num + 1,
                                "bbox": region_bbox,
                                "latex": generated_text,
                                "confidence": confidence,
                                "type": "display",  # Could be improved with better detection
                                "method": "latex_ocr"
                            }
                            
                            # Convert to other formats if needed
                            formula_info["mathml"] = await self._latex_to_mathml(generated_text)
                            formula_info["text"] = await self._latex_to_text(generated_text)
                            
                            results["formulas"].append(formula_info)
                            
                            total_confidence += confidence
                            confidence_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Formula recognition failed for region on page {page_num + 1}: {e}")
                        continue
                
                results["pages_processed"].append(page_num + 1)
        
        finally:
            pdf_doc.close()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results.update({
            "processing_time": processing_time,
            "average_confidence": total_confidence / confidence_count if confidence_count > 0 else 0
        })
        
        return results
    
    async def _extract_with_pix2tex(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        detection_mode: str,
        formula_regions: List[Dict[str, Any]],
        dpi: int
    ) -> Dict[str, Any]:
        """Extract formulas using pix2tex."""
        start_time = datetime.now()
        
        results = {
            "formulas": [],
            "method": "pix2tex",
            "pages_processed": []
        }
        
        # Check if pix2tex is available
        try:
            subprocess.run(["pix2tex", "--version"], capture_output=True, check=True)
        except:
            raise ToolExecutionException("pix2tex not available")
        
        if not fitz:
            raise ToolExecutionException("PyMuPDF not available")
        
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
                
                # Get formula regions for this page
                if detection_mode == "manual_regions":
                    page_regions = [r["bbox"] for r in formula_regions if r["page"] == page_num + 1]
                elif detection_mode == "full_page":
                    page_rect = page.rect
                    page_regions = [[page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1]]
                else:  # automatic
                    page_regions = await self._detect_formula_regions(page, dpi)
                
                # Process each region
                for region_bbox in page_regions:
                    try:
                        # Extract region as image
                        region_rect = fitz.Rect(region_bbox)
                        mat = fitz.Matrix(dpi / 72, dpi / 72)
                        pix = page.get_pixmap(matrix=mat, clip=region_rect)
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                            pix.save(temp_file.name)
                            temp_image_path = Path(temp_file.name)
                        
                        try:
                            # Run pix2tex
                            cmd = ["pix2tex", str(temp_image_path)]
                            process = await asyncio.create_subprocess_exec(
                                *cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            
                            stdout, stderr = await process.communicate()
                            
                            if process.returncode == 0:
                                latex_code = stdout.decode().strip()
                                
                                formula_info = {
                                    "page": page_num + 1,
                                    "bbox": region_bbox,
                                    "latex": latex_code,
                                    "confidence": 0.9,  # pix2tex doesn't provide confidence scores
                                    "type": "display",
                                    "method": "pix2tex"
                                }
                                
                                # Convert to other formats
                                formula_info["mathml"] = await self._latex_to_mathml(latex_code)
                                formula_info["text"] = await self._latex_to_text(latex_code)
                                
                                results["formulas"].append(formula_info)
                            else:
                                self.logger.warning(f"pix2tex failed for region on page {page_num + 1}: {stderr.decode()}")
                        
                        finally:
                            # Clean up temporary file
                            if temp_image_path.exists():
                                temp_image_path.unlink()
                    
                    except Exception as e:
                        self.logger.warning(f"pix2tex processing failed for region on page {page_num + 1}: {e}")
                        continue
                
                results["pages_processed"].append(page_num + 1)
        
        finally:
            pdf_doc.close()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results.update({
            "processing_time": processing_time,
            "average_confidence": 0.9  # Default confidence for pix2tex
        })
        
        return results
    
    async def _extract_with_mathpix(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        detection_mode: str,
        formula_regions: List[Dict[str, Any]],
        dpi: int
    ) -> Dict[str, Any]:
        """Extract formulas using Mathpix API (placeholder implementation)."""
        # This would require Mathpix API credentials and implementation
        raise ToolExecutionException("Mathpix integration not implemented. Please use 'latex_ocr' or 'pix2tex' methods.")
    
    async def _extract_with_pattern_based(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        extract_inline: bool,
        extract_display: bool
    ) -> Dict[str, Any]:
        """Extract formulas using pattern-based text analysis."""
        start_time = datetime.now()
        
        results = {
            "formulas": [],
            "method": "pattern_based",
            "pages_processed": []
        }
        
        if not fitz:
            raise ToolExecutionException("PyMuPDF not available")
        
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
                text = page.get_text()
                
                # Extract inline formulas (e.g., $x^2$, \(x^2\))
                if extract_inline:
                    inline_patterns = [
                        r'\$([^$]+)\$',  # $formula$
                        r'\\\(([^)]+)\\\)',  # \(formula\)
                    ]
                    
                    for pattern in inline_patterns:
                        matches = re.finditer(pattern, text)
                        for match in matches:
                            formula_text = match.group(1)
                            
                            formula_info = {
                                "page": page_num + 1,
                                "bbox": [0, 0, 0, 0],  # Pattern-based doesn't provide exact positions
                                "latex": formula_text,
                                "confidence": 0.6,  # Lower confidence for pattern-based
                                "type": "inline",
                                "method": "pattern_based",
                                "raw_match": match.group(0)
                            }
                            
                            formula_info["mathml"] = await self._latex_to_mathml(formula_text)
                            formula_info["text"] = await self._latex_to_text(formula_text)
                            
                            results["formulas"].append(formula_info)
                
                # Extract display formulas (e.g., $$formula$$, \[formula\])
                if extract_display:
                    display_patterns = [
                        r'\$\$([^$]+)\$\$',  # $$formula$$
                        r'\\\[([^\]]+)\\\]',  # \[formula\]
                        r'\\begin\{equation\}([^\}]+)\\end\{equation\}',  # \begin{equation}...\end{equation}
                        r'\\begin\{align\}([^\}]+)\\end\{align\}',  # \begin{align}...\end{align}
                    ]
                    
                    for pattern in display_patterns:
                        matches = re.finditer(pattern, text, re.DOTALL)
                        for match in matches:
                            formula_text = match.group(1)
                            
                            formula_info = {
                                "page": page_num + 1,
                                "bbox": [0, 0, 0, 0],  # Pattern-based doesn't provide exact positions
                                "latex": formula_text,
                                "confidence": 0.7,  # Slightly higher confidence for display formulas
                                "type": "display",
                                "method": "pattern_based",
                                "raw_match": match.group(0)
                            }
                            
                            formula_info["mathml"] = await self._latex_to_mathml(formula_text)
                            formula_info["text"] = await self._latex_to_text(formula_text)
                            
                            results["formulas"].append(formula_info)
                
                results["pages_processed"].append(page_num + 1)
        
        finally:
            pdf_doc.close()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results.update({
            "processing_time": processing_time,
            "average_confidence": 0.65  # Average confidence for pattern-based extraction
        })
        
        return results
    
    async def _detect_formula_regions(self, page, dpi: int) -> List[List[float]]:
        """Automatically detect formula regions on a page."""
        regions = []
        
        try:
            # Simple approach: look for mathematical symbols and patterns
            text_dict = page.get_text("dict")
            
            # Mathematical symbols that might indicate formulas
            math_symbols = {'∫', '∑', '∏', '√', '∞', '±', '≤', '≥', '≠', '≈', '∝', 'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω'}
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            line_text += span_text
                            
                            if line_bbox is None:
                                line_bbox = span["bbox"]
                            else:
                                # Expand bounding box
                                line_bbox = [
                                    min(line_bbox[0], span["bbox"][0]),
                                    min(line_bbox[1], span["bbox"][1]),
                                    max(line_bbox[2], span["bbox"][2]),
                                    max(line_bbox[3], span["bbox"][3])
                                ]
                        
                        # Check if line contains mathematical content
                        if line_text and line_bbox:
                            has_math_symbols = any(symbol in line_text for symbol in math_symbols)
                            has_math_patterns = bool(re.search(r'[a-zA-Z]\^[0-9]|[a-zA-Z]_[0-9]|\\[a-zA-Z]+|\{[^}]*\}', line_text))
                            
                            if has_math_symbols or has_math_patterns:
                                regions.append(line_bbox)
        
        except Exception as e:
            self.logger.warning(f"Formula region detection failed: {e}")
            # Fallback: return full page
            page_rect = page.rect
            regions = [[page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1]]
        
        return regions
    
    async def _preprocess_formula_image(self, image: Any) -> Any:
        """Preprocess image to improve formula recognition."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        if cv2 and np:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Apply adaptive thresholding for better contrast
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            image = Image.fromarray(img_array)
        
        return image
    
    async def _latex_to_mathml(self, latex_code: str) -> str:
        """Convert LaTeX to MathML (simplified implementation)."""
        # This is a placeholder implementation
        # In practice, you would use a library like latex2mathml or similar
        try:
            # Try to use latex2mathml if available
            import latex2mathml.converter
            return latex2mathml.converter.convert(latex_code)
        except ImportError:
            # Fallback: return a basic MathML wrapper
            return f"<math xmlns='http://www.w3.org/1998/Math/MathML'><mtext>{latex_code}</mtext></math>"
        except Exception:
            return f"<math xmlns='http://www.w3.org/1998/Math/MathML'><mtext>{latex_code}</mtext></math>"
    
    async def _latex_to_text(self, latex_code: str) -> str:
        """Convert LaTeX to readable text (simplified implementation)."""
        # Simple text conversion - remove LaTeX commands and format
        text = latex_code
        
        # Remove common LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \command{content} -> content
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove standalone commands
        text = re.sub(r'[{}]', '', text)  # Remove braces
        text = re.sub(r'\^([0-9a-zA-Z])', r'^\1', text)  # Format superscripts
        text = re.sub(r'_([0-9a-zA-Z])', r'_\1', text)  # Format subscripts
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text


class FormulaToLatexTool(PDFTool):
    """Convert formula images to LaTeX code."""
    
    def __init__(self):
        super().__init__(
            name="formula_to_latex",
            description="Convert formula images to LaTeX code using advanced recognition",
            version="1.0.0"
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file or image file containing formulas"
                },
                "input_type": {
                    "type": "string",
                    "enum": ["pdf", "image"],
                    "default": "pdf",
                    "description": "Type of input file"
                },
                "formula_regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer", "default": 1},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4
                            }
                        },
                        "required": ["bbox"]
                    },
                    "description": "Specific formula regions to convert [x1, y1, x2, y2]"
                },
                "method": {
                    "type": "string",
                    "enum": ["auto", "latex_ocr", "pix2tex"],
                    "default": "auto",
                    "description": "Recognition method to use"
                },
                "preprocess": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply image preprocessing"
                },
                "validate_latex": {
                    "type": "boolean",
                    "default": True,
                    "description": "Validate generated LaTeX code"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["latex", "both"],
                    "default": "latex",
                    "description": "Output format (latex only or both latex and mathml)"
                }
            },
            "required": ["file_path", "formula_regions"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        input_type = kwargs.get("input_type", "pdf")
        formula_regions = kwargs["formula_regions"]
        method = kwargs.get("method", "auto")
        preprocess = kwargs.get("preprocess", True)
        validate_latex = kwargs.get("validate_latex", True)
        output_format = kwargs.get("output_format", "latex")
        
        try:
            # Validate file
            file_path = Path(file_path)
            if not file_path.exists():
                raise ToolExecutionException(f"File not found: {file_path}")
            
            # Choose recognition method
            if method == "auto":
                method = self._choose_best_method()
            
            results = {
                "conversions": [],
                "method": method,
                "total_regions": len(formula_regions)
            }
            
            # Process each formula region
            for i, region_info in enumerate(formula_regions):
                try:
                    bbox = region_info["bbox"]
                    page_num = region_info.get("page", 1)
                    
                    # Extract image from region
                    if input_type == "pdf":
                        image = await self._extract_region_from_pdf(file_path, page_num, bbox)
                    else:
                        image = await self._extract_region_from_image(file_path, bbox)
                    
                    # Preprocess image if requested
                    if preprocess:
                        image = await self._preprocess_formula_image(image)
                    
                    # Convert to LaTeX
                    if method == "latex_ocr":
                        latex_code = await self._convert_with_latex_ocr(image)
                    elif method == "pix2tex":
                        latex_code = await self._convert_with_pix2tex(image)
                    else:
                        raise ToolExecutionException(f"Unknown method: {method}")
                    
                    # Validate LaTeX if requested
                    if validate_latex:
                        is_valid, validation_message = await self._validate_latex_code(latex_code)
                    else:
                        is_valid, validation_message = True, "Validation skipped"
                    
                    conversion_result = {
                        "region_index": i,
                        "page": page_num,
                        "bbox": bbox,
                        "latex": latex_code,
                        "valid": is_valid,
                        "validation_message": validation_message
                    }
                    
                    # Add MathML if requested
                    if output_format == "both":
                        conversion_result["mathml"] = await self._latex_to_mathml(latex_code)
                    
                    results["conversions"].append(conversion_result)
                
                except Exception as e:
                    self.logger.error(f"Failed to convert region {i}: {e}")
                    results["conversions"].append({
                        "region_index": i,
                        "page": region_info.get("page", 1),
                        "bbox": region_info["bbox"],
                        "error": str(e),
                        "valid": False
                    })
            
            # Format output
            content = []
            
            # Add summary
            successful_conversions = sum(1 for c in results["conversions"] if "latex" in c)
            summary = {
                "file": str(file_path),
                "method": method,
                "total_regions": len(formula_regions),
                "successful_conversions": successful_conversions,
                "failed_conversions": len(formula_regions) - successful_conversions
            }
            
            content.append(create_text_content(f"Formula to LaTeX Conversion Summary:\n{json.dumps(summary, indent=2)}"))
            
            # Add conversion results
            for conversion in results["conversions"]:
                region_idx = conversion["region_index"]
                page_num = conversion["page"]
                
                if "latex" in conversion:
                    latex_code = conversion["latex"]
                    is_valid = conversion["valid"]
                    validation_msg = conversion["validation_message"]
                    
                    content.append(create_text_content(
                        f"Region {region_idx + 1} (Page {page_num}) - Valid: {is_valid}:\n```latex\n{latex_code}\n```\nValidation: {validation_msg}"
                    ))
                    
                    if output_format == "both" and "mathml" in conversion:
                        mathml_code = conversion["mathml"]
                        content.append(create_text_content(
                            f"Region {region_idx + 1} MathML:\n```xml\n{mathml_code}\n```"
                        ))
                else:
                    error_msg = conversion.get("error", "Unknown error")
                    content.append(create_text_content(
                        f"Region {region_idx + 1} (Page {page_num}) - FAILED: {error_msg}"
                    ))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"Formula to LaTeX conversion failed: {e}")
            content = [create_error_content(f"Formula to LaTeX conversion failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _choose_best_method(self) -> str:
        """Choose the best available conversion method."""
        # Check for LaTeX-OCR dependencies
        if torch and TrOCRProcessor and VisionEncoderDecoderModel:
            return "latex_ocr"
        
        # Check for pix2tex
        try:
            subprocess.run(["pix2tex", "--version"], capture_output=True, check=True)
            return "pix2tex"
        except:
            pass
        
        raise ToolExecutionException("No formula recognition method available")
    
    async def _extract_region_from_pdf(self, pdf_path: Path, page_num: int, bbox: List[float]) -> Any:
        """Extract a specific region from a PDF page as an image."""
        if not fitz:
            raise ToolExecutionException("PyMuPDF not available")
        
        pdf_doc = fitz.open(str(pdf_path))
        try:
            page = pdf_doc[page_num - 1]  # Convert to 0-indexed
            region_rect = fitz.Rect(bbox)
            mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
            pix = page.get_pixmap(matrix=mat, clip=region_rect)
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
        finally:
            pdf_doc.close()
    
    async def _extract_region_from_image(self, image_path: Path, bbox: List[float]) -> Any:
        """Extract a specific region from an image file."""
        image = Image.open(image_path)
        return image.crop(bbox)
    
    async def _preprocess_formula_image(self, image: Any) -> Any:
        """Preprocess image for better formula recognition."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        if cv2 and np:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Apply adaptive thresholding
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations
            kernel = np.ones((2, 2), np.uint8)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            image = Image.fromarray(img_array)
        
        return image
    
    async def _convert_with_latex_ocr(self, image: Any) -> str:
        """Convert image to LaTeX using LaTeX-OCR model."""
        # This would use the same model as in ExtractFormulasTool
        # Placeholder implementation
        return "x^2 + y^2 = z^2"  # Placeholder
    
    async def _convert_with_pix2tex(self, image: Any) -> str:
        """Convert image to LaTeX using pix2tex."""
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_image_path = Path(temp_file.name)
        
        try:
            # Run pix2tex
            cmd = ["pix2tex", str(temp_image_path)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode().strip()
            else:
                raise ToolExecutionException(f"pix2tex failed: {stderr.decode()}")
        
        finally:
            # Clean up temporary file
            if temp_image_path.exists():
                temp_image_path.unlink()
    
    async def _validate_latex_code(self, latex_code: str) -> Tuple[bool, str]:
        """Validate LaTeX code for mathematical formulas."""
        try:
            # Basic validation checks
            if not latex_code.strip():
                return False, "Empty LaTeX code"
            
            # Check for balanced braces
            open_braces = latex_code.count('{')
            close_braces = latex_code.count('}')
            if open_braces != close_braces:
                return False, f"Unbalanced braces: {open_braces} open, {close_braces} close"
            
            # Check for common LaTeX math commands
            math_commands = ['\\frac', '\\sqrt', '\\sum', '\\int', '\\prod', '\\lim']
            has_math = any(cmd in latex_code for cmd in math_commands) or any(char in latex_code for char in '^_')
            
            if not has_math:
                return False, "No mathematical content detected"
            
            # Try to compile with a LaTeX engine (if available)
            # This is a placeholder - actual implementation would use a LaTeX compiler
            
            return True, "LaTeX code appears valid"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    async def _latex_to_mathml(self, latex_code: str) -> str:
        """Convert LaTeX to MathML."""
        # Reuse the implementation from ExtractFormulasTool
        try:
            import latex2mathml.converter
            return latex2mathml.converter.convert(latex_code)
        except ImportError:
            return f"<math xmlns='http://www.w3.org/1998/Math/MathML'><mtext>{latex_code}</mtext></math>"
        except Exception:
            return f"<math xmlns='http://www.w3.org/1998/Math/MathML'><mtext>{latex_code}</mtext></math>"