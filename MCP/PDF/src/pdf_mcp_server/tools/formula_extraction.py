#!/usr/bin/env python3
"""
PDF Formula Extraction Tools

Implements mathematical formula recognition and extraction from PDF files
using LaTeX-OCR, pix2tex, and other formula recognition engines.

Author: PDF-MCP Team
License: MIT
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import tempfile
import os
import re

try:
    from pix2tex.cli import LatexOCR
except ImportError:
    LatexOCR = None

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdf2image
except ImportError:
    pdf2image = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class ExtractFormulasTool(PDFTool):
    """Extract mathematical formulas from PDF files."""
    
    def __init__(self):
        super().__init__(
            name="extract_formulas",
            description="Extract mathematical formulas from PDF files and convert to LaTeX",
            version="1.0.0"
        )
        self.logger = logging.getLogger(__name__)
        self._model = None
    
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
                    "enum": ["auto", "pix2tex", "latex_ocr"],
                    "default": "auto",
                    "description": "Formula recognition engine to use"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to process (1-indexed). If not provided, processes all pages"
                },
                "detection_method": {
                    "type": "string",
                    "enum": ["auto", "bbox", "contour", "text_analysis"],
                    "default": "auto",
                    "description": "Method for detecting formula regions"
                },
                "min_formula_size": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 20,
                    "description": "Minimum size (pixels) for formula detection"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence threshold for formula recognition"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["latex", "mathml", "json", "detailed"],
                    "default": "latex",
                    "description": "Output format for extracted formulas"
                },
                "include_images": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include formula images in the output"
                },
                "dpi": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 150,
                    "maximum": 600,
                    "description": "DPI for PDF to image conversion"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute formula extraction."""
        try:
            file_path = Path(kwargs["file_path"])
            engine = kwargs.get("engine", "auto")
            pages = kwargs.get("pages")
            detection_method = kwargs.get("detection_method", "auto")
            min_formula_size = kwargs.get("min_formula_size", 50)
            confidence_threshold = kwargs.get("confidence_threshold", 0.7)
            output_format = kwargs.get("output_format", "latex")
            include_images = kwargs.get("include_images", False)
            dpi = kwargs.get("dpi", 300)
            
            if not file_path.exists():
                raise MCPResourceException(f"PDF file not found: {file_path}")
            
            # Choose engine
            if engine == "auto":
                engine = self._choose_best_engine()
            
            # Initialize model
            await self._initialize_model(engine)
            
            # Convert PDF to images
            images = await self._pdf_to_images(file_path, pages, dpi)
            
            # Extract formulas
            all_formulas = []
            page_results = []
            
            for page_num, image in enumerate(images, 1):
                page_formulas = await self._extract_formulas_from_image(
                    image, page_num, detection_method, min_formula_size,
                    confidence_threshold, engine, include_images
                )
                
                all_formulas.extend(page_formulas)
                page_results.append({
                    "page": page_num,
                    "formulas": page_formulas,
                    "formula_count": len(page_formulas)
                })
            
            # Format output
            result = {
                "formulas": all_formulas,
                "pages": page_results,
                "metadata": {
                    "total_pages": len(images),
                    "total_formulas": len(all_formulas),
                    "engine": engine,
                    "detection_method": detection_method,
                    "dpi": dpi
                },
                "statistics": {
                    "formulas_per_page": len(all_formulas) / len(images) if images else 0,
                    "confidence_threshold": confidence_threshold,
                    "min_formula_size": min_formula_size
                }
            }
            
            if output_format == "latex":
                content = "\n\n".join([f["latex"] for f in all_formulas if f.get("latex")])
            elif output_format == "mathml":
                content = "\n\n".join([f["mathml"] for f in all_formulas if f.get("mathml")])
            elif output_format == "json":
                content = json.dumps([f for f in all_formulas], indent=2, ensure_ascii=False)
            else:  # detailed
                content = json.dumps(result, indent=2, ensure_ascii=False)
            
            return MCPToolResult(
                content=[create_text_content(content)],
                isError=False
            )
            
        except Exception as e:
            self.logger.error(f"Formula extraction failed: {str(e)}")
            return MCPToolResult(
                content=[create_error_content(f"Formula extraction failed: {str(e)}")],
                isError=True
            )
    
    def _choose_best_engine(self) -> str:
        """Choose the best available formula recognition engine."""
        if LatexOCR:
            return "pix2tex"
        else:
            raise ToolExecutionException("No formula recognition engine available. Please install pix2tex.")
    
    async def _initialize_model(self, engine: str):
        """Initialize the formula recognition model."""
        if engine == "pix2tex" and not self._model:
            if not LatexOCR:
                raise ToolExecutionException("pix2tex not available. Please install pix2tex.")
            
            try:
                self._model = LatexOCR()
                self.logger.info("pix2tex model initialized")
            except Exception as e:
                raise ToolExecutionException(f"Failed to initialize pix2tex model: {str(e)}")
    
    async def _pdf_to_images(self, pdf_path: Path, pages: Optional[List[int]], dpi: int) -> List[Any]:
        """Convert PDF pages to images."""
        if pdf2image:
            if pages:
                images = pdf2image.convert_from_path(
                    pdf_path, dpi=dpi, first_page=min(pages), last_page=max(pages)
                )
                page_indices = [p - min(pages) for p in pages]
                images = [images[i] for i in page_indices if i < len(images)]
            else:
                images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        elif fitz:
            doc = fitz.open(pdf_path)
            images = []
            
            page_range = pages if pages else range(1, doc.page_count + 1)
            
            for page_num in page_range:
                page = doc[page_num - 1]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                
                from io import BytesIO
                image = Image.open(BytesIO(img_data))
                images.append(image)
            
            doc.close()
        else:
            raise ToolExecutionException("No PDF to image conversion library available.")
        
        return images
    
    async def _extract_formulas_from_image(
        self,
        image: Any,
        page_num: int,
        detection_method: str,
        min_formula_size: int,
        confidence_threshold: float,
        engine: str,
        include_images: bool
    ) -> List[Dict[str, Any]]:
        """Extract formulas from a single image."""
        # Detect formula regions
        if detection_method == "auto":
            detection_method = "contour" if cv2 and np else "bbox"
        
        formula_regions = await self._detect_formula_regions(
            image, detection_method, min_formula_size
        )
        
        formulas = []
        
        for i, region in enumerate(formula_regions):
            try:
                # Extract formula image
                x1, y1, x2, y2 = region["bbox"]
                formula_image = image.crop((x1, y1, x2, y2))
                
                # Recognize formula
                if engine == "pix2tex":
                    latex = await self._recognize_with_pix2tex(formula_image)
                else:
                    latex = ""
                
                if latex and len(latex.strip()) > 0:
                    formula_data = {
                        "id": f"formula_{page_num}_{i+1}",
                        "page": page_num,
                        "bbox": region["bbox"],
                        "latex": latex.strip(),
                        "confidence": region.get("confidence", 1.0)
                    }
                    
                    # Add MathML if requested
                    if "mathml" in ["mathml", "detailed"]:
                        formula_data["mathml"] = self._latex_to_mathml(latex)
                    
                    # Add image if requested
                    if include_images:
                        formula_data["image_base64"] = self._image_to_base64(formula_image)
                    
                    formulas.append(formula_data)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process formula region {i}: {str(e)}")
        
        return formulas
    
    async def _detect_formula_regions(
        self,
        image: Any,
        method: str,
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Detect regions that likely contain mathematical formulas."""
        if method == "contour" and cv2 and np:
            return await self._detect_with_contours(image, min_size)
        elif method == "bbox":
            return await self._detect_with_bbox(image, min_size)
        elif method == "text_analysis":
            return await self._detect_with_text_analysis(image, min_size)
        else:
            # Fallback to simple grid-based detection
            return await self._detect_with_grid(image, min_size)
    
    async def _detect_with_contours(self, image: Any, min_size: int) -> List[Dict[str, Any]]:
        """Detect formula regions using contour analysis."""
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w >= min_size and h >= min_size:
                # Check if region looks like a formula (has mathematical symbols)
                region_image = image.crop((x, y, x + w, y + h))
                if self._looks_like_formula(region_image):
                    regions.append({
                        "bbox": [x, y, x + w, y + h],
                        "confidence": 0.8,
                        "area": w * h
                    })
        
        # Sort by area (larger formulas first)
        regions.sort(key=lambda r: r["area"], reverse=True)
        return regions
    
    async def _detect_with_bbox(self, image: Any, min_size: int) -> List[Dict[str, Any]]:
        """Detect formula regions using bounding box analysis."""
        # Simple grid-based approach
        width, height = image.size
        regions = []
        
        # Divide image into overlapping regions
        step_x = width // 4
        step_y = height // 4
        
        for y in range(0, height - min_size, step_y):
            for x in range(0, width - min_size, step_x):
                region_width = min(width - x, min_size * 2)
                region_height = min(height - y, min_size * 2)
                
                if region_width >= min_size and region_height >= min_size:
                    region_image = image.crop((x, y, x + region_width, y + region_height))
                    
                    if self._looks_like_formula(region_image):
                        regions.append({
                            "bbox": [x, y, x + region_width, y + region_height],
                            "confidence": 0.6,
                            "area": region_width * region_height
                        })
        
        return regions
    
    async def _detect_with_text_analysis(self, image: Any, min_size: int) -> List[Dict[str, Any]]:
        """Detect formula regions using text pattern analysis."""
        # This would require OCR to analyze text patterns
        # For now, fallback to grid detection
        return await self._detect_with_grid(image, min_size)
    
    async def _detect_with_grid(self, image: Any, min_size: int) -> List[Dict[str, Any]]:
        """Simple grid-based formula detection."""
        width, height = image.size
        regions = []
        
        # Create a 3x3 grid
        for i in range(3):
            for j in range(3):
                x = (width * j) // 3
                y = (height * i) // 3
                w = width // 3
                h = height // 3
                
                if w >= min_size and h >= min_size:
                    regions.append({
                        "bbox": [x, y, x + w, y + h],
                        "confidence": 0.5,
                        "area": w * h
                    })
        
        return regions
    
    def _looks_like_formula(self, image: Any) -> bool:
        """Heuristic to determine if an image region contains a formula."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Calculate some basic statistics
        pixels = list(image.getdata())
        
        # Check for sufficient contrast (formulas usually have black text on white)
        min_pixel = min(pixels)
        max_pixel = max(pixels)
        contrast = max_pixel - min_pixel
        
        if contrast < 50:  # Low contrast, probably not text
            return False
        
        # Check for reasonable amount of non-white pixels
        non_white_pixels = sum(1 for p in pixels if p < 240)
        total_pixels = len(pixels)
        
        if total_pixels == 0:
            return False
        
        non_white_ratio = non_white_pixels / total_pixels
        
        # Formulas typically have 5-30% non-white pixels
        return 0.05 <= non_white_ratio <= 0.3
    
    async def _recognize_with_pix2tex(self, image: Any) -> str:
        """Recognize formula using pix2tex."""
        try:
            if not self._model:
                raise ToolExecutionException("pix2tex model not initialized")
            
            # Convert PIL image to format expected by pix2tex
            latex = self._model(image)
            return latex if latex else ""
            
        except Exception as e:
            self.logger.warning(f"pix2tex recognition failed: {str(e)}")
            return ""
    
    def _latex_to_mathml(self, latex: str) -> str:
        """Convert LaTeX to MathML (basic implementation)."""
        # This is a very basic conversion - in practice, you'd use a proper library
        # like latex2mathml or similar
        
        # Basic substitutions
        mathml = latex
        
        # Replace common LaTeX commands with MathML equivalents
        replacements = {
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'<mfrac><mrow>\1</mrow><mrow>\2</mrow></mfrac>',
            r'\\sqrt\{([^}]+)\}': r'<msqrt><mrow>\1</mrow></msqrt>',
            r'\\sum': '<mo>∑</mo>',
            r'\\int': '<mo>∫</mo>',
            r'\\alpha': '<mi>α</mi>',
            r'\\beta': '<mi>β</mi>',
            r'\\gamma': '<mi>γ</mi>',
            r'\\pi': '<mi>π</mi>',
        }
        
        for pattern, replacement in replacements.items():
            mathml = re.sub(pattern, replacement, mathml)
        
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML">{mathml}</math>'
    
    def _image_to_base64(self, image: Any) -> str:
        """Convert PIL image to base64 string."""
        from io import BytesIO
        import base64
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str


class ExtractFormulasAdvancedTool(PDFTool):
    """Advanced formula extraction with multiple engines and post-processing."""
    
    def __init__(self):
        super().__init__(
            name="extract_formulas_advanced",
            description="Advanced formula extraction with multiple engines and validation",
            version="1.0.0"
        )
        self.logger = logging.getLogger(__name__)
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "validate_latex": {
                    "type": "boolean",
                    "default": True,
                    "description": "Validate extracted LaTeX formulas"
                },
                "merge_similar": {
                    "type": "boolean",
                    "default": True,
                    "description": "Merge similar formulas to reduce duplicates"
                },
                "extract_inline": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract inline formulas in addition to display formulas"
                },
                "save_images": {
                    "type": "boolean",
                    "default": False,
                    "description": "Save formula images to files"
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory to save formula images (if save_images is true)"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute advanced formula extraction."""
        try:
            file_path = Path(kwargs["file_path"])
            validate_latex = kwargs.get("validate_latex", True)
            merge_similar = kwargs.get("merge_similar", True)
            extract_inline = kwargs.get("extract_inline", True)
            save_images = kwargs.get("save_images", False)
            output_dir = kwargs.get("output_dir")
            
            if not file_path.exists():
                raise MCPResourceException(f"PDF file not found: {file_path}")
            
            # Extract formulas using basic tool
            basic_tool = ExtractFormulasTool()
            result = await basic_tool.execute(
                file_path=str(file_path),
                output_format="detailed",
                include_images=save_images
            )
            
            if result.isError:
                return result
            
            data = json.loads(result.content[0].text)
            formulas = data["formulas"]
            
            # Post-process formulas
            if validate_latex:
                formulas = self._validate_formulas(formulas)
            
            if merge_similar:
                formulas = self._merge_similar_formulas(formulas)
            
            if extract_inline:
                inline_formulas = await self._extract_inline_formulas(file_path)
                formulas.extend(inline_formulas)
            
            # Save images if requested
            if save_images and output_dir:
                await self._save_formula_images(formulas, output_dir)
            
            # Generate statistics
            stats = self._generate_statistics(formulas)
            
            result_data = {
                "formulas": formulas,
                "statistics": stats,
                "metadata": data["metadata"],
                "processing": {
                    "validated": validate_latex,
                    "merged_similar": merge_similar,
                    "extracted_inline": extract_inline,
                    "saved_images": save_images
                }
            }
            
            return MCPToolResult(
                content=[create_text_content(json.dumps(result_data, indent=2, ensure_ascii=False))],
                isError=False
            )
            
        except Exception as e:
            self.logger.error(f"Advanced formula extraction failed: {str(e)}")
            return MCPToolResult(
                content=[create_error_content(f"Advanced formula extraction failed: {str(e)}")],
                isError=True
            )
    
    def _validate_formulas(self, formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate LaTeX formulas and mark invalid ones."""
        valid_formulas = []
        
        for formula in formulas:
            latex = formula.get("latex", "")
            
            # Basic LaTeX validation
            is_valid = self._is_valid_latex(latex)
            formula["valid"] = is_valid
            
            if is_valid:
                valid_formulas.append(formula)
            else:
                self.logger.warning(f"Invalid LaTeX formula: {latex}")
        
        return valid_formulas
    
    def _is_valid_latex(self, latex: str) -> bool:
        """Basic LaTeX validation."""
        if not latex or len(latex.strip()) == 0:
            return False
        
        # Check for balanced braces
        brace_count = 0
        for char in latex:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count < 0:
                    return False
        
        if brace_count != 0:
            return False
        
        # Check for common LaTeX patterns
        if re.search(r'[a-zA-Z0-9+\-*/=<>()\[\]{}\\^_]', latex):
            return True
        
        return False
    
    def _merge_similar_formulas(self, formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar formulas to reduce duplicates."""
        if len(formulas) <= 1:
            return formulas
        
        merged = []
        used_indices = set()
        
        for i, formula1 in enumerate(formulas):
            if i in used_indices:
                continue
            
            similar_group = [formula1]
            used_indices.add(i)
            
            for j, formula2 in enumerate(formulas[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._are_similar_formulas(formula1, formula2):
                    similar_group.append(formula2)
                    used_indices.add(j)
            
            # Merge the group
            if len(similar_group) == 1:
                merged.append(formula1)
            else:
                merged_formula = self._merge_formula_group(similar_group)
                merged.append(merged_formula)
        
        return merged
    
    def _are_similar_formulas(self, formula1: Dict[str, Any], formula2: Dict[str, Any]) -> bool:
        """Check if two formulas are similar enough to merge."""
        latex1 = formula1.get("latex", "")
        latex2 = formula2.get("latex", "")
        
        # Exact match
        if latex1 == latex2:
            return True
        
        # Normalize and compare
        norm1 = self._normalize_latex(latex1)
        norm2 = self._normalize_latex(latex2)
        
        if norm1 == norm2:
            return True
        
        # Check similarity ratio
        similarity = self._calculate_latex_similarity(norm1, norm2)
        return similarity > 0.9
    
    def _normalize_latex(self, latex: str) -> str:
        """Normalize LaTeX for comparison."""
        # Remove extra spaces
        latex = re.sub(r'\s+', ' ', latex.strip())
        
        # Normalize common variations
        latex = latex.replace('\\left(', '(')
        latex = latex.replace('\\right)', ')')
        latex = latex.replace('\\left[', '[')
        latex = latex.replace('\\right]', ']')
        
        return latex
    
    def _calculate_latex_similarity(self, latex1: str, latex2: str) -> float:
        """Calculate similarity between two LaTeX strings."""
        if not latex1 or not latex2:
            return 0.0
        
        # Simple character-based similarity
        longer = max(len(latex1), len(latex2))
        if longer == 0:
            return 1.0
        
        # Calculate edit distance (simplified)
        matches = sum(c1 == c2 for c1, c2 in zip(latex1, latex2))
        return matches / longer
    
    def _merge_formula_group(self, formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of similar formulas."""
        # Use the formula with highest confidence
        best_formula = max(formulas, key=lambda f: f.get("confidence", 0))
        
        # Collect all occurrences
        occurrences = []
        for formula in formulas:
            occurrences.append({
                "page": formula["page"],
                "bbox": formula["bbox"],
                "confidence": formula.get("confidence", 0)
            })
        
        merged = best_formula.copy()
        merged["occurrences"] = occurrences
        merged["occurrence_count"] = len(occurrences)
        
        return merged
    
    async def _extract_inline_formulas(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract inline formulas from text."""
        # This would require text extraction and pattern matching
        # For now, return empty list
        return []
    
    async def _save_formula_images(self, formulas: List[Dict[str, Any]], output_dir: str):
        """Save formula images to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for formula in formulas:
            if "image_base64" in formula:
                import base64
                
                image_data = base64.b64decode(formula["image_base64"])
                image_file = output_path / f"{formula['id']}.png"
                
                with open(image_file, 'wb') as f:
                    f.write(image_data)
                
                formula["image_file"] = str(image_file)
    
    def _generate_statistics(self, formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about extracted formulas."""
        if not formulas:
            return {"total_formulas": 0}
        
        # Basic statistics
        total = len(formulas)
        pages_with_formulas = len(set(f["page"] for f in formulas))
        avg_confidence = sum(f.get("confidence", 0) for f in formulas) / total
        
        # Formula complexity (based on LaTeX length)
        latex_lengths = [len(f.get("latex", "")) for f in formulas]
        avg_complexity = sum(latex_lengths) / total if latex_lengths else 0
        
        # Most common formula types (simplified)
        formula_types = {}
        for formula in formulas:
            latex = formula.get("latex", "")
            if "frac" in latex:
                formula_types["fraction"] = formula_types.get("fraction", 0) + 1
            if "sum" in latex or "∑" in latex:
                formula_types["summation"] = formula_types.get("summation", 0) + 1
            if "int" in latex or "∫" in latex:
                formula_types["integral"] = formula_types.get("integral", 0) + 1
            if "sqrt" in latex:
                formula_types["square_root"] = formula_types.get("square_root", 0) + 1
        
        return {
            "total_formulas": total,
            "pages_with_formulas": pages_with_formulas,
            "average_confidence": avg_confidence,
            "average_complexity": avg_complexity,
            "formula_types": formula_types
        }