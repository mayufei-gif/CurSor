#!/usr/bin/env python3
"""
PDF OCR Extraction Tools

Implements OCR functionality for PDF files using multiple OCR engines
including Tesseract, EasyOCR, and PaddleOCR for text recognition from images and scanned PDFs.

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
    import paddleocr
except ImportError:
    paddleocr = None

try:
    import pdf2image
except ImportError:
    pdf2image = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ..mcp.tools import PDFOCRTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class ExtractTextOCRTool(PDFOCRTool):
    """Extract text from PDF files using OCR technology."""
    
    def __init__(self):
        super().__init__(
            name="extract_text_ocr",
            description="Extract text from PDF files using OCR technology for scanned documents",
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
                "engine": {
                    "type": "string",
                    "enum": ["auto", "tesseract", "easyocr", "paddleocr"],
                    "default": "auto",
                    "description": "OCR engine to use for text extraction"
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["eng"],
                    "description": "Languages for OCR recognition (e.g., ['eng', 'chi_sim'])"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to process (1-indexed). If not provided, processes all pages"
                },
                "dpi": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 150,
                    "maximum": 600,
                    "description": "DPI for PDF to image conversion (higher = better quality, slower)"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence threshold for OCR results"
                },
                "preprocess": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply image preprocessing to improve OCR accuracy"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["text", "json", "detailed"],
                    "default": "text",
                    "description": "Output format for OCR results"
                },
                "preserve_layout": {
                    "type": "boolean",
                    "default": False,
                    "description": "Attempt to preserve text layout and formatting"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute OCR text extraction."""
        try:
            file_path = Path(kwargs["file_path"])
            engine = kwargs.get("engine", "auto")
            languages = kwargs.get("languages", ["eng"])
            pages = kwargs.get("pages")
            dpi = kwargs.get("dpi", 300)
            confidence_threshold = kwargs.get("confidence_threshold", 0.6)
            preprocess = kwargs.get("preprocess", True)
            output_format = kwargs.get("output_format", "text")
            preserve_layout = kwargs.get("preserve_layout", False)
            
            if not file_path.exists():
                raise MCPResourceException(f"PDF file not found: {file_path}")
            
            # Choose OCR engine
            if engine == "auto":
                engine = self._choose_best_engine()
            
            # Extract text using selected engine
            if engine == "tesseract":
                result = await self._extract_with_tesseract(
                    file_path, languages, pages, dpi, confidence_threshold, 
                    preprocess, preserve_layout
                )
            elif engine == "easyocr":
                result = await self._extract_with_easyocr(
                    file_path, languages, pages, dpi, confidence_threshold, preprocess
                )
            elif engine == "paddleocr":
                result = await self._extract_with_paddleocr(
                    file_path, languages, pages, dpi, confidence_threshold, preprocess
                )
            else:
                raise ToolExecutionException(f"Unsupported OCR engine: {engine}")
            
            # Format output
            if output_format == "text":
                content = result["text"]
            elif output_format == "json":
                content = json.dumps(result, indent=2, ensure_ascii=False)
            else:  # detailed
                content = json.dumps({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "pages": result["pages"],
                    "statistics": result["statistics"]
                }, indent=2, ensure_ascii=False)
            
            return MCPToolResult(
                content=[create_text_content(content)],
                isError=False
            )
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return MCPToolResult(
                content=[create_error_content(f"OCR extraction failed: {str(e)}")],
                isError=True
            )
    
    def _choose_best_engine(self) -> str:
        """Choose the best available OCR engine."""
        if pytesseract and Image:
            return "tesseract"
        elif easyocr:
            return "easyocr"
        elif paddleocr:
            return "paddleocr"
        else:
            raise ToolExecutionException("No OCR engine available. Please install pytesseract, easyocr, or paddleocr.")
    
    async def _extract_with_tesseract(
        self,
        pdf_path: Path,
        languages: List[str],
        pages: Optional[List[int]],
        dpi: int,
        confidence_threshold: float,
        preprocess: bool,
        preserve_layout: bool
    ) -> Dict[str, Any]:
        """Extract text using Tesseract OCR."""
        if not pytesseract or not Image:
            raise ToolExecutionException("Tesseract OCR not available. Please install pytesseract and Pillow.")
        
        try:
            # Convert PDF to images
            images = await self._pdf_to_images(pdf_path, pages, dpi)
            
            extracted_text = []
            page_results = []
            total_confidence = 0
            total_words = 0
            
            lang_string = "+".join(languages)
            
            for page_num, image in enumerate(images, 1):
                if preprocess:
                    image = self._preprocess_image(image)
                
                # Extract text with confidence data
                if preserve_layout:
                    config = '--psm 6 -c preserve_interword_spaces=1'
                else:
                    config = '--psm 3'
                
                # Get detailed data
                data = pytesseract.image_to_data(
                    image, lang=lang_string, config=config, output_type=pytesseract.Output.DICT
                )
                
                page_text = []
                page_words = []
                
                for i, word in enumerate(data['text']):
                    if word.strip():
                        confidence = float(data['conf'][i])
                        if confidence >= confidence_threshold * 100:  # Tesseract uses 0-100 scale
                            page_text.append(word)
                            page_words.append({
                                "text": word,
                                "confidence": confidence / 100,
                                "bbox": [
                                    data['left'][i],
                                    data['top'][i],
                                    data['left'][i] + data['width'][i],
                                    data['top'][i] + data['height'][i]
                                ]
                            })
                            total_confidence += confidence
                            total_words += 1
                
                page_text_str = " ".join(page_text)
                extracted_text.append(page_text_str)
                
                page_results.append({
                    "page": page_num,
                    "text": page_text_str,
                    "words": page_words,
                    "word_count": len(page_words)
                })
            
            avg_confidence = (total_confidence / total_words / 100) if total_words > 0 else 0
            
            return {
                "text": "\n\n".join(extracted_text),
                "engine": "tesseract",
                "languages": languages,
                "pages": page_results,
                "metadata": {
                    "total_pages": len(images),
                    "dpi": dpi,
                    "preprocessed": preprocess,
                    "preserve_layout": preserve_layout
                },
                "statistics": {
                    "total_words": total_words,
                    "average_confidence": avg_confidence,
                    "confidence_threshold": confidence_threshold
                }
            }
            
        except Exception as e:
            raise ToolExecutionException(f"Tesseract OCR failed: {str(e)}")
    
    async def _extract_with_easyocr(
        self,
        pdf_path: Path,
        languages: List[str],
        pages: Optional[List[int]],
        dpi: int,
        confidence_threshold: float,
        preprocess: bool
    ) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        if not easyocr:
            raise ToolExecutionException("EasyOCR not available. Please install easyocr.")
        
        try:
            # Convert PDF to images
            images = await self._pdf_to_images(pdf_path, pages, dpi)
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(languages, gpu=False)  # Set gpu=True if CUDA available
            
            extracted_text = []
            page_results = []
            total_confidence = 0
            total_words = 0
            
            for page_num, image in enumerate(images, 1):
                if preprocess:
                    image = self._preprocess_image(image)
                
                # Convert PIL image to numpy array
                import numpy as np
                image_array = np.array(image)
                
                # Extract text
                results = reader.readtext(image_array)
                
                page_text = []
                page_words = []
                
                for bbox, text, confidence in results:
                    if confidence >= confidence_threshold:
                        page_text.append(text)
                        page_words.append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": [
                                min(point[0] for point in bbox),
                                min(point[1] for point in bbox),
                                max(point[0] for point in bbox),
                                max(point[1] for point in bbox)
                            ]
                        })
                        total_confidence += confidence
                        total_words += 1
                
                page_text_str = " ".join(page_text)
                extracted_text.append(page_text_str)
                
                page_results.append({
                    "page": page_num,
                    "text": page_text_str,
                    "words": page_words,
                    "word_count": len(page_words)
                })
            
            avg_confidence = (total_confidence / total_words) if total_words > 0 else 0
            
            return {
                "text": "\n\n".join(extracted_text),
                "engine": "easyocr",
                "languages": languages,
                "pages": page_results,
                "metadata": {
                    "total_pages": len(images),
                    "dpi": dpi,
                    "preprocessed": preprocess
                },
                "statistics": {
                    "total_words": total_words,
                    "average_confidence": avg_confidence,
                    "confidence_threshold": confidence_threshold
                }
            }
            
        except Exception as e:
            raise ToolExecutionException(f"EasyOCR failed: {str(e)}")
    
    async def _extract_with_paddleocr(
        self,
        pdf_path: Path,
        languages: List[str],
        pages: Optional[List[int]],
        dpi: int,
        confidence_threshold: float,
        preprocess: bool
    ) -> Dict[str, Any]:
        """Extract text using PaddleOCR."""
        if not paddleocr:
            raise ToolExecutionException("PaddleOCR not available. Please install paddlepaddle and paddleocr.")
        
        try:
            # Convert PDF to images
            images = await self._pdf_to_images(pdf_path, pages, dpi)
            
            # Initialize PaddleOCR
            lang_map = {"eng": "en", "chi_sim": "ch", "chi_tra": "chinese_cht"}
            paddle_langs = [lang_map.get(lang, lang) for lang in languages]
            
            ocr = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang=paddle_langs[0] if paddle_langs else "en",
                use_gpu=False
            )
            
            extracted_text = []
            page_results = []
            total_confidence = 0
            total_words = 0
            
            for page_num, image in enumerate(images, 1):
                if preprocess:
                    image = self._preprocess_image(image)
                
                # Convert PIL image to numpy array
                import numpy as np
                image_array = np.array(image)
                
                # Extract text
                results = ocr.ocr(image_array, cls=True)
                
                page_text = []
                page_words = []
                
                if results and results[0]:
                    for line in results[0]:
                        if line:
                            bbox, (text, confidence) = line
                            if confidence >= confidence_threshold:
                                page_text.append(text)
                                page_words.append({
                                    "text": text,
                                    "confidence": confidence,
                                    "bbox": [
                                        min(point[0] for point in bbox),
                                        min(point[1] for point in bbox),
                                        max(point[0] for point in bbox),
                                        max(point[1] for point in bbox)
                                    ]
                                })
                                total_confidence += confidence
                                total_words += 1
                
                page_text_str = " ".join(page_text)
                extracted_text.append(page_text_str)
                
                page_results.append({
                    "page": page_num,
                    "text": page_text_str,
                    "words": page_words,
                    "word_count": len(page_words)
                })
            
            avg_confidence = (total_confidence / total_words) if total_words > 0 else 0
            
            return {
                "text": "\n\n".join(extracted_text),
                "engine": "paddleocr",
                "languages": languages,
                "pages": page_results,
                "metadata": {
                    "total_pages": len(images),
                    "dpi": dpi,
                    "preprocessed": preprocess
                },
                "statistics": {
                    "total_words": total_words,
                    "average_confidence": avg_confidence,
                    "confidence_threshold": confidence_threshold
                }
            }
            
        except Exception as e:
            raise ToolExecutionException(f"PaddleOCR failed: {str(e)}")
    
    async def _pdf_to_images(self, pdf_path: Path, pages: Optional[List[int]], dpi: int) -> List[Any]:
        """Convert PDF pages to images."""
        if pdf2image:
            # Use pdf2image if available
            if pages:
                images = pdf2image.convert_from_path(
                    pdf_path, dpi=dpi, first_page=min(pages), last_page=max(pages)
                )
                # Filter to requested pages
                page_indices = [p - min(pages) for p in pages]
                images = [images[i] for i in page_indices if i < len(images)]
            else:
                images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        elif fitz:
            # Use PyMuPDF as fallback
            doc = fitz.open(pdf_path)
            images = []
            
            page_range = pages if pages else range(1, doc.page_count + 1)
            
            for page_num in page_range:
                page = doc[page_num - 1]  # Convert to 0-indexed
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                
                from io import BytesIO
                image = Image.open(BytesIO(img_data))
                images.append(image)
            
            doc.close()
        else:
            raise ToolExecutionException("No PDF to image conversion library available. Please install pdf2image or PyMuPDF.")
        
        return images
    
    def _preprocess_image(self, image: Any) -> Any:
        """Apply image preprocessing to improve OCR accuracy."""
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {str(e)}")
            return image


class ExtractTextOCRAdvancedTool(PDFOCRTool):
    """Advanced OCR tool with multiple engine comparison and quality assessment."""
    
    def __init__(self):
        super().__init__(
            name="extract_text_ocr_advanced",
            description="Advanced OCR extraction with multiple engines and quality assessment",
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
                "engines": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["tesseract", "easyocr", "paddleocr"]
                    },
                    "default": ["tesseract", "easyocr"],
                    "description": "OCR engines to compare"
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
                "quality_assessment": {
                    "type": "boolean",
                    "default": True,
                    "description": "Perform quality assessment and choose best result"
                },
                "save_comparison": {
                    "type": "boolean",
                    "default": False,
                    "description": "Save detailed comparison results"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute advanced OCR with multiple engines."""
        try:
            file_path = Path(kwargs["file_path"])
            engines = kwargs.get("engines", ["tesseract", "easyocr"])
            languages = kwargs.get("languages", ["eng"])
            pages = kwargs.get("pages")
            quality_assessment = kwargs.get("quality_assessment", True)
            save_comparison = kwargs.get("save_comparison", False)
            
            if not file_path.exists():
                raise MCPResourceException(f"PDF file not found: {file_path}")
            
            # Run OCR with each engine
            basic_tool = ExtractTextOCRTool()
            results = {}
            
            for engine in engines:
                try:
                    result = await basic_tool.execute(
                        file_path=str(file_path),
                        engine=engine,
                        languages=languages,
                        pages=pages,
                        output_format="detailed"
                    )
                    
                    if not result.isError:
                        results[engine] = json.loads(result.content[0].text)
                    else:
                        self.logger.warning(f"Engine {engine} failed: {result.content[0].text}")
                        
                except Exception as e:
                    self.logger.warning(f"Engine {engine} failed: {str(e)}")
            
            if not results:
                raise ToolExecutionException("All OCR engines failed")
            
            # Quality assessment and best result selection
            if quality_assessment and len(results) > 1:
                best_result = self._select_best_result(results)
                comparison = self._compare_results(results)
            else:
                best_result = list(results.values())[0]
                comparison = None
            
            # Prepare output
            output = {
                "text": best_result["text"],
                "best_engine": best_result["engine"],
                "metadata": best_result["metadata"],
                "statistics": best_result["statistics"]
            }
            
            if comparison:
                output["comparison"] = comparison
            
            if save_comparison and comparison:
                comparison_file = file_path.parent / f"{file_path.stem}_ocr_comparison.json"
                with open(comparison_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "results": results,
                        "comparison": comparison,
                        "best_engine": best_result["engine"]
                    }, f, indent=2, ensure_ascii=False)
                output["comparison_file"] = str(comparison_file)
            
            return MCPToolResult(
                content=[create_text_content(json.dumps(output, indent=2, ensure_ascii=False))],
                isError=False
            )
            
        except Exception as e:
            self.logger.error(f"Advanced OCR extraction failed: {str(e)}")
            return MCPToolResult(
                content=[create_error_content(f"Advanced OCR extraction failed: {str(e)}")],
                isError=True
            )
    
    def _select_best_result(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best OCR result based on quality metrics."""
        scores = {}
        
        for engine, result in results.items():
            score = 0
            stats = result.get("statistics", {})
            
            # Confidence score (40% weight)
            avg_confidence = stats.get("average_confidence", 0)
            score += avg_confidence * 0.4
            
            # Word count (30% weight) - more words usually better
            total_words = stats.get("total_words", 0)
            max_words = max(r.get("statistics", {}).get("total_words", 0) for r in results.values())
            if max_words > 0:
                score += (total_words / max_words) * 0.3
            
            # Text length (20% weight)
            text_length = len(result.get("text", ""))
            max_length = max(len(r.get("text", "")) for r in results.values())
            if max_length > 0:
                score += (text_length / max_length) * 0.2
            
            # Engine reliability (10% weight)
            engine_weights = {"tesseract": 0.9, "easyocr": 0.8, "paddleocr": 0.7}
            score += engine_weights.get(engine, 0.5) * 0.1
            
            scores[engine] = score
        
        best_engine = max(scores, key=scores.get)
        return results[best_engine]
    
    def _compare_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare OCR results from different engines."""
        comparison = {
            "engines_compared": list(results.keys()),
            "metrics": {},
            "text_similarity": {},
            "recommendations": []
        }
        
        # Extract metrics for each engine
        for engine, result in results.items():
            stats = result.get("statistics", {})
            comparison["metrics"][engine] = {
                "confidence": stats.get("average_confidence", 0),
                "word_count": stats.get("total_words", 0),
                "text_length": len(result.get("text", "")),
                "pages_processed": result.get("metadata", {}).get("total_pages", 0)
            }
        
        # Calculate text similarity between engines
        engines = list(results.keys())
        for i, engine1 in enumerate(engines):
            for engine2 in engines[i+1:]:
                similarity = self._calculate_text_similarity(
                    results[engine1]["text"],
                    results[engine2]["text"]
                )
                comparison["text_similarity"][f"{engine1}_vs_{engine2}"] = similarity
        
        # Generate recommendations
        if len(results) > 1:
            confidences = [r.get("statistics", {}).get("average_confidence", 0) for r in results.values()]
            if max(confidences) - min(confidences) > 0.2:
                comparison["recommendations"].append(
                    "Significant confidence difference between engines - consider using the highest confidence result"
                )
            
            similarities = list(comparison["text_similarity"].values())
            if similarities and min(similarities) < 0.7:
                comparison["recommendations"].append(
                    "Low similarity between engine results - manual review recommended"
                )
        
        return comparison
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0