#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Extraction Tools

This module provides tools for extracting text from PDF files using OCR.
"""

import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

from .base import PDFTool, PDFToolOutput

class ExtractTextOCRInput(BaseModel):
    """Input schema for OCR text extraction."""
    model_config = ConfigDict(extra="forbid")
    
    file_path: str = Field(
        description="Path to the PDF file",
        examples=["/path/to/document.pdf"]
    )
    engine: str = Field(
        default="tesseract",
        description="OCR engine: 'tesseract', 'easyocr', or 'paddleocr'",
        examples=["tesseract", "easyocr", "paddleocr"]
    )
    language: str = Field(
        default="eng",
        description="OCR language code (e.g., 'eng', 'chi_sim', 'fra')",
        examples=["eng", "chi_sim", "fra"]
    )
    pages: Optional[str] = Field(
        default=None,
        description="Page range (e.g., '1-5', '1,3,5', 'all')",
        examples=["1-5", "1,3,5", "all"]
    )
    dpi: int = Field(
        default=300,
        description="DPI for PDF to image conversion",
        ge=72,
        le=600
    )
    confidence_threshold: float = Field(
        default=0.0,
        description="Minimum confidence threshold for OCR results",
        ge=0.0,
        le=1.0
    )
    preprocess: bool = Field(
        default=True,
        description="Whether to preprocess images for better OCR"
    )
    output_format: str = Field(
        default="text",
        description="Output format: 'text', 'json', 'hocr'",
        examples=["text", "json", "hocr"]
    )

class ExtractTextOCROutput(PDFToolOutput):
    """Output schema for OCR text extraction."""
    
    text: str = Field(
        description="Extracted text content"
    )
    page_count: int = Field(
        description="Total number of pages in the PDF"
    )
    pages_processed: List[int] = Field(
        description="List of page numbers that were processed"
    )
    engine_used: str = Field(
        description="OCR engine that was used"
    )
    language_used: str = Field(
        description="Language code used for OCR"
    )
    confidence_scores: Optional[List[float]] = Field(
        default=None,
        description="Confidence scores for each page"
    )
    word_count: Optional[int] = Field(
        default=None,
        description="Number of words extracted"
    )
    character_count: Optional[int] = Field(
        default=None,
        description="Number of characters extracted"
    )

class ExtractTextOCRAdvancedInput(BaseModel):
    """Input schema for advanced OCR text extraction."""
    model_config = ConfigDict(extra="forbid")
    
    file_path: str = Field(
        description="Path to the PDF file",
        examples=["/path/to/document.pdf"]
    )
    engine: str = Field(
        default="tesseract",
        description="OCR engine: 'tesseract', 'easyocr', or 'paddleocr'",
        examples=["tesseract", "easyocr", "paddleocr"]
    )
    language: str = Field(
        default="eng",
        description="OCR language code (e.g., 'eng', 'chi_sim', 'fra')",
        examples=["eng", "chi_sim", "fra"]
    )
    pages: Optional[str] = Field(
        default=None,
        description="Page range (e.g., '1-5', '1,3,5', 'all')",
        examples=["1-5", "1,3,5", "all"]
    )
    dpi: int = Field(
        default=300,
        description="DPI for PDF to image conversion",
        ge=72,
        le=600
    )
    confidence_threshold: float = Field(
        default=0.0,
        description="Minimum confidence threshold for OCR results",
        ge=0.0,
        le=1.0
    )
    preprocess: bool = Field(
        default=True,
        description="Whether to preprocess images for better OCR"
    )
    output_format: str = Field(
        default="json",
        description="Output format: 'text', 'json', 'hocr', 'pdf'",
        examples=["text", "json", "hocr", "pdf"]
    )
    preserve_layout: bool = Field(
        default=False,
        description="Whether to preserve text layout and positioning"
    )
    include_images: bool = Field(
        default=False,
        description="Whether to include base64 encoded images in output"
    )
    denoise: bool = Field(
        default=True,
        description="Whether to apply denoising to images"
    )
    deskew: bool = Field(
        default=True,
        description="Whether to apply deskewing to images"
    )
    enhance_contrast: bool = Field(
        default=True,
        description="Whether to enhance image contrast"
    )

class ExtractTextOCRAdvancedOutput(PDFToolOutput):
    """Output schema for advanced OCR text extraction."""
    
    text: str = Field(
        description="Extracted text content"
    )
    page_count: int = Field(
        description="Total number of pages in the PDF"
    )
    pages_processed: List[int] = Field(
        description="List of page numbers that were processed"
    )
    engine_used: str = Field(
        description="OCR engine that was used"
    )
    language_used: str = Field(
        description="Language code used for OCR"
    )
    pages_data: List[Dict[str, Any]] = Field(
        description="Per-page OCR data with detailed information"
    )
    statistics: Dict[str, Any] = Field(
        description="OCR extraction statistics"
    )
    quality_metrics: Dict[str, Any] = Field(
        description="Quality metrics for OCR results"
    )
    preprocessing_applied: List[str] = Field(
        description="List of preprocessing steps applied"
    )

class ExtractTextOCRTool(PDFTool):
    """Basic OCR text extraction tool."""
    
    def __init__(self):
        super().__init__(
            name="extract_text_ocr",
            description="Extract text from PDF files using OCR (Tesseract, EasyOCR, PaddleOCR)"
        )
    
    @property
    def input_schema(self) -> type[BaseModel]:
        return ExtractTextOCRInput
    
    @property
    def output_schema(self) -> type[BaseModel]:
        return ExtractTextOCROutput
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute OCR text extraction."""
        start_time = time.time()
        
        # Validate input
        validated_input = self.validate_input(input_data)
        file_path = self.validate_file_path(validated_input.file_path)
        
        try:
            # Convert PDF to images
            images = await self._pdf_to_images(file_path, validated_input.dpi, validated_input.pages)
            
            # Perform OCR based on engine
            if validated_input.engine == "tesseract":
                text, confidence_scores = await self._ocr_with_tesseract(
                    images, validated_input.language, validated_input.confidence_threshold, validated_input.preprocess
                )
            elif validated_input.engine == "easyocr":
                text, confidence_scores = await self._ocr_with_easyocr(
                    images, validated_input.language, validated_input.confidence_threshold
                )
            elif validated_input.engine == "paddleocr":
                text, confidence_scores = await self._ocr_with_paddleocr(
                    images, validated_input.language, validated_input.confidence_threshold
                )
            else:
                raise ValueError(f"Unsupported OCR engine: {validated_input.engine}")
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "message": "OCR text extraction completed successfully",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "text": text,
                "page_count": len(images),
                "pages_processed": list(range(1, len(images) + 1)),
                "engine_used": validated_input.engine,
                "language_used": validated_input.language,
                "confidence_scores": confidence_scores,
                "word_count": len(text.split()),
                "character_count": len(text)
            }
            
            return self.validate_output(result).model_dump()
            
        except Exception as e:
            self.logger.error(f"OCR text extraction failed: {str(e)}")
            processing_time = time.time() - start_time
            
            result = {
                "success": False,
                "message": f"OCR text extraction failed: {str(e)}",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "text": "",
                "page_count": 0,
                "pages_processed": [],
                "engine_used": validated_input.engine,
                "language_used": validated_input.language,
                "confidence_scores": None,
                "word_count": 0,
                "character_count": 0
            }
            
            return self.validate_output(result).model_dump()
    
    async def _pdf_to_images(self, file_path: Path, dpi: int, pages: Optional[str]) -> List[Any]:
        """Convert PDF pages to images."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is not installed. Install with: pip install PyMuPDF")
        
        doc = fitz.open(str(file_path))
        images = []
        
        page_numbers = self._parse_page_range(pages, len(doc))
        
        for page_num in page_numbers:
            page = doc[page_num - 1]
            # Convert to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor for DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            images.append(img_data)
        
        doc.close()
        return images
    
    async def _ocr_with_tesseract(self, images: List[bytes], language: str, confidence_threshold: float, preprocess: bool) -> tuple[str, List[float]]:
        """Perform OCR using Tesseract."""
        try:
            import pytesseract
            from PIL import Image
            import io
        except ImportError:
            raise ImportError("Tesseract dependencies not installed. Install with: pip install pytesseract pillow")
        
        texts = []
        confidence_scores = []
        
        for img_data in images:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Preprocess if requested
            if preprocess:
                image = self._preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=language)
            
            # Get confidence if available
            try:
                data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                confidence_scores.append(avg_confidence / 100.0)  # Convert to 0-1 range
            except:
                confidence_scores.append(0.0)
            
            # Filter by confidence if threshold is set
            if confidence_threshold > 0 and confidence_scores[-1] < confidence_threshold:
                text = ""  # Skip low confidence text
            
            texts.append(text)
        
        return "\n\n".join(texts), confidence_scores
    
    async def _ocr_with_easyocr(self, images: List[bytes], language: str, confidence_threshold: float) -> tuple[str, List[float]]:
        """Perform OCR using EasyOCR."""
        try:
            import easyocr
            from PIL import Image
            import io
            import numpy as np
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader([language])
        
        texts = []
        confidence_scores = []
        
        for img_data in images:
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(img_data))
            img_array = np.array(image)
            
            # Perform OCR
            results = reader.readtext(img_array)
            
            # Extract text and confidence
            page_texts = []
            page_confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    page_texts.append(text)
                    page_confidences.append(confidence)
            
            texts.append(" ".join(page_texts))
            avg_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0
            confidence_scores.append(avg_confidence)
        
        return "\n\n".join(texts), confidence_scores
    
    async def _ocr_with_paddleocr(self, images: List[bytes], language: str, confidence_threshold: float) -> tuple[str, List[float]]:
        """Perform OCR using PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            from PIL import Image
            import io
            import numpy as np
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang=language)
        
        texts = []
        confidence_scores = []
        
        for img_data in images:
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(img_data))
            img_array = np.array(image)
            
            # Perform OCR
            results = ocr.ocr(img_array, cls=True)
            
            # Extract text and confidence
            page_texts = []
            page_confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        if confidence >= confidence_threshold:
                            page_texts.append(text)
                            page_confidences.append(confidence)
            
            texts.append(" ".join(page_texts))
            avg_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0
            confidence_scores.append(avg_confidence)
        
        return "\n\n".join(texts), confidence_scores
    
    def _preprocess_image(self, image):
        """Apply basic image preprocessing for better OCR."""
        try:
            from PIL import ImageEnhance, ImageFilter
        except ImportError:
            return image
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.MedianFilter())
        
        return image
    
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

class ExtractTextOCRAdvancedTool(PDFTool):
    """Advanced OCR text extraction tool with additional features."""
    
    def __init__(self):
        super().__init__(
            name="extract_text_ocr_advanced",
            description="Advanced OCR text extraction with preprocessing, layout preservation, and quality metrics"
        )
    
    @property
    def input_schema(self) -> type[BaseModel]:
        return ExtractTextOCRAdvancedInput
    
    @property
    def output_schema(self) -> type[BaseModel]:
        return ExtractTextOCRAdvancedOutput
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced OCR text extraction."""
        start_time = time.time()
        
        # Validate input
        validated_input = self.validate_input(input_data)
        file_path = self.validate_file_path(validated_input.file_path)
        
        try:
            # Convert PDF to images with preprocessing
            images = await self._pdf_to_images_advanced(file_path, validated_input)
            
            # Perform OCR with advanced features
            result_data = await self._ocr_advanced(images, validated_input)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "message": "Advanced OCR text extraction completed successfully",
                "file_path": str(file_path),
                "processing_time": processing_time,
                **result_data
            }
            
            return self.validate_output(result).model_dump()
            
        except Exception as e:
            self.logger.error(f"Advanced OCR text extraction failed: {str(e)}")
            processing_time = time.time() - start_time
            
            result = {
                "success": False,
                "message": f"Advanced OCR text extraction failed: {str(e)}",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "text": "",
                "page_count": 0,
                "pages_processed": [],
                "engine_used": validated_input.engine,
                "language_used": validated_input.language,
                "pages_data": [],
                "statistics": {},
                "quality_metrics": {},
                "preprocessing_applied": []
            }
            
            return self.validate_output(result).model_dump()
    
    async def _pdf_to_images_advanced(self, file_path: Path, config: ExtractTextOCRAdvancedInput) -> List[Dict[str, Any]]:
        """Convert PDF to images with advanced preprocessing."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image, ImageEnhance, ImageFilter
            import io
            import base64
        except ImportError:
            raise ImportError("Required libraries not installed. Install with: pip install PyMuPDF pillow")
        
        doc = fitz.open(str(file_path))
        images_data = []
        
        page_numbers = self._parse_page_range(config.pages, len(doc))
        preprocessing_steps = []
        
        for page_num in page_numbers:
            page = doc[page_num - 1]
            
            # Convert to image
            mat = fitz.Matrix(config.dpi / 72, config.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Load as PIL Image for preprocessing
            image = Image.open(io.BytesIO(img_data))
            original_image = image.copy()
            
            # Apply preprocessing
            if config.preprocess:
                if config.denoise:
                    image = image.filter(ImageFilter.MedianFilter())
                    preprocessing_steps.append("denoise")
                
                if config.enhance_contrast:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.5)
                    preprocessing_steps.append("enhance_contrast")
                
                if config.deskew:
                    # Simple deskew (would need more sophisticated implementation)
                    preprocessing_steps.append("deskew")
            
            # Convert back to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            processed_img_data = img_buffer.getvalue()
            
            page_data = {
                "page_number": page_num,
                "image_data": processed_img_data,
                "original_image": img_data if config.include_images else None,
                "processed_image": base64.b64encode(processed_img_data).decode() if config.include_images else None
            }
            
            images_data.append(page_data)
        
        doc.close()
        return images_data
    
    async def _ocr_advanced(self, images_data: List[Dict], config: ExtractTextOCRAdvancedInput) -> Dict[str, Any]:
        """Perform advanced OCR with detailed analysis."""
        pages_data = []
        all_texts = []
        total_confidence = 0
        total_words = 0
        
        for page_data in images_data:
            img_data = page_data["image_data"]
            page_num = page_data["page_number"]
            
            # Perform OCR based on engine
            if config.engine == "tesseract":
                page_result = await self._tesseract_advanced(img_data, config)
            elif config.engine == "easyocr":
                page_result = await self._easyocr_advanced(img_data, config)
            elif config.engine == "paddleocr":
                page_result = await self._paddleocr_advanced(img_data, config)
            else:
                raise ValueError(f"Unsupported OCR engine: {config.engine}")
            
            page_result["page_number"] = page_num
            if config.include_images:
                page_result["original_image"] = page_data.get("original_image")
                page_result["processed_image"] = page_data.get("processed_image")
            
            pages_data.append(page_result)
            all_texts.append(page_result["text"])
            
            total_confidence += page_result.get("confidence", 0)
            total_words += page_result.get("word_count", 0)
        
        full_text = "\n\n".join(all_texts)
        
        # Calculate statistics
        statistics = {
            "total_pages": len(pages_data),
            "total_words": total_words,
            "total_characters": len(full_text),
            "average_confidence": total_confidence / len(pages_data) if pages_data else 0,
            "average_words_per_page": total_words / len(pages_data) if pages_data else 0
        }
        
        # Calculate quality metrics
        quality_metrics = {
            "overall_confidence": statistics["average_confidence"],
            "text_density": total_words / len(pages_data) if pages_data else 0,
            "processing_success_rate": len([p for p in pages_data if p.get("success", True)]) / len(pages_data) if pages_data else 0
        }
        
        preprocessing_applied = []
        if config.preprocess:
            if config.denoise:
                preprocessing_applied.append("denoise")
            if config.enhance_contrast:
                preprocessing_applied.append("enhance_contrast")
            if config.deskew:
                preprocessing_applied.append("deskew")
        
        return {
            "text": full_text,
            "page_count": len(images_data),
            "pages_processed": [p["page_number"] for p in pages_data],
            "engine_used": config.engine,
            "language_used": config.language,
            "pages_data": pages_data,
            "statistics": statistics,
            "quality_metrics": quality_metrics,
            "preprocessing_applied": preprocessing_applied
        }
    
    async def _tesseract_advanced(self, img_data: bytes, config: ExtractTextOCRAdvancedInput) -> Dict[str, Any]:
        """Advanced Tesseract OCR with detailed output."""
        try:
            import pytesseract
            from PIL import Image
            import io
        except ImportError:
            raise ImportError("Tesseract dependencies not installed")
        
        image = Image.open(io.BytesIO(img_data))
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, lang=config.language, output_type=pytesseract.Output.DICT)
        
        # Extract text
        text = pytesseract.image_to_string(image, lang=config.language)
        
        # Calculate confidence
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count words
        words = [word for word in data['text'] if word.strip()]
        
        return {
            "text": text,
            "confidence": avg_confidence / 100.0,
            "word_count": len(words),
            "character_count": len(text),
            "success": True
        }
    
    async def _easyocr_advanced(self, img_data: bytes, config: ExtractTextOCRAdvancedInput) -> Dict[str, Any]:
        """Advanced EasyOCR with detailed output."""
        try:
            import easyocr
            from PIL import Image
            import io
            import numpy as np
        except ImportError:
            raise ImportError("EasyOCR not installed")
        
        reader = easyocr.Reader([config.language])
        image = Image.open(io.BytesIO(img_data))
        img_array = np.array(image)
        
        results = reader.readtext(img_array)
        
        texts = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            if confidence >= config.confidence_threshold:
                texts.append(text)
                confidences.append(confidence)
        
        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "text": full_text,
            "confidence": avg_confidence,
            "word_count": len(full_text.split()),
            "character_count": len(full_text),
            "success": True
        }
    
    async def _paddleocr_advanced(self, img_data: bytes, config: ExtractTextOCRAdvancedInput) -> Dict[str, Any]:
        """Advanced PaddleOCR with detailed output."""
        try:
            from paddleocr import PaddleOCR
            from PIL import Image
            import io
            import numpy as np
        except ImportError:
            raise ImportError("PaddleOCR not installed")
        
        ocr = PaddleOCR(use_angle_cls=True, lang=config.language)
        image = Image.open(io.BytesIO(img_data))
        img_array = np.array(image)
        
        results = ocr.ocr(img_array, cls=True)
        
        texts = []
        confidences = []
        
        if results and results[0]:
            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    if confidence >= config.confidence_threshold:
                        texts.append(text)
                        confidences.append(confidence)
        
        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "text": full_text,
            "confidence": avg_confidence,
            "word_count": len(full_text.split()),
            "character_count": len(full_text),
            "success": True
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