"""Formula extraction processor using LaTeX-OCR and pix2tex.

This module provides mathematical formula recognition functionality,
converting formula images to LaTeX code.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import io
import base64

try:
    import torch
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    transforms = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ..models import (
    ProcessingRequest,
    FormulaExtractionResult,
    FormulaData,
    BoundingBox,
    FormulaModel
)
from ..utils.config import Config
from ..utils.exceptions import PDFProcessingError


class FormulaExtractor:
    """Extracts mathematical formulas from PDF documents."""
    
    def __init__(self, config: Config):
        """Initialize the formula extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        self.has_torch = torch is not None
        self.has_pil = Image is not None
        self.has_cv2 = cv2 is not None
        self.has_numpy = np is not None
        self.has_fitz = fitz is not None
        
        # Model settings
        self.device = 'cuda' if self.has_torch and torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "pdf_mcp_formulas"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Formula detection settings
        self.min_formula_area = getattr(config, 'min_formula_area', 100)
        self.max_formula_area = getattr(config, 'max_formula_area', 50000)
        self.confidence_threshold = getattr(config, 'formula_confidence_threshold', 0.5)
        
        self.logger.info(f"Formula extractor initialized. Device: {self.device}")
    
    async def initialize(self):
        """Initialize the extractor and load models."""
        if not self.has_torch or not self.has_pil:
            self.logger.warning("PyTorch or PIL not available, formula extraction will be limited")
            return
        
        try:
            # Try to load available models
            await self._load_models()
            self.logger.info("Formula extractor initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load formula models: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        # Clean up models
        for model_name in list(self.models.keys()):
            try:
                del self.models[model_name]
            except Exception as e:
                self.logger.warning(f"Failed to cleanup model {model_name}: {e}")
        
        # Clean up temporary files
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp directory: {e}")
        
        # Clear CUDA cache if available
        if self.has_torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Formula extractor cleanup complete")
    
    async def health_check(self) -> bool:
        """Check if the extractor is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return self.has_pil and self.has_fitz
    
    async def extract(self, file_path: Path, request: ProcessingRequest) -> FormulaExtractionResult:
        """Extract formulas from PDF.
        
        Args:
            file_path: Path to PDF file
            request: Processing request with options
            
        Returns:
            Formula extraction result
            
        Raises:
            PDFProcessingError: If extraction fails
        """
        self.logger.info(f"Extracting formulas from: {file_path}")
        
        try:
            if not self.has_fitz:
                raise PDFProcessingError("PyMuPDF required for formula extraction")
            
            formulas = []
            
            # Open PDF document
            doc = fitz.open(str(file_path))
            
            try:
                # Determine pages to process
                page_range = request.pages if request.pages else range(len(doc))
                
                for page_num in page_range:
                    if page_num >= len(doc):
                        continue
                    
                    page_formulas = await self._extract_formulas_from_page(
                        doc[page_num], page_num, request
                    )
                    formulas.extend(page_formulas)
                
            finally:
                doc.close()
            
            return FormulaExtractionResult(
                formulas=formulas,
                total_formulas=len(formulas),
                model_used=str(request.formula_model) if request.formula_model else "heuristic",
                processing_time=0.0,
            )
            
        except Exception as e:
            self.logger.error(f"Formula extraction failed: {e}")
            raise PDFProcessingError(f"Formula extraction failed: {e}")
    
    async def _load_models(self):
        """Load available formula recognition models."""
        # Try to load pix2tex/LaTeX-OCR
        try:
            # This is a placeholder - actual implementation would load the model
            # from huggingface or local files
            self.logger.info("Attempting to load LaTeX-OCR model...")
            
            # Example model loading (would need actual implementation)
            # from transformers import VisionEncoderDecoderModel, TrOCRProcessor
            # self.models['latex_ocr'] = {
            #     'model': VisionEncoderDecoderModel.from_pretrained("breezedeus/pix2text-mfd"),
            #     'processor': TrOCRProcessor.from_pretrained("breezedeus/pix2text-mfd")
            # }
            
            self.logger.info("LaTeX-OCR model loaded (placeholder)")
            
        except Exception as e:
            self.logger.warning(f"Failed to load LaTeX-OCR model: {e}")
        
        # Try to load other models (pix2tex, texify, etc.)
        # Similar placeholder implementations would go here
    
    async def _extract_formulas_from_page(self, page, page_num: int, request: ProcessingRequest) -> List[FormulaData]:
        """Extract formulas from a single page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            request: Processing request
            
        Returns:
            List of extracted formulas
        """
        formulas = []
        
        try:
            # Method 1: Detect formula regions using heuristics
            formula_regions = await self._detect_formula_regions(page, page_num)
            
            # Method 2: Extract images that might contain formulas
            image_regions = await self._extract_formula_images(page, page_num)
            
            # Combine and deduplicate regions
            all_regions = formula_regions + image_regions
            unique_regions = self._deduplicate_regions(all_regions)
            
            # Process each region
            for i, region in enumerate(unique_regions):
                try:
                    formula = await self._process_formula_region(
                        page, region, page_num, i, request
                    )
                    if formula:
                        formulas.append(formula)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process formula region {i}: {e}")
                    continue
            
        except Exception as e:
            self.logger.warning(f"Formula extraction failed for page {page_num}: {e}")
        
        return formulas
    
    async def _detect_formula_regions(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Detect potential formula regions using heuristics.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            List of potential formula regions
        """
        regions = []
        
        try:
            # Get text blocks with detailed information
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    # Look for mathematical symbols and patterns
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                    
                    # Heuristics for formula detection
                    if self._is_likely_formula(block_text):
                        bbox = block["bbox"]
                        regions.append({
                            "bbox": BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                            "type": "text_formula",
                            "confidence": self._calculate_formula_confidence(block_text),
                            "text": block_text
                        })
            
        except Exception as e:
            self.logger.warning(f"Formula region detection failed: {e}")
        
        return regions
    
    async def _extract_formula_images(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images that might contain formulas.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            List of potential formula image regions
        """
        regions = []
        
        try:
            # Get images from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image bbox
                    img_rects = page.get_image_rects(img[0])
                    
                    for rect in img_rects:
                        # Check if image size suggests it might be a formula
                        width = rect.width
                        height = rect.height
                        area = width * height
                        
                        if (self.min_formula_area <= area <= self.max_formula_area and
                            self._is_formula_aspect_ratio(width, height)):
                            
                            regions.append({
                                "bbox": BoundingBox(
                                    x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1
                                ),
                                "type": "image_formula",
                                "confidence": 0.7,  # Default confidence for images
                                "image_index": img_index
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Failed to process image {img_index}: {e}")
                    continue
            
        except Exception as e:
            self.logger.warning(f"Image extraction failed: {e}")
        
        return regions
    
    def _is_likely_formula(self, text: str) -> bool:
        """Check if text is likely to contain mathematical formulas.
        
        Args:
            text: Text to check
            
        Returns:
            True if likely contains formulas
        """
        if not text:
            return False
        
        # Mathematical symbols and patterns
        math_indicators = [
            '∫', '∑', '∏', '√', '∂', '∇', '∞', 'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω',
            '≤', '≥', '≠', '≈', '≡', '∈', '∉', '⊂', '⊃', '∪', '∩', '±', '×', '÷', '²', '³', '₁', '₂', '₃',
            'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'lim', 'max', 'min', 'arg',
            'dx', 'dy', 'dt', 'dr', 'dθ'
        ]
        
        # Check for mathematical symbols
        symbol_count = sum(1 for symbol in math_indicators if symbol in text)
        
        # Check for formula patterns
        pattern_indicators = [
            r'\b\w+\s*[=<>]\s*\w+',  # Variable assignments
            r'\b\w+\s*\(\s*\w+\s*\)',  # Function calls
            r'\b\d+\s*[+\-*/]\s*\d+',  # Arithmetic
            r'\b\w+\^\d+',  # Exponents
            r'\b\w+_\d+',  # Subscripts
        ]
        
        import re
        pattern_count = sum(1 for pattern in pattern_indicators if re.search(pattern, text))
        
        # Scoring
        score = symbol_count * 2 + pattern_count
        text_length = len(text.strip())
        
        # Adjust score based on text length
        if text_length > 0:
            density = score / text_length
            return density > 0.1 or score >= 2
        
        return False
    
    def _calculate_formula_confidence(self, text: str) -> float:
        """Calculate confidence that text contains formulas.
        
        Args:
            text: Text to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        if not text:
            return 0.0
        
        # Count mathematical indicators
        math_symbols = ['∫', '∑', '∏', '√', '∂', '∇', '∞', '≤', '≥', '≠', '≈', '≡']
        symbol_count = sum(1 for symbol in math_symbols if symbol in text)
        
        # Count Greek letters
        greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω']
        greek_count = sum(1 for letter in greek_letters if letter in text)
        
        # Count mathematical functions
        math_functions = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'lim']
        function_count = sum(1 for func in math_functions if func in text.lower())
        
        # Calculate confidence
        total_indicators = symbol_count + greek_count + function_count
        text_length = len(text.strip())
        
        if text_length == 0:
            return 0.0
        
        # Base confidence from indicator density
        density = total_indicators / text_length
        confidence = min(density * 10, 1.0)  # Scale and cap at 1.0
        
        # Boost confidence for very short text with indicators
        if text_length < 20 and total_indicators > 0:
            confidence = min(confidence + 0.3, 1.0)
        
        return confidence
    
    def _is_formula_aspect_ratio(self, width: float, height: float) -> bool:
        """Check if image dimensions suggest it might be a formula.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            True if aspect ratio suggests formula
        """
        if width <= 0 or height <= 0:
            return False
        
        aspect_ratio = width / height
        
        # Formulas are often wider than they are tall, but not extremely so
        return 0.5 <= aspect_ratio <= 8.0
    
    def _deduplicate_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping regions.
        
        Args:
            regions: List of regions
            
        Returns:
            List of unique regions
        """
        if not regions:
            return []
        
        # Sort by confidence (highest first)
        sorted_regions = sorted(regions, key=lambda r: r.get('confidence', 0), reverse=True)
        
        unique_regions = []
        
        for region in sorted_regions:
            bbox = region['bbox']
            
            # Check for significant overlap with existing regions
            is_duplicate = False
            for existing in unique_regions:
                existing_bbox = existing['bbox']
                
                if self._calculate_overlap(bbox, existing_bbox) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_regions.append(region)
        
        return unique_regions
    
    def _calculate_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Overlap ratio (0 to 1)
        """
        # Calculate intersection
        x_left = max(bbox1.x0, bbox2.x0)
        y_top = max(bbox1.y0, bbox2.y0)
        x_right = min(bbox1.x1, bbox2.x1)
        y_bottom = min(bbox1.y1, bbox2.y1)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0)
        area2 = (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0)
        
        # Calculate union
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    async def _process_formula_region(self, page, region: Dict[str, Any], page_num: int, 
                                    region_id: int, request: ProcessingRequest) -> Optional[FormulaData]:
        """Process a single formula region.
        
        Args:
            page: PyMuPDF page object
            region: Region information
            page_num: Page number
            region_id: Region identifier
            request: Processing request
            
        Returns:
            FormulaData object or None
        """
        try:
            bbox = region['bbox']
            
            # Extract region as image
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=rect)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Try to recognize formula
            latex_code = await self._recognize_formula(image, request.formula_model)
            
            # Save image to temp file for downstream usage
            out_name = f"formula_{page_num+1}_{region_id}.png"
            out_path = self.temp_dir / out_name
            try:
                with open(out_path, "wb") as f:
                    f.write(img_data)
            except Exception:
                out_path = None

            # Build FormulaData regardless of recognition success; use fallback
            return FormulaData(
                page=page_num + 1,
                formula_id=region_id,
                bbox=bbox,
                latex=latex_code or region.get('text') or "",
                confidence=float(region.get('confidence', 0.5) or 0.5),
                model=str(request.formula_model) if request.formula_model else "heuristic",
                image_path=str(out_path) if out_path else None,
                raw_text=region.get('text'),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to process formula region: {e}")
        
        return None
    
    async def _recognize_formula(self, image: Image.Image, model: Optional[FormulaModel]) -> Optional[str]:
        """Recognize formula from image.
        
        Args:
            image: PIL Image containing formula
            model: Formula recognition model to use
            
        Returns:
            LaTeX code or None
        """
        try:
            # This is a placeholder implementation
            # In practice, you would use actual models like:
            # - pix2tex/LaTeX-OCR
            # - texify
            # - im2latex models
            
            if model == FormulaModel.PIX2TEX:
                return await self._recognize_with_pix2tex(image)
            elif model == FormulaModel.LATEX_OCR:
                return await self._recognize_with_latex_ocr(image)
            elif model == FormulaModel.TEXIFY:
                return await self._recognize_with_texify(image)
            else:
                # Fallback to simple heuristic
                return await self._recognize_with_heuristic(image)
                
        except Exception as e:
            self.logger.warning(f"Formula recognition failed: {e}")
            return None
    
    async def _recognize_with_pix2tex(self, image: Image.Image) -> Optional[str]:
        """Recognize formula using pix2tex via CLI if available.

        Expects environment variable PIX2TEX_CMD pointing to the executable.
        Fallback to None if unavailable.
        """
        import os, subprocess, tempfile
        cmd = os.getenv("PIX2TEX_CMD")
        timeout = int(os.getenv("PIX2TEX_TIMEOUT", "30"))
        if not cmd:
            self.logger.debug("PIX2TEX_CMD not set; skipping pix2tex")
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, format="PNG")
                tmp.flush()
                proc = await asyncio.create_subprocess_exec(
                    cmd, tmp.name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    self.logger.warning("pix2tex timed out")
                    return None
                if proc.returncode == 0:
                    text = (out or b"").decode("utf-8", errors="ignore").strip()
                    return text or None
                else:
                    self.logger.warning(f"pix2tex failed rc={proc.returncode}: {(err or b'').decode(errors='ignore')}")
                    return None
        except Exception as e:
            self.logger.warning(f"pix2tex invocation error: {e}")
            return None
    
    async def _recognize_with_latex_ocr(self, image: Image.Image) -> Optional[str]:
        """Recognize formula using LaTeX-OCR (placeholder: not bundled)."""
        # If a CLI is provided via LATEX_OCR_CMD, try to use it
        import os, subprocess, tempfile
        cmd = os.getenv("LATEX_OCR_CMD")
        timeout = int(os.getenv("LATEX_OCR_TIMEOUT", "30"))
        if not cmd:
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, format="PNG")
                tmp.flush()
                proc = await asyncio.create_subprocess_exec(
                    cmd, tmp.name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    self.logger.warning("LaTeX-OCR timed out")
                    return None
                if proc.returncode == 0:
                    text = (out or b"").decode("utf-8", errors="ignore").strip()
                    return text or None
                else:
                    self.logger.warning(f"LaTeX-OCR failed rc={proc.returncode}: {(err or b'').decode(errors='ignore')}")
                    return None
        except Exception as e:
            self.logger.warning(f"LaTeX-OCR invocation error: {e}")
            return None
    
    async def _recognize_with_texify(self, image: Image.Image) -> Optional[str]:
        """Recognize formula using texify model."""
        # Placeholder implementation
        self.logger.info("Using texify model (placeholder)")
        return "\\text{Formula recognized with texify}"
    
    async def _recognize_with_heuristic(self, image: Image.Image) -> Optional[str]:
        """Simple heuristic formula recognition."""
        # Very basic placeholder - just return a generic formula
        return "\\text{Formula detected}"
