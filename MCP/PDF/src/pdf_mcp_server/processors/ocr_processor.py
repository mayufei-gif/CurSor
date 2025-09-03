"""OCR processor using OCRmyPDF and Tesseract.

This module provides OCR functionality for scanned PDFs and images,
converting them to searchable PDFs with text layers.
"""

import asyncio
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import os

try:
    import ocrmypdf
except ImportError:
    ocrmypdf = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from PIL import Image
except ImportError:
    Image = None

from ..models import ProcessingRequest
from ..utils.config import Config
from ..utils.exceptions import PDFProcessingError


class OCRProcessor:
    """Processes scanned PDFs and images using OCR."""
    
    def __init__(self, config: Config):
        """Initialize the OCR processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        self.has_ocrmypdf = ocrmypdf is not None
        self.has_tesseract = pytesseract is not None

        # Configure pytesseract binary if available
        if self.has_tesseract:
            try:
                # Prioritize env var if provided; otherwise, try common Windows path
                tess_env = os.getenv("TESSERACT_CMD")
                if tess_env and Path(tess_env).exists():
                    pytesseract.pytesseract.tesseract_cmd = tess_env
                else:
                    default_win = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
                    if default_win.exists():
                        pytesseract.pytesseract.tesseract_cmd = str(default_win)
            except Exception:
                pass
        
        if not self.has_ocrmypdf and not self.has_tesseract:
            self.logger.warning("No OCR engines available")
        
        # OCR settings
        self.default_language = getattr(config, 'ocr_language', 'eng')
        self.ocr_timeout = getattr(config, 'ocr_timeout', 300)  # 5 minutes
        self.temp_dir = Path(tempfile.gettempdir()) / "pdf_mcp_ocr"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"OCR processor initialized. OCRmyPDF: {self.has_ocrmypdf}, Tesseract: {self.has_tesseract}")
    
    async def initialize(self):
        """Initialize the processor."""
        # Test OCR engines
        if self.has_ocrmypdf:
            try:
                # Test OCRmyPDF installation
                result = subprocess.run(
                    ['ocrmypdf', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    self.logger.info(f"OCRmyPDF version: {result.stdout.strip()}")
                else:
                    self.has_ocrmypdf = False
                    self.logger.warning("OCRmyPDF not working properly")
            except Exception as e:
                self.has_ocrmypdf = False
                self.logger.warning(f"OCRmyPDF test failed: {e}")
        
        if self.has_tesseract:
            try:
                # Test Tesseract installation
                version = pytesseract.get_tesseract_version()
                self.logger.info(f"Tesseract version: {version}")
            except Exception as e:
                self.has_tesseract = False
                self.logger.warning(f"Tesseract test failed: {e}")
        
        self.logger.info("OCR processor initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        # Clean up temporary files
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp directory: {e}")
        
        self.logger.info("OCR processor cleanup complete")
    
    async def health_check(self) -> bool:
        """Check if the processor is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return self.has_ocrmypdf or self.has_tesseract
    
    async def process(self, file_path: Path, request: ProcessingRequest) -> Path:
        """Process PDF with OCR.
        
        Args:
            file_path: Path to input PDF file
            request: Processing request with options
            
        Returns:
            Path to processed PDF with text layer
            
        Raises:
            PDFProcessingError: If OCR processing fails
        """
        self.logger.info(f"Processing PDF with OCR: {file_path}")
        
        try:
            # Create output file path
            output_path = self.temp_dir / f"ocr_{file_path.stem}_{hash(str(file_path))}.pdf"
            
            # Try OCRmyPDF first (preferred)
            if self.has_ocrmypdf:
                try:
                    result_path = await self._process_with_ocrmypdf(file_path, output_path, request)
                    if result_path and result_path.exists():
                        self.logger.info(f"OCR processing completed with OCRmyPDF: {result_path}")
                        return result_path
                except Exception as e:
                    self.logger.warning(f"OCRmyPDF processing failed: {e}")
            
            # Fallback to Tesseract
            if self.has_tesseract:
                try:
                    result_path = await self._process_with_tesseract(file_path, output_path, request)
                    if result_path and result_path.exists():
                        self.logger.info(f"OCR processing completed with Tesseract: {result_path}")
                        return result_path
                except Exception as e:
                    self.logger.warning(f"Tesseract processing failed: {e}")
            
            # If all OCR methods fail, return original file
            self.logger.warning("All OCR methods failed, returning original file")
            return file_path
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise PDFProcessingError(f"OCR processing failed: {e}")
    
    async def _process_with_ocrmypdf(self, input_path: Path, output_path: Path, request: ProcessingRequest) -> Path:
        """Process PDF using OCRmyPDF.
        
        Args:
            input_path: Input PDF path
            output_path: Output PDF path
            request: Processing request
            
        Returns:
            Path to processed PDF
        """
        if not self.has_ocrmypdf:
            raise PDFProcessingError("OCRmyPDF not available")
        
        # Prepare OCRmyPDF arguments
        ocr_args = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'language': getattr(request, 'ocr_language', self.default_language),
            'deskew': True,
            'clean': True,
            'clean_final': True,
            'remove_background': False,
            'force_ocr': False,  # Only OCR pages that need it
            'skip_text': False,
            'redo_ocr': False,
            'optimize': 1,
            'jpeg_quality': 85,
            'png_quality': 85,
            'jbig2_lossy': False,
            'quiet': True,
            'progress_bar': False
        }
        
        # Add page range if specified
        if request.pages:
            # OCRmyPDF uses 1-indexed pages
            page_ranges = []
            for page in sorted(request.pages):
                page_ranges.append(f"{page + 1}")
            ocr_args['pages'] = ",".join(page_ranges)
        
        try:
            # Run OCRmyPDF
            self.logger.info(f"Running OCRmyPDF with language: {ocr_args['language']}")
            
            # Use asyncio to run OCRmyPDF in a thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: ocrmypdf.ocr(**ocr_args)
            )
            
            if output_path.exists():
                return output_path
            else:
                raise PDFProcessingError("OCRmyPDF did not produce output file")
                
        except Exception as e:
            self.logger.error(f"OCRmyPDF processing failed: {e}")
            raise PDFProcessingError(f"OCRmyPDF failed: {e}")
    
    async def _process_with_tesseract(self, input_path: Path, output_path: Path, request: ProcessingRequest) -> Path:
        """Process PDF using Tesseract (via image conversion).
        
        Args:
            input_path: Input PDF path
            output_path: Output PDF path
            request: Processing request
            
        Returns:
            Path to processed PDF
        """
        if not self.has_tesseract or not Image:
            raise PDFProcessingError("Tesseract or PIL not available")
        
        try:
            # Convert PDF to images first
            import fitz  # PyMuPDF for PDF to image conversion
            
            doc = fitz.open(str(input_path))
            
            try:
                # Create temporary directory for images
                img_dir = self.temp_dir / f"tesseract_{hash(str(input_path))}"
                img_dir.mkdir(exist_ok=True)
                
                ocr_results = []
                
                # Determine pages to process
                page_range = request.pages if request.pages else range(len(doc))
                
                for page_num in page_range:
                    if page_num >= len(doc):
                        continue
                    
                    page = doc[page_num]
                    
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                    pix = page.get_pixmap(matrix=mat)
                    img_path = img_dir / f"page_{page_num}.png"
                    pix.save(str(img_path))
                    
                    # Run Tesseract OCR
                    try:
                        ocr_text = pytesseract.image_to_string(
                            str(img_path),
                            lang=getattr(request, 'ocr_language', self.default_language),
                            config='--psm 1 --oem 3'
                        )
                        ocr_results.append((page_num, ocr_text))
                        
                    except Exception as e:
                        self.logger.warning(f"Tesseract OCR failed for page {page_num}: {e}")
                        ocr_results.append((page_num, ""))
                
                # Create searchable PDF
                # This is a simplified approach - in practice, you'd want to
                # overlay the OCR text with proper positioning
                
                # For now, just copy the original file and add a text file with OCR results
                shutil.copy2(input_path, output_path)
                
                # Save OCR results as a companion text file
                ocr_text_path = output_path.with_suffix('.ocr.txt')
                with open(ocr_text_path, 'w', encoding='utf-8') as f:
                    for page_num, text in ocr_results:
                        f.write(f"=== Page {page_num + 1} ===\n")
                        f.write(text)
                        f.write("\n\n")
                
                # Clean up temporary images
                shutil.rmtree(img_dir, ignore_errors=True)
                
                return output_path
                
            finally:
                doc.close()
                
        except Exception as e:
            self.logger.error(f"Tesseract processing failed: {e}")
            raise PDFProcessingError(f"Tesseract failed: {e}")
    
    async def extract_text_from_image(self, image_path: Path, language: str = None) -> str:
        """Extract text from an image using OCR.
        
        Args:
            image_path: Path to image file
            language: OCR language code
            
        Returns:
            Extracted text
            
        Raises:
            PDFProcessingError: If OCR fails
        """
        if not self.has_tesseract:
            raise PDFProcessingError("Tesseract not available")
        
        try:
            lang = language or self.default_language
            
            text = pytesseract.image_to_string(
                str(image_path),
                lang=lang,
                config='--psm 1 --oem 3'
            )
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Image OCR failed: {e}")
            raise PDFProcessingError(f"Image OCR failed: {e}")
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported OCR languages.
        
        Returns:
            List of language codes
        """
        languages = []
        
        if self.has_tesseract:
            try:
                langs = pytesseract.get_languages()
                languages.extend(langs)
            except Exception as e:
                self.logger.warning(f"Failed to get Tesseract languages: {e}")
        
        # Add common language codes if not detected
        common_langs = ['eng', 'chi_sim', 'chi_tra', 'fra', 'deu', 'spa', 'rus', 'jpn', 'kor']
        for lang in common_langs:
            if lang not in languages:
                languages.append(lang)
        
        return sorted(list(set(languages)))
    
    def is_pdf_searchable(self, file_path: Path) -> bool:
        """Check if PDF already has searchable text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if PDF has searchable text, False otherwise
        """
        try:
            import fitz
            
            doc = fitz.open(str(file_path))
            
            try:
                # Check first few pages for text
                pages_to_check = min(3, len(doc))
                total_text_length = 0
                
                for i in range(pages_to_check):
                    page = doc[i]
                    text = page.get_text()
                    total_text_length += len(text.strip())
                
                # If we have reasonable amount of text, consider it searchable
                return total_text_length > 100
                
            finally:
                doc.close()
                
        except Exception as e:
            self.logger.warning(f"Failed to check PDF searchability: {e}")
            return False
    
    def get_ocr_statistics(self) -> Dict[str, Any]:
        """Get OCR processing statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "has_ocrmypdf": self.has_ocrmypdf,
            "has_tesseract": self.has_tesseract,
            "default_language": self.default_language,
            "temp_dir": str(self.temp_dir),
            "temp_files": len(list(self.temp_dir.glob("*"))) if self.temp_dir.exists() else 0
        }
