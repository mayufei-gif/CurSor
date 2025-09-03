#!/usr/bin/env python3
"""
Formula extraction tools for PDF documents.
Supports LaTeX formula recognition and extraction using various OCR engines.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json

from pydantic import Field, ConfigDict

from .base import PDFTool, PDFToolInput, PDFToolOutput

logger = logging.getLogger(__name__)


class ExtractFormulasInput(PDFToolInput):
    """Input schema for formula extraction."""
    
    model_config = ConfigDict(extra='forbid')
    
    method: str = Field(
        default="latex-ocr",
        description="Formula extraction method: 'latex-ocr', 'pix2tex', 'mathpix', 'tesseract'"
    )
    pages: Optional[str] = Field(
        default=None,
        description="Page range to extract from (e.g., '1-5', '1,3,5', 'all')"
    )
    output_format: str = Field(
        default="latex",
        description="Output format: 'latex', 'mathml', 'text', 'json'"
    )
    confidence_threshold: float = Field(
        default=0.8,
        description="Minimum confidence threshold for formula detection (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    include_images: bool = Field(
        default=False,
        description="Include formula images in output"
    )
    dpi: int = Field(
        default=300,
        description="DPI for PDF to image conversion",
        ge=72,
        le=600
    )
    language: str = Field(
        default="en",
        description="Language for OCR processing"
    )


class ExtractFormulasOutput(PDFToolOutput):
    """Output schema for formula extraction."""
    
    formulas: List[Dict[str, Any]] = Field(
        description="List of extracted formulas with metadata"
    )
    total_formulas: int = Field(
        description="Total number of formulas found"
    )
    pages_processed: List[int] = Field(
        description="List of page numbers processed"
    )
    extraction_method: str = Field(
        description="Method used for extraction"
    )
    statistics: Dict[str, Any] = Field(
        description="Extraction statistics and metadata"
    )


class ExtractFormulasTool(PDFTool):
    """Tool for extracting mathematical formulas from PDF documents."""
    
    name = "extract_formulas"
    description = "Extract mathematical formulas from PDF documents using OCR and LaTeX recognition"
    input_schema = ExtractFormulasInput
    output_schema = ExtractFormulasOutput
    
    def execute(self, input_data: ExtractFormulasInput) -> ExtractFormulasOutput:
        """Execute formula extraction."""
        try:
            # Validate file exists
            if not Path(input_data.file_path).exists():
                raise FileNotFoundError(f"PDF file not found: {input_data.file_path}")
            
            # Parse page range
            pages_to_process = self._parse_page_range(input_data.pages, input_data.file_path)
            
            # Extract formulas based on method
            if input_data.method == "latex-ocr":
                formulas = self._extract_with_latex_ocr(
                    input_data.file_path, 
                    pages_to_process, 
                    input_data
                )
            elif input_data.method == "pix2tex":
                formulas = self._extract_with_pix2tex(
                    input_data.file_path, 
                    pages_to_process, 
                    input_data
                )
            elif input_data.method == "mathpix":
                formulas = self._extract_with_mathpix(
                    input_data.file_path, 
                    pages_to_process, 
                    input_data
                )
            elif input_data.method == "tesseract":
                formulas = self._extract_with_tesseract(
                    input_data.file_path, 
                    pages_to_process, 
                    input_data
                )
            else:
                raise ValueError(f"Unsupported extraction method: {input_data.method}")
            
            # Filter by confidence threshold
            filtered_formulas = [
                formula for formula in formulas 
                if formula.get('confidence', 1.0) >= input_data.confidence_threshold
            ]
            
            # Format output
            formatted_formulas = self._format_formulas(
                filtered_formulas, 
                input_data.output_format
            )
            
            # Generate statistics
            statistics = self._generate_statistics(formulas, filtered_formulas, input_data)
            
            return ExtractFormulasOutput(
                success=True,
                formulas=formatted_formulas,
                total_formulas=len(formatted_formulas),
                pages_processed=pages_to_process,
                extraction_method=input_data.method,
                statistics=statistics,
                metadata={
                    "file_path": input_data.file_path,
                    "method": input_data.method,
                    "confidence_threshold": input_data.confidence_threshold,
                    "output_format": input_data.output_format
                }
            )
            
        except Exception as e:
            logger.error(f"Formula extraction failed: {str(e)}")
            return ExtractFormulasOutput(
                success=False,
                error=str(e),
                formulas=[],
                total_formulas=0,
                pages_processed=[],
                extraction_method=input_data.method,
                statistics={},
                metadata={"file_path": input_data.file_path}
            )
    
    def _parse_page_range(self, pages: Optional[str], file_path: str) -> List[int]:
        """Parse page range specification."""
        if not pages or pages.lower() == 'all':
            # Get total pages from PDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                total_pages = len(doc)
                doc.close()
                return list(range(1, total_pages + 1))
            except ImportError:
                logger.warning("PyMuPDF not available, defaulting to first page")
                return [1]
        
        page_list = []
        for part in pages.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_list.extend(range(start, end + 1))
            else:
                page_list.append(int(part))
        
        return sorted(list(set(page_list)))
    
    def _extract_with_latex_ocr(self, file_path: str, pages: List[int], input_data: ExtractFormulasInput) -> List[Dict[str, Any]]:
        """Extract formulas using LaTeX-OCR."""
        formulas = []
        
        try:
            # This is a placeholder implementation
            # In a real implementation, you would use latex-ocr library
            logger.info(f"Using LaTeX-OCR to extract formulas from {len(pages)} pages")
            
            # Mock formula extraction
            for page_num in pages:
                # Simulate finding formulas on each page
                mock_formulas = [
                    {
                        "page": page_num,
                        "bbox": [100, 200, 300, 250],
                        "latex": "E = mc^2",
                        "confidence": 0.95,
                        "type": "inline"
                    },
                    {
                        "page": page_num,
                        "bbox": [150, 400, 400, 480],
                        "latex": "\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}",
                        "confidence": 0.88,
                        "type": "display"
                    }
                ]
                formulas.extend(mock_formulas)
            
        except Exception as e:
            logger.error(f"LaTeX-OCR extraction failed: {str(e)}")
            
        return formulas
    
    def _extract_with_pix2tex(self, file_path: str, pages: List[int], input_data: ExtractFormulasInput) -> List[Dict[str, Any]]:
        """Extract formulas using pix2tex."""
        formulas = []
        
        try:
            # This is a placeholder implementation
            # In a real implementation, you would use pix2tex library
            logger.info(f"Using pix2tex to extract formulas from {len(pages)} pages")
            
            # Mock formula extraction
            for page_num in pages:
                mock_formulas = [
                    {
                        "page": page_num,
                        "bbox": [120, 180, 280, 220],
                        "latex": "\\frac{d}{dx}[f(x)] = f'(x)",
                        "confidence": 0.92,
                        "type": "inline"
                    }
                ]
                formulas.extend(mock_formulas)
            
        except Exception as e:
            logger.error(f"pix2tex extraction failed: {str(e)}")
            
        return formulas
    
    def _extract_with_mathpix(self, file_path: str, pages: List[int], input_data: ExtractFormulasInput) -> List[Dict[str, Any]]:
        """Extract formulas using Mathpix API."""
        formulas = []
        
        try:
            # This is a placeholder implementation
            # In a real implementation, you would use Mathpix API
            logger.info(f"Using Mathpix to extract formulas from {len(pages)} pages")
            
            # Mock formula extraction
            for page_num in pages:
                mock_formulas = [
                    {
                        "page": page_num,
                        "bbox": [80, 300, 350, 360],
                        "latex": "\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}",
                        "confidence": 0.97,
                        "type": "display"
                    }
                ]
                formulas.extend(mock_formulas)
            
        except Exception as e:
            logger.error(f"Mathpix extraction failed: {str(e)}")
            
        return formulas
    
    def _extract_with_tesseract(self, file_path: str, pages: List[int], input_data: ExtractFormulasInput) -> List[Dict[str, Any]]:
        """Extract formulas using Tesseract OCR."""
        formulas = []
        
        try:
            # This is a placeholder implementation
            # In a real implementation, you would use Tesseract with math detection
            logger.info(f"Using Tesseract to extract formulas from {len(pages)} pages")
            
            # Mock formula extraction
            for page_num in pages:
                mock_formulas = [
                    {
                        "page": page_num,
                        "bbox": [200, 150, 450, 200],
                        "text": "x^2 + y^2 = z^2",
                        "confidence": 0.75,
                        "type": "text"
                    }
                ]
                formulas.extend(mock_formulas)
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {str(e)}")
            
        return formulas
    
    def _format_formulas(self, formulas: List[Dict[str, Any]], output_format: str) -> List[Dict[str, Any]]:
        """Format extracted formulas according to output format."""
        formatted = []
        
        for formula in formulas:
            formatted_formula = formula.copy()
            
            if output_format == "latex":
                # Keep LaTeX format
                pass
            elif output_format == "mathml":
                # Convert to MathML (placeholder)
                if 'latex' in formula:
                    formatted_formula['mathml'] = f"<math>{formula['latex']}</math>"
            elif output_format == "text":
                # Convert to plain text (placeholder)
                if 'latex' in formula:
                    formatted_formula['text'] = formula['latex'].replace('\\', '')
            elif output_format == "json":
                # Keep as structured JSON
                pass
            
            formatted.append(formatted_formula)
        
        return formatted
    
    def _generate_statistics(self, all_formulas: List[Dict[str, Any]], 
                           filtered_formulas: List[Dict[str, Any]], 
                           input_data: ExtractFormulasInput) -> Dict[str, Any]:
        """Generate extraction statistics."""
        stats = {
            "total_detected": len(all_formulas),
            "total_filtered": len(filtered_formulas),
            "confidence_threshold": input_data.confidence_threshold,
            "average_confidence": 0.0,
            "formula_types": {},
            "pages_with_formulas": set(),
            "extraction_method": input_data.method
        }
        
        if all_formulas:
            # Calculate average confidence
            confidences = [f.get('confidence', 1.0) for f in all_formulas]
            stats["average_confidence"] = sum(confidences) / len(confidences)
            
            # Count formula types
            for formula in all_formulas:
                formula_type = formula.get('type', 'unknown')
                stats["formula_types"][formula_type] = stats["formula_types"].get(formula_type, 0) + 1
            
            # Track pages with formulas
            for formula in all_formulas:
                stats["pages_with_formulas"].add(formula.get('page', 0))
        
        stats["pages_with_formulas"] = list(stats["pages_with_formulas"])
        
        return stats


class ExtractFormulasAdvancedTool(PDFTool):
    """Advanced formula extraction tool with additional features."""
    
    name = "extract_formulas_advanced"
    description = "Advanced mathematical formula extraction with preprocessing and validation"
    input_schema = ExtractFormulasInput
    output_schema = ExtractFormulasOutput
    
    def execute(self, input_data: ExtractFormulasInput) -> ExtractFormulasOutput:
        """Execute advanced formula extraction."""
        # This would include additional features like:
        # - Image preprocessing
        # - Formula validation
        # - Multiple method combination
        # - Post-processing and cleanup
        
        # For now, delegate to basic extraction
        basic_tool = ExtractFormulasTool()
        return basic_tool.execute(input_data)