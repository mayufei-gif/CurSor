#!/usr/bin/env python3
"""
PDF Analysis Tools

Implements comprehensive PDF analysis functionality including document type detection,
structure analysis, content classification, and intelligent processing recommendations.

Author: PDF-MCP Team
License: MIT
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
import re
from collections import Counter

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

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class AnalyzePDFTool(PDFTool):
    """Comprehensive PDF analysis tool."""
    
    def __init__(self):
        super().__init__(
            name="analyze_pdf",
            description="Perform comprehensive analysis of PDF documents including structure, content, and processing recommendations"
        )
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return self.get_input_schema()
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "analysis_result": {
                    "type": "object",
                    "description": "Comprehensive PDF analysis results"
                },
                "processing_recommendations": {
                    "type": "array",
                    "description": "Recommended processing methods"
                }
            }
        }
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to analyze"
                },
                "analysis_depth": {
                    "type": "string",
                    "enum": ["basic", "detailed", "comprehensive"],
                    "default": "detailed",
                    "description": "Depth of analysis to perform"
                },
                "include_content_analysis": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include detailed content analysis (text, images, tables)"
                },
                "include_structure_analysis": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include document structure analysis (headings, sections, layout)"
                },
                "include_quality_assessment": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include quality assessment (OCR needs, image quality, etc.)"
                },
                "include_processing_recommendations": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include processing method recommendations"
                },
                "sample_pages": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of pages to sample for detailed analysis (0 = all pages)"
                },
                "extract_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract and analyze document metadata"
                },
                "detect_languages": {
                    "type": "boolean",
                    "default": True,
                    "description": "Detect languages used in the document"
                },
                "analyze_fonts": {
                    "type": "boolean",
                    "default": True,
                    "description": "Analyze fonts and typography"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        analysis_depth = kwargs.get("analysis_depth", "detailed")
        include_content_analysis = kwargs.get("include_content_analysis", True)
        include_structure_analysis = kwargs.get("include_structure_analysis", True)
        include_quality_assessment = kwargs.get("include_quality_assessment", True)
        include_processing_recommendations = kwargs.get("include_processing_recommendations", True)
        sample_pages = kwargs.get("sample_pages", 5)
        extract_metadata = kwargs.get("extract_metadata", True)
        detect_languages = kwargs.get("detect_languages", True)
        analyze_fonts = kwargs.get("analyze_fonts", True)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            start_time = datetime.now()
            
            # Initialize analysis results
            analysis_results = {
                "file_info": await self._get_file_info(pdf_path),
                "analysis_timestamp": start_time.isoformat(),
                "analysis_depth": analysis_depth
            }
            
            if not fitz:
                raise ToolExecutionException("PyMuPDF not available")
            
            # Open PDF
            pdf_doc = fitz.open(str(pdf_path))
            
            try:
                # Basic document information
                analysis_results["document_info"] = await self._analyze_document_info(pdf_doc)
                
                # Metadata analysis
                if extract_metadata:
                    analysis_results["metadata"] = await self._analyze_metadata(pdf_doc)
                
                # Page analysis
                analysis_results["page_analysis"] = await self._analyze_pages(
                    pdf_doc, sample_pages, analysis_depth
                )
                
                # Content analysis
                if include_content_analysis:
                    analysis_results["content_analysis"] = await self._analyze_content(
                        pdf_doc, sample_pages, detect_languages, analyze_fonts
                    )
                
                # Structure analysis
                if include_structure_analysis:
                    analysis_results["structure_analysis"] = await self._analyze_structure(
                        pdf_doc, sample_pages
                    )
                
                # Quality assessment
                if include_quality_assessment:
                    analysis_results["quality_assessment"] = await self._assess_quality(
                        pdf_doc, sample_pages
                    )
                
                # Processing recommendations
                if include_processing_recommendations:
                    analysis_results["processing_recommendations"] = await self._generate_recommendations(
                        analysis_results
                    )
            
            finally:
                pdf_doc.close()
            
            end_time = datetime.now()
            analysis_results["processing_time"] = (end_time - start_time).total_seconds()
            
            # Format output
            content = []
            
            # Add executive summary
            summary = await self._create_executive_summary(analysis_results)
            content.append(create_text_content(f"PDF Analysis Executive Summary:\n{summary}"))
            
            # Add detailed results
            if analysis_depth in ["detailed", "comprehensive"]:
                content.append(create_text_content(
                    f"Detailed Analysis Results:\n```json\n{json.dumps(analysis_results, indent=2, ensure_ascii=False)}\n```"
                ))
            
            # Add processing recommendations
            if include_processing_recommendations and "processing_recommendations" in analysis_results:
                recommendations = analysis_results["processing_recommendations"]
                rec_text = await self._format_recommendations(recommendations)
                content.append(create_text_content(f"Processing Recommendations:\n{rec_text}"))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"PDF analysis failed: {e}")
            content = [create_error_content(f"PDF analysis failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    async def _get_file_info(self, pdf_path: Path) -> Dict[str, Any]:
        """Get basic file information."""
        stat = pdf_path.stat()
        
        # Calculate file hash for uniqueness
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        return {
            "path": str(pdf_path),
            "name": pdf_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "md5_hash": file_hash
        }
    
    async def _analyze_document_info(self, pdf_doc) -> Dict[str, Any]:
        """Analyze basic document information."""
        return {
            "page_count": len(pdf_doc),
            "is_encrypted": pdf_doc.needs_pass,
            "is_pdf_a": pdf_doc.is_pdf,
            "version": pdf_doc.pdf_version(),
            "page_mode": pdf_doc.page_mode,
            "page_layout": pdf_doc.page_layout,
            "permissions": {
                "print": pdf_doc.permissions & fitz.PDF_PERM_PRINT != 0,
                "modify": pdf_doc.permissions & fitz.PDF_PERM_MODIFY != 0,
                "copy": pdf_doc.permissions & fitz.PDF_PERM_COPY != 0,
                "annotate": pdf_doc.permissions & fitz.PDF_PERM_ANNOTATE != 0
            } if hasattr(pdf_doc, 'permissions') else None
        }
    
    async def _analyze_metadata(self, pdf_doc) -> Dict[str, Any]:
        """Analyze document metadata."""
        metadata = pdf_doc.metadata
        
        # Clean and structure metadata
        structured_metadata = {}
        for key, value in metadata.items():
            if value and value.strip():
                structured_metadata[key.lower()] = value.strip()
        
        # Extract additional information
        creation_date = metadata.get('creationDate', '')
        modification_date = metadata.get('modDate', '')
        
        return {
            "raw_metadata": structured_metadata,
            "title": metadata.get('title', ''),
            "author": metadata.get('author', ''),
            "subject": metadata.get('subject', ''),
            "creator": metadata.get('creator', ''),
            "producer": metadata.get('producer', ''),
            "creation_date": creation_date,
            "modification_date": modification_date,
            "keywords": metadata.get('keywords', ''),
            "has_metadata": bool(any(v for v in metadata.values() if v and v.strip()))
        }
    
    async def _analyze_pages(self, pdf_doc, sample_pages: int, analysis_depth: str) -> Dict[str, Any]:
        """Analyze page characteristics."""
        total_pages = len(pdf_doc)
        
        # Determine which pages to analyze
        if sample_pages == 0 or sample_pages >= total_pages:
            page_indices = list(range(total_pages))
        else:
            # Sample pages evenly distributed
            step = max(1, total_pages // sample_pages)
            page_indices = list(range(0, total_pages, step))[:sample_pages]
        
        page_info = []
        page_sizes = []
        orientations = []
        
        for page_idx in page_indices:
            page = pdf_doc[page_idx]
            rect = page.rect
            
            width = rect.width
            height = rect.height
            orientation = "portrait" if height > width else "landscape" if width > height else "square"
            
            page_data = {
                "page_number": page_idx + 1,
                "width": width,
                "height": height,
                "orientation": orientation,
                "rotation": page.rotation
            }
            
            if analysis_depth == "comprehensive":
                # Add more detailed page analysis
                page_data.update({
                    "has_text": bool(page.get_text().strip()),
                    "has_images": len(page.get_images()) > 0,
                    "has_drawings": len(page.get_drawings()) > 0,
                    "text_length": len(page.get_text()),
                    "image_count": len(page.get_images()),
                    "drawing_count": len(page.get_drawings())
                })
            
            page_info.append(page_data)
            page_sizes.append((width, height))
            orientations.append(orientation)
        
        # Calculate statistics
        orientation_counts = Counter(orientations)
        
        return {
            "total_pages": total_pages,
            "analyzed_pages": len(page_indices),
            "sample_pages": page_info,
            "page_size_statistics": {
                "unique_sizes": len(set(page_sizes)),
                "most_common_size": Counter(page_sizes).most_common(1)[0] if page_sizes else None,
                "uniform_size": len(set(page_sizes)) == 1
            },
            "orientation_statistics": dict(orientation_counts),
            "predominant_orientation": orientation_counts.most_common(1)[0][0] if orientations else None
        }
    
    async def _analyze_content(self, pdf_doc, sample_pages: int, detect_languages: bool, analyze_fonts: bool) -> Dict[str, Any]:
        """Analyze document content."""
        total_pages = len(pdf_doc)
        
        # Determine which pages to analyze
        if sample_pages == 0 or sample_pages >= total_pages:
            page_indices = list(range(total_pages))
        else:
            step = max(1, total_pages // sample_pages)
            page_indices = list(range(0, total_pages, step))[:sample_pages]
        
        all_text = ""
        total_text_length = 0
        total_images = 0
        total_drawings = 0
        fonts_used = set()
        
        content_types = {
            "text_pages": 0,
            "image_pages": 0,
            "mixed_pages": 0,
            "empty_pages": 0
        }
        
        for page_idx in page_indices:
            page = pdf_doc[page_idx]
            
            # Text analysis
            page_text = page.get_text()
            text_length = len(page_text.strip())
            total_text_length += text_length
            
            if text_length > 100:  # Significant text content
                all_text += page_text + "\n"
            
            # Image analysis
            images = page.get_images()
            total_images += len(images)
            
            # Drawing analysis
            drawings = page.get_drawings()
            total_drawings += len(drawings)
            
            # Font analysis
            if analyze_fonts:
                try:
                    text_dict = page.get_text("dict")
                    for block in text_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    font_name = span.get("font", "")
                                    if font_name:
                                        fonts_used.add(font_name)
                except Exception:
                    pass
            
            # Categorize page content
            has_text = text_length > 50
            has_images = len(images) > 0
            has_drawings = len(drawings) > 0
            
            if has_text and (has_images or has_drawings):
                content_types["mixed_pages"] += 1
            elif has_text:
                content_types["text_pages"] += 1
            elif has_images or has_drawings:
                content_types["image_pages"] += 1
            else:
                content_types["empty_pages"] += 1
        
        # Text analysis
        text_analysis = await self._analyze_text_content(all_text, detect_languages)
        
        return {
            "text_analysis": text_analysis,
            "content_statistics": {
                "total_text_length": total_text_length,
                "average_text_per_page": total_text_length / len(page_indices) if page_indices else 0,
                "total_images": total_images,
                "total_drawings": total_drawings,
                "content_distribution": content_types
            },
            "font_analysis": {
                "fonts_used": list(fonts_used),
                "font_count": len(fonts_used),
                "has_embedded_fonts": len(fonts_used) > 0
            } if analyze_fonts else None
        }
    
    async def _analyze_text_content(self, text: str, detect_languages: bool) -> Dict[str, Any]:
        """Analyze text content characteristics."""
        if not text.strip():
            return {
                "has_text": False,
                "character_count": 0,
                "word_count": 0,
                "line_count": 0
            }
        
        # Basic text statistics
        lines = text.split('\n')
        words = text.split()
        
        # Character analysis
        char_counts = {
            "total": len(text),
            "alphabetic": sum(1 for c in text if c.isalpha()),
            "numeric": sum(1 for c in text if c.isdigit()),
            "whitespace": sum(1 for c in text if c.isspace()),
            "punctuation": sum(1 for c in text if not c.isalnum() and not c.isspace())
        }
        
        # Mathematical content detection
        math_indicators = {
            "has_equations": bool(re.search(r'[=≠≈≤≥<>]', text)),
            "has_formulas": bool(re.search(r'[∫∑∏√∞±α-ωΑ-Ω]', text)),
            "has_superscripts": bool(re.search(r'\^[0-9a-zA-Z]', text)),
            "has_subscripts": bool(re.search(r'_[0-9a-zA-Z]', text)),
            "has_fractions": bool(re.search(r'\d+/\d+', text))
        }
        
        # Table indicators
        table_indicators = {
            "has_tabular_data": bool(re.search(r'\t.*\t', text)),
            "has_aligned_numbers": bool(re.search(r'\d+\.\d+\s+\d+\.\d+', text)),
            "has_column_headers": bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+', text))
        }
        
        analysis = {
            "has_text": True,
            "character_count": char_counts["total"],
            "word_count": len(words),
            "line_count": len(lines),
            "character_distribution": char_counts,
            "mathematical_content": math_indicators,
            "table_indicators": table_indicators,
            "text_density": char_counts["alphabetic"] / char_counts["total"] if char_counts["total"] > 0 else 0
        }
        
        # Language detection (simplified)
        if detect_languages:
            analysis["language_analysis"] = await self._detect_languages(text)
        
        return analysis
    
    async def _detect_languages(self, text: str) -> Dict[str, Any]:
        """Detect languages in the text (simplified implementation)."""
        # This is a simplified language detection
        # In practice, you would use a library like langdetect or polyglot
        
        # Basic character set analysis
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_cyrillic = bool(re.search(r'[а-яё]', text, re.IGNORECASE))
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        has_arabic = bool(re.search(r'[\u0600-\u06ff]', text))
        has_japanese = bool(re.search(r'[ひらがなカタカナ]', text))
        
        detected_scripts = []
        if has_latin:
            detected_scripts.append("Latin")
        if has_cyrillic:
            detected_scripts.append("Cyrillic")
        if has_chinese:
            detected_scripts.append("Chinese")
        if has_arabic:
            detected_scripts.append("Arabic")
        if has_japanese:
            detected_scripts.append("Japanese")
        
        # Simple language guessing based on common words
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        english_score = sum(1 for word in english_words if word.lower() in text.lower())
        
        return {
            "detected_scripts": detected_scripts,
            "is_multilingual": len(detected_scripts) > 1,
            "primary_script": detected_scripts[0] if detected_scripts else "Unknown",
            "english_likelihood": min(english_score / 10, 1.0) if has_latin else 0
        }
    
    async def _analyze_structure(self, pdf_doc, sample_pages: int) -> Dict[str, Any]:
        """Analyze document structure."""
        total_pages = len(pdf_doc)
        
        # Determine which pages to analyze
        if sample_pages == 0 or sample_pages >= total_pages:
            page_indices = list(range(total_pages))
        else:
            step = max(1, total_pages // sample_pages)
            page_indices = list(range(0, total_pages, step))[:sample_pages]
        
        # Analyze text structure
        headings = []
        paragraphs = []
        lists = []
        
        for page_idx in page_indices:
            page = pdf_doc[page_idx]
            
            try:
                # Get text with formatting information
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        block_text = ""
                        font_sizes = []
                        
                        for line in block["lines"]:
                            line_text = ""
                            for span in line.get("spans", []):
                                span_text = span.get("text", "")
                                font_size = span.get("size", 0)
                                line_text += span_text
                                font_sizes.append(font_size)
                            
                            block_text += line_text + "\n"
                        
                        block_text = block_text.strip()
                        if block_text:
                            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                            
                            # Classify text blocks
                            if self._is_heading(block_text, avg_font_size):
                                headings.append({
                                    "text": block_text,
                                    "page": page_idx + 1,
                                    "font_size": avg_font_size,
                                    "level": self._estimate_heading_level(avg_font_size)
                                })
                            elif self._is_list_item(block_text):
                                lists.append({
                                    "text": block_text,
                                    "page": page_idx + 1,
                                    "type": "bullet" if block_text.strip().startswith(('•', '-', '*')) else "numbered"
                                })
                            else:
                                paragraphs.append({
                                    "text": block_text[:100] + "..." if len(block_text) > 100 else block_text,
                                    "page": page_idx + 1,
                                    "length": len(block_text)
                                })
            
            except Exception as e:
                self.logger.warning(f"Structure analysis failed for page {page_idx + 1}: {e}")
                continue
        
        return {
            "headings": {
                "count": len(headings),
                "items": headings[:10],  # Limit to first 10
                "has_hierarchy": len(set(h["level"] for h in headings)) > 1
            },
            "paragraphs": {
                "count": len(paragraphs),
                "average_length": sum(p["length"] for p in paragraphs) / len(paragraphs) if paragraphs else 0
            },
            "lists": {
                "count": len(lists),
                "bullet_lists": sum(1 for l in lists if l["type"] == "bullet"),
                "numbered_lists": sum(1 for l in lists if l["type"] == "numbered")
            },
            "structure_quality": {
                "has_headings": len(headings) > 0,
                "has_lists": len(lists) > 0,
                "well_structured": len(headings) > 0 and len(paragraphs) > 0
            }
        }
    
    def _is_heading(self, text: str, font_size: float) -> bool:
        """Determine if text is likely a heading."""
        # Simple heuristics for heading detection
        text = text.strip()
        
        # Check length (headings are usually shorter)
        if len(text) > 200:
            return False
        
        # Check if it's all caps (common for headings)
        if text.isupper() and len(text) > 3:
            return True
        
        # Check if it starts with a number (like "1. Introduction")
        if re.match(r'^\d+\.?\s+[A-Z]', text):
            return True
        
        # Check if it's a single line with title case
        if '\n' not in text and text.istitle() and len(text.split()) <= 10:
            return True
        
        # Check font size (if significantly larger than average)
        if font_size > 14:  # Arbitrary threshold
            return True
        
        return False
    
    def _estimate_heading_level(self, font_size: float) -> int:
        """Estimate heading level based on font size."""
        if font_size >= 20:
            return 1
        elif font_size >= 16:
            return 2
        elif font_size >= 14:
            return 3
        else:
            return 4
    
    def _is_list_item(self, text: str) -> bool:
        """Determine if text is likely a list item."""
        text = text.strip()
        
        # Check for bullet points
        if text.startswith(('•', '-', '*', '◦', '▪', '▫')):
            return True
        
        # Check for numbered lists
        if re.match(r'^\d+[.)\s]', text):
            return True
        
        # Check for lettered lists
        if re.match(r'^[a-zA-Z][.)\s]', text):
            return True
        
        return False
    
    async def _assess_quality(self, pdf_doc, sample_pages: int) -> Dict[str, Any]:
        """Assess document quality for processing."""
        total_pages = len(pdf_doc)
        
        # Determine which pages to analyze
        if sample_pages == 0 or sample_pages >= total_pages:
            page_indices = list(range(total_pages))
        else:
            step = max(1, total_pages // sample_pages)
            page_indices = list(range(0, total_pages, step))[:sample_pages]
        
        quality_metrics = {
            "text_extractability": [],
            "image_quality": [],
            "ocr_needed": [],
            "scanned_pages": 0,
            "native_text_pages": 0
        }
        
        for page_idx in page_indices:
            page = pdf_doc[page_idx]
            
            # Check text extractability
            text = page.get_text()
            text_length = len(text.strip())
            
            # Check if page is likely scanned
            images = page.get_images()
            has_large_images = any(img for img in images if self._is_large_image(page, img))
            
            if text_length < 50 and has_large_images:
                # Likely a scanned page
                quality_metrics["scanned_pages"] += 1
                quality_metrics["ocr_needed"].append(page_idx + 1)
                quality_metrics["text_extractability"].append("poor")
            elif text_length > 100:
                # Good text extraction
                quality_metrics["native_text_pages"] += 1
                quality_metrics["text_extractability"].append("good")
            else:
                # Moderate text extraction
                quality_metrics["text_extractability"].append("moderate")
            
            # Assess image quality (simplified)
            if images:
                # This is a simplified assessment
                # In practice, you would analyze actual image data
                quality_metrics["image_quality"].append("unknown")
        
        # Calculate overall quality scores
        text_extractability_score = quality_metrics["text_extractability"].count("good") / len(page_indices) if page_indices else 0
        scanned_ratio = quality_metrics["scanned_pages"] / len(page_indices) if page_indices else 0
        
        return {
            "text_extractability": {
                "score": text_extractability_score,
                "distribution": Counter(quality_metrics["text_extractability"]),
                "needs_ocr": scanned_ratio > 0.5
            },
            "document_type": {
                "is_scanned": scanned_ratio > 0.7,
                "is_native": scanned_ratio < 0.3,
                "is_mixed": 0.3 <= scanned_ratio <= 0.7,
                "scanned_page_ratio": scanned_ratio
            },
            "processing_complexity": {
                "low": text_extractability_score > 0.8,
                "medium": 0.4 <= text_extractability_score <= 0.8,
                "high": text_extractability_score < 0.4
            },
            "pages_needing_ocr": quality_metrics["ocr_needed"]
        }
    
    def _is_large_image(self, page, img_ref) -> bool:
        """Check if an image is large (likely a scanned page)."""
        try:
            # Get image dimensions
            img_dict = page.get_image_info(img_ref[0])
            if img_dict:
                width = img_dict.get("width", 0)
                height = img_dict.get("height", 0)
                
                # Consider image large if it covers significant portion of page
                page_area = page.rect.width * page.rect.height
                img_area = width * height
                
                return img_area > (page_area * 0.5)  # More than 50% of page
        except:
            pass
        
        return False
    
    async def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing recommendations based on analysis."""
        recommendations = {
            "primary_methods": [],
            "secondary_methods": [],
            "preprocessing_steps": [],
            "quality_improvements": [],
            "processing_order": []
        }
        
        # Get key metrics
        quality_assessment = analysis_results.get("quality_assessment", {})
        content_analysis = analysis_results.get("content_analysis", {})
        structure_analysis = analysis_results.get("structure_analysis", {})
        
        # Text extraction recommendations
        text_extractability = quality_assessment.get("text_extractability", {})
        if text_extractability.get("score", 0) > 0.8:
            recommendations["primary_methods"].append("text_extraction")
            recommendations["processing_order"].append("1. Extract text using native PDF text")
        elif text_extractability.get("needs_ocr", False):
            recommendations["primary_methods"].append("ocr_processing")
            recommendations["processing_order"].append("1. Apply OCR to extract text from scanned pages")
            recommendations["preprocessing_steps"].append("Image preprocessing for better OCR accuracy")
        
        # Table extraction recommendations
        table_indicators = content_analysis.get("text_analysis", {}).get("table_indicators", {})
        if any(table_indicators.values()):
            recommendations["secondary_methods"].append("table_extraction")
            recommendations["processing_order"].append("2. Extract tables using specialized tools")
        
        # Formula extraction recommendations
        math_content = content_analysis.get("text_analysis", {}).get("mathematical_content", {})
        if any(math_content.values()):
            recommendations["secondary_methods"].append("formula_recognition")
            recommendations["processing_order"].append("3. Extract mathematical formulas")
        
        # Structure analysis recommendations
        structure_quality = structure_analysis.get("structure_quality", {})
        if structure_quality.get("well_structured", False):
            recommendations["quality_improvements"].append("Document has good structure - preserve hierarchy")
        else:
            recommendations["quality_improvements"].append("Consider structure enhancement during processing")
        
        # Document type specific recommendations
        doc_type = quality_assessment.get("document_type", {})
        if doc_type.get("is_scanned", False):
            recommendations["preprocessing_steps"].extend([
                "Deskew and straighten pages",
                "Enhance image contrast and resolution",
                "Remove noise and artifacts"
            ])
        
        # Processing complexity recommendations
        complexity = quality_assessment.get("processing_complexity", {})
        if complexity.get("high", False):
            recommendations["quality_improvements"].append("High complexity document - consider manual review")
        elif complexity.get("low", False):
            recommendations["quality_improvements"].append("Low complexity - automated processing recommended")
        
        # Final processing order
        if not recommendations["processing_order"]:
            recommendations["processing_order"] = ["1. Start with text extraction", "2. Analyze content for specialized processing needs"]
        
        return recommendations
    
    async def _create_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Create an executive summary of the analysis."""
        file_info = analysis_results.get("file_info", {})
        doc_info = analysis_results.get("document_info", {})
        content_analysis = analysis_results.get("content_analysis", {})
        quality_assessment = analysis_results.get("quality_assessment", {})
        
        summary_parts = []
        
        # Basic document info
        summary_parts.append(f"**Document:** {file_info.get('name', 'Unknown')}")
        summary_parts.append(f"**Size:** {file_info.get('size_mb', 0):.1f} MB, {doc_info.get('page_count', 0)} pages")
        
        # Document type
        doc_type = quality_assessment.get("document_type", {})
        if doc_type.get("is_scanned", False):
            summary_parts.append("**Type:** Scanned document (OCR required)")
        elif doc_type.get("is_native", False):
            summary_parts.append("**Type:** Native PDF (text extractable)")
        else:
            summary_parts.append("**Type:** Mixed content document")
        
        # Content summary
        text_stats = content_analysis.get("content_statistics", {})
        content_dist = text_stats.get("content_distribution", {})
        
        if content_dist:
            content_summary = []
            if content_dist.get("text_pages", 0) > 0:
                content_summary.append(f"{content_dist['text_pages']} text pages")
            if content_dist.get("image_pages", 0) > 0:
                content_summary.append(f"{content_dist['image_pages']} image pages")
            if content_dist.get("mixed_pages", 0) > 0:
                content_summary.append(f"{content_dist['mixed_pages']} mixed pages")
            
            if content_summary:
                summary_parts.append(f"**Content:** {', '.join(content_summary)}")
        
        # Processing recommendations
        recommendations = analysis_results.get("processing_recommendations", {})
        primary_methods = recommendations.get("primary_methods", [])
        if primary_methods:
            summary_parts.append(f"**Recommended Processing:** {', '.join(primary_methods)}")
        
        # Quality assessment
        text_extractability = quality_assessment.get("text_extractability", {})
        score = text_extractability.get("score", 0)
        if score > 0.8:
            summary_parts.append("**Quality:** Excellent text extractability")
        elif score > 0.5:
            summary_parts.append("**Quality:** Good text extractability")
        else:
            summary_parts.append("**Quality:** Poor text extractability - OCR recommended")
        
        return "\n".join(summary_parts)
    
    async def _format_recommendations(self, recommendations: Dict[str, Any]) -> str:
        """Format processing recommendations for display."""
        formatted = []
        
        # Primary methods
        primary = recommendations.get("primary_methods", [])
        if primary:
            formatted.append(f"**Primary Processing Methods:**\n- {chr(10).join(primary)}")
        
        # Secondary methods
        secondary = recommendations.get("secondary_methods", [])
        if secondary:
            formatted.append(f"**Secondary Processing Methods:**\n- {chr(10).join(secondary)}")
        
        # Preprocessing steps
        preprocessing = recommendations.get("preprocessing_steps", [])
        if preprocessing:
            formatted.append(f"**Preprocessing Steps:**\n- {chr(10).join(preprocessing)}")
        
        # Processing order
        order = recommendations.get("processing_order", [])
        if order:
            formatted.append(f"**Recommended Processing Order:**\n{chr(10).join(order)}")
        
        # Quality improvements
        quality = recommendations.get("quality_improvements", [])
        if quality:
            formatted.append(f"**Quality Considerations:**\n- {chr(10).join(quality)}")
        
        return "\n\n".join(formatted)


class DetectPDFTypeTool(PDFTool):
    """Detect PDF document type and characteristics."""
    
    def __init__(self):
        super().__init__(
            name="detect_pdf_type",
            description="Detect PDF document type, content characteristics, and optimal processing methods"
        )
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return self.get_input_schema()
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "Detected document type"
                },
                "confidence_scores": {
                    "type": "object",
                    "description": "Confidence scores for type detection"
                }
            }
        }
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to analyze"
                },
                "quick_analysis": {
                    "type": "boolean",
                    "default": True,
                    "description": "Perform quick analysis (first few pages only)"
                },
                "sample_pages": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of pages to sample for quick analysis"
                },
                "include_confidence": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include confidence scores for type detection"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        quick_analysis = kwargs.get("quick_analysis", True)
        sample_pages = kwargs.get("sample_pages", 3)
        include_confidence = kwargs.get("include_confidence", True)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            if not fitz:
                raise ToolExecutionException("PyMuPDF not available")
            
            # Open PDF
            pdf_doc = fitz.open(str(pdf_path))
            
            try:
                # Perform type detection
                detection_results = await self._detect_document_type(
                    pdf_doc, quick_analysis, sample_pages, include_confidence
                )
                
                # Format output
                content = []
                
                # Add type detection summary
                summary = self._create_type_summary(detection_results)
                content.append(create_text_content(f"PDF Type Detection Summary:\n{summary}"))
                
                # Add detailed results if requested
                if include_confidence:
                    content.append(create_text_content(
                        f"Detailed Detection Results:\n```json\n{json.dumps(detection_results, indent=2, ensure_ascii=False)}\n```"
                    ))
                
                # Add processing recommendations
                recommendations = self._get_type_based_recommendations(detection_results)
                content.append(create_text_content(f"Processing Recommendations:\n{recommendations}"))
                
                return MCPToolResult(content=content)
            
            finally:
                pdf_doc.close()
        
        except Exception as e:
            self.logger.error(f"PDF type detection failed: {e}")
            content = [create_error_content(f"PDF type detection failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    async def _detect_document_type(
        self,
        pdf_doc,
        quick_analysis: bool,
        sample_pages: int,
        include_confidence: bool
    ) -> Dict[str, Any]:
        """Detect document type and characteristics."""
        total_pages = len(pdf_doc)
        
        # Determine pages to analyze
        if quick_analysis:
            page_indices = list(range(min(sample_pages, total_pages)))
        else:
            page_indices = list(range(total_pages))
        
        # Initialize detection metrics
        metrics = {
            "text_pages": 0,
            "image_pages": 0,
            "mixed_pages": 0,
            "empty_pages": 0,
            "total_text_length": 0,
            "total_images": 0,
            "has_tables": False,
            "has_formulas": False,
            "has_forms": False,
            "has_annotations": False,
            "font_diversity": 0,
            "page_sizes": [],
            "scanned_indicators": 0
        }
        
        fonts_used = set()
        
        for page_idx in page_indices:
            page = pdf_doc[page_idx]
            
            # Text analysis
            text = page.get_text()
            text_length = len(text.strip())
            metrics["total_text_length"] += text_length
            
            # Image analysis
            images = page.get_images()
            image_count = len(images)
            metrics["total_images"] += image_count
            
            # Check for large images (scanned page indicator)
            has_large_image = any(self._is_large_image(page, img) for img in images)
            if has_large_image and text_length < 100:
                metrics["scanned_indicators"] += 1
            
            # Page classification
            if text_length > 100 and image_count > 0:
                metrics["mixed_pages"] += 1
            elif text_length > 100:
                metrics["text_pages"] += 1
            elif image_count > 0:
                metrics["image_pages"] += 1
            else:
                metrics["empty_pages"] += 1
            
            # Content analysis
            if text:
                # Table detection
                if not metrics["has_tables"]:
                    metrics["has_tables"] = self._detect_tables_in_text(text)
                
                # Formula detection
                if not metrics["has_formulas"]:
                    metrics["has_formulas"] = self._detect_formulas_in_text(text)
            
            # Form detection
            if not metrics["has_forms"]:
                widgets = page.widgets()
                metrics["has_forms"] = len(widgets) > 0
            
            # Annotation detection
            if not metrics["has_annotations"]:
                annotations = page.annots()
                metrics["has_annotations"] = len(list(annotations)) > 0
            
            # Font analysis
            try:
                text_dict = page.get_text("dict")
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                font_name = span.get("font", "")
                                if font_name:
                                    fonts_used.add(font_name)
            except Exception:
                pass
            
            # Page size
            rect = page.rect
            metrics["page_sizes"].append((rect.width, rect.height))
        
        metrics["font_diversity"] = len(fonts_used)
        
        # Determine document type
        type_detection = self._classify_document_type(metrics, len(page_indices))
        
        # Calculate confidence scores if requested
        if include_confidence:
            confidence_scores = self._calculate_confidence_scores(metrics, len(page_indices))
            type_detection["confidence_scores"] = confidence_scores
        
        return {
            "document_type": type_detection,
            "content_metrics": metrics,
            "analysis_scope": {
                "total_pages": total_pages,
                "analyzed_pages": len(page_indices),
                "quick_analysis": quick_analysis
            }
        }
    
    def _is_large_image(self, page, img_ref) -> bool:
        """Check if an image is large (likely a scanned page)."""
        try:
            # Get image dimensions
            img_dict = page.get_image_info(img_ref[0])
            if img_dict:
                width = img_dict.get("width", 0)
                height = img_dict.get("height", 0)
                
                # Consider image large if it covers significant portion of page
                page_area = page.rect.width * page.rect.height
                img_area = width * height
                
                return img_area > (page_area * 0.5)  # More than 50% of page
        except:
            pass
        
        return False
    
    def _detect_tables_in_text(self, text: str) -> bool:
        """Detect if text contains tabular data."""
        # Simple table detection heuristics
        lines = text.split('\n')
        
        # Check for tab-separated data
        tab_lines = sum(1 for line in lines if '\t' in line and line.count('\t') >= 2)
        if tab_lines >= 3:  # At least 3 lines with tabs
            return True
        
        # Check for aligned numeric data
        numeric_pattern = r'\d+\.\d+\s+\d+\.\d+'
        if len(re.findall(numeric_pattern, text)) >= 3:
            return True
        
        # Check for common table headers
        table_headers = ['name', 'value', 'date', 'amount', 'total', 'description', 'type']
        header_count = sum(1 for header in table_headers if header.lower() in text.lower())
        if header_count >= 2:
            return True
        
        return False
    
    def _detect_formulas_in_text(self, text: str) -> bool:
        """Detect if text contains mathematical formulas."""
        # Mathematical symbols
        math_symbols = ['∫', '∑', '∏', '√', '∞', '±', '≤', '≥', '≠', '≈', '∝']
        if any(symbol in text for symbol in math_symbols):
            return True
        
        # Greek letters (common in formulas)
        greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω']
        if any(letter in text for letter in greek_letters):
            return True
        
        # LaTeX-style patterns
        latex_patterns = [r'\\[a-zA-Z]+', r'\^[0-9a-zA-Z]', r'_[0-9a-zA-Z]', r'\{[^}]*\}']
        if any(re.search(pattern, text) for pattern in latex_patterns):
            return True
        
        # Mathematical expressions
        math_expressions = [r'\d+/\d+', r'[a-zA-Z]\^\d+', r'[a-zA-Z]_\d+']
        if any(re.search(expr, text) for expr in math_expressions):
            return True
        
        return False
    
    def _classify_document_type(self, metrics: Dict[str, Any], analyzed_pages: int) -> Dict[str, Any]:
        """Classify document type based on metrics."""
        if analyzed_pages == 0:
            return {"primary_type": "unknown", "characteristics": []}
        
        # Calculate ratios
        scanned_ratio = metrics["scanned_indicators"] / analyzed_pages
        text_ratio = metrics["text_pages"] / analyzed_pages
        image_ratio = metrics["image_pages"] / analyzed_pages
        mixed_ratio = metrics["mixed_pages"] / analyzed_pages
        
        # Primary type classification
        if scanned_ratio > 0.7:
            primary_type = "scanned_document"
        elif text_ratio > 0.8:
            primary_type = "text_document"
        elif image_ratio > 0.6:
            primary_type = "image_document"
        elif mixed_ratio > 0.5:
            primary_type = "mixed_content"
        else:
            primary_type = "hybrid_document"
        
        # Secondary characteristics
        characteristics = []
        
        if metrics["has_tables"]:
            characteristics.append("contains_tables")
        
        if metrics["has_formulas"]:
            characteristics.append("contains_formulas")
        
        if metrics["has_forms"]:
            characteristics.append("interactive_forms")
        
        if metrics["has_annotations"]:
            characteristics.append("has_annotations")
        
        if metrics["font_diversity"] > 10:
            characteristics.append("high_font_diversity")
        elif metrics["font_diversity"] < 3:
            characteristics.append("low_font_diversity")
        
        # Page uniformity
        unique_sizes = len(set(metrics["page_sizes"]))
        if unique_sizes == 1:
            characteristics.append("uniform_page_size")
        elif unique_sizes > analyzed_pages * 0.5:
            characteristics.append("varied_page_sizes")
        
        # Content density
        avg_text_per_page = metrics["total_text_length"] / analyzed_pages if analyzed_pages > 0 else 0
        if avg_text_per_page > 2000:
            characteristics.append("high_text_density")
        elif avg_text_per_page < 500:
            characteristics.append("low_text_density")
        
        return {
            "primary_type": primary_type,
            "characteristics": characteristics,
            "content_ratios": {
                "text_pages": text_ratio,
                "image_pages": image_ratio,
                "mixed_pages": mixed_ratio,
                "scanned_pages": scanned_ratio
            }
        }
    
    def _calculate_confidence_scores(self, metrics: Dict[str, Any], analyzed_pages: int) -> Dict[str, float]:
        """Calculate confidence scores for type detection."""
        if analyzed_pages == 0:
            return {"overall": 0.0}
        
        scores = {}
        
        # Text document confidence
        text_ratio = metrics["text_pages"] / analyzed_pages
        avg_text_length = metrics["total_text_length"] / analyzed_pages
        scores["text_document"] = min(text_ratio + (avg_text_length / 5000), 1.0)
        
        # Scanned document confidence
        scanned_ratio = metrics["scanned_indicators"] / analyzed_pages
        image_ratio = metrics["image_pages"] / analyzed_pages
        scores["scanned_document"] = min(scanned_ratio + (image_ratio * 0.5), 1.0)
        
        # Mixed content confidence
        mixed_ratio = metrics["mixed_pages"] / analyzed_pages
        scores["mixed_content"] = mixed_ratio
        
        # Overall confidence (based on strongest indicator)
        scores["overall"] = max(scores.values())
        
        return scores
    
    def _create_type_summary(self, detection_results: Dict[str, Any]) -> str:
        """Create a summary of type detection results."""
        doc_type = detection_results.get("document_type", {})
        metrics = detection_results.get("content_metrics", {})
        analysis_scope = detection_results.get("analysis_scope", {})
        
        summary_parts = []
        
        # Primary type
        primary_type = doc_type.get("primary_type", "unknown")
        summary_parts.append(f"**Primary Type:** {primary_type.replace('_', ' ').title()}")
        
        # Characteristics
        characteristics = doc_type.get("characteristics", [])
        if characteristics:
            char_list = [char.replace('_', ' ').title() for char in characteristics]
            summary_parts.append(f"**Characteristics:** {', '.join(char_list)}")
        
        # Content distribution
        ratios = doc_type.get("content_ratios", {})
        if ratios:
            distribution = []
            for content_type, ratio in ratios.items():
                if ratio > 0.1:  # Only show significant ratios
                    percentage = int(ratio * 100)
                    distribution.append(f"{percentage}% {content_type.replace('_', ' ')}")
            
            if distribution:
                summary_parts.append(f"**Content Distribution:** {', '.join(distribution)}")
        
        # Analysis scope
        total_pages = analysis_scope.get("total_pages", 0)
        analyzed_pages = analysis_scope.get("analyzed_pages", 0)
        summary_parts.append(f"**Analysis Scope:** {analyzed_pages}/{total_pages} pages")
        
        # Confidence (if available)
        confidence_scores = detection_results.get("document_type", {}).get("confidence_scores", {})
        if confidence_scores:
            overall_confidence = confidence_scores.get("overall", 0)
            summary_parts.append(f"**Confidence:** {overall_confidence:.1%}")
        
        return "\n".join(summary_parts)
    
    def _get_type_based_recommendations(self, detection_results: Dict[str, Any]) -> str:
        """Get processing recommendations based on detected type."""
        doc_type = detection_results.get("document_type", {})
        primary_type = doc_type.get("primary_type", "unknown")
        characteristics = doc_type.get("characteristics", [])
        
        recommendations = []
        
        # Type-specific recommendations
        if primary_type == "scanned_document":
            recommendations.extend([
                "• Use OCR processing for text extraction",
                "• Apply image preprocessing (deskew, enhance contrast)",
                "• Consider manual review for critical content"
            ])
        elif primary_type == "text_document":
            recommendations.extend([
                "• Use native text extraction methods",
                "• Fast processing with high accuracy expected",
                "• Consider structure preservation during extraction"
            ])
        elif primary_type == "mixed_content":
            recommendations.extend([
                "• Use hybrid processing approach",
                "• Extract text from text regions, OCR for image regions",
                "• Maintain content relationships during processing"
            ])
        elif primary_type == "image_document":
            recommendations.extend([
                "• Focus on image processing and OCR",
                "• Consider image enhancement techniques",
                "• May require specialized image analysis tools"
            ])
        
        # Characteristic-specific recommendations
        if "contains_tables" in characteristics:
            recommendations.append("• Use specialized table extraction tools (Camelot, Tabula)")
        
        if "contains_formulas" in characteristics:
            recommendations.append("• Apply mathematical formula recognition (LaTeX-OCR, pix2tex)")
        
        if "interactive_forms" in characteristics:
            recommendations.append("• Extract form fields and values separately")
        
        if "has_annotations" in characteristics:
            recommendations.append("• Preserve annotations and comments during processing")
        
        if "high_font_diversity" in characteristics:
            recommendations.append("• Pay attention to font-based formatting and hierarchy")
        
        if "varied_page_sizes" in characteristics:
            recommendations.append("• Handle variable page layouts carefully")
        
        # Processing order
        recommendations.append("\n**Recommended Processing Order:**")
        if primary_type == "scanned_document":
            recommendations.extend([
                "1. Image preprocessing and enhancement",
                "2. OCR processing with layout detection",
                "3. Post-processing and text cleanup"
            ])
        else:
            recommendations.extend([
                "1. Native text extraction",
                "2. Specialized content extraction (tables, formulas)",
                "3. Structure analysis and formatting"
            ])
        
        return "\n".join(recommendations) if recommendations else "No specific recommendations available."