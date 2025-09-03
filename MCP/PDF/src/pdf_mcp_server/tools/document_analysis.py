#!/usr/bin/env python3
"""
PDF Document Analysis and Type Detection Tools

Implements intelligent document type detection and analysis for PDF files,
including academic papers, financial reports, technical manuals, forms, and general documents.

Author: PDF-MCP Team
License: MIT
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from collections import Counter
import statistics

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..mcp.tools import PDFAnalysisTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class AnalyzeDocumentTool(PDFAnalysisTool):
    """Analyze PDF document structure and determine document type."""
    
    def __init__(self):
        super().__init__(
            name="analyze_document",
            description="Analyze PDF document structure and determine document type for intelligent processing",
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
                "analysis_depth": {
                    "type": "string",
                    "enum": ["basic", "detailed", "comprehensive"],
                    "default": "detailed",
                    "description": "Depth of analysis to perform"
                },
                "sample_pages": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of pages to sample for analysis (for large documents)"
                },
                "detect_language": {
                    "type": "boolean",
                    "default": True,
                    "description": "Detect document language"
                },
                "extract_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract document metadata"
                },
                "analyze_structure": {
                    "type": "boolean",
                    "default": True,
                    "description": "Analyze document structure (headings, sections, etc.)"
                },
                "recommend_tools": {
                    "type": "boolean",
                    "default": True,
                    "description": "Recommend optimal extraction tools based on analysis"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute document analysis."""
        try:
            file_path = Path(kwargs["file_path"])
            analysis_depth = kwargs.get("analysis_depth", "detailed")
            sample_pages = kwargs.get("sample_pages", 5)
            detect_language = kwargs.get("detect_language", True)
            extract_metadata = kwargs.get("extract_metadata", True)
            analyze_structure = kwargs.get("analyze_structure", True)
            recommend_tools = kwargs.get("recommend_tools", True)
            
            if not file_path.exists():
                raise MCPResourceException(f"PDF file not found: {file_path}")
            
            # Initialize analysis result
            analysis = {
                "file_path": str(file_path),
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_depth": analysis_depth
            }
            
            # Extract basic document information
            doc_info = await self._extract_document_info(file_path)
            analysis["document_info"] = doc_info
            
            # Extract metadata if requested
            if extract_metadata:
                metadata = await self._extract_metadata(file_path)
                analysis["metadata"] = metadata
            
            # Analyze document structure
            if analyze_structure:
                structure = await self._analyze_structure(file_path, sample_pages, analysis_depth)
                analysis["structure"] = structure
            
            # Detect document type
            doc_type = await self._detect_document_type(file_path, analysis, sample_pages)
            analysis["document_type"] = doc_type
            
            # Detect language if requested
            if detect_language:
                language = await self._detect_language(file_path, sample_pages)
                analysis["language"] = language
            
            # Content analysis
            content_analysis = await self._analyze_content(file_path, sample_pages, analysis_depth)
            analysis["content_analysis"] = content_analysis
            
            # Generate recommendations
            if recommend_tools:
                recommendations = self._generate_recommendations(analysis)
                analysis["recommendations"] = recommendations
            
            # Generate summary
            summary = self._generate_summary(analysis)
            analysis["summary"] = summary
            
            return MCPToolResult(
                content=[create_text_content(json.dumps(analysis, indent=2, ensure_ascii=False))],
                isError=False
            )
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {str(e)}")
            return MCPToolResult(
                content=[create_error_content(f"Document analysis failed: {str(e)}")],
                isError=True
            )
    
    async def _extract_document_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic document information."""
        info = {
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
        }
        
        if fitz:
            try:
                doc = fitz.open(file_path)
                info.update({
                    "page_count": doc.page_count,
                    "is_encrypted": doc.is_encrypted,
                    "is_pdf": doc.is_pdf,
                    "pdf_version": getattr(doc, 'pdf_version', None)
                })
                doc.close()
            except Exception as e:
                self.logger.warning(f"Failed to extract basic info with PyMuPDF: {str(e)}")
        
        return info
    
    async def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract document metadata."""
        metadata = {}
        
        if fitz:
            try:
                doc = fitz.open(file_path)
                pdf_metadata = doc.metadata
                
                metadata.update({
                    "title": pdf_metadata.get("title", ""),
                    "author": pdf_metadata.get("author", ""),
                    "subject": pdf_metadata.get("subject", ""),
                    "keywords": pdf_metadata.get("keywords", ""),
                    "creator": pdf_metadata.get("creator", ""),
                    "producer": pdf_metadata.get("producer", ""),
                    "creation_date": pdf_metadata.get("creationDate", ""),
                    "modification_date": pdf_metadata.get("modDate", "")
                })
                
                doc.close()
            except Exception as e:
                self.logger.warning(f"Failed to extract metadata: {str(e)}")
        
        return metadata
    
    async def _analyze_structure(self, file_path: Path, sample_pages: int, depth: str) -> Dict[str, Any]:
        """Analyze document structure."""
        structure = {
            "has_toc": False,
            "heading_levels": [],
            "sections": [],
            "page_layouts": [],
            "text_blocks": [],
            "images": [],
            "tables": []
        }
        
        if fitz:
            try:
                doc = fitz.open(file_path)
                
                # Check for table of contents
                toc = doc.get_toc()
                if toc:
                    structure["has_toc"] = True
                    structure["toc_entries"] = len(toc)
                    structure["heading_levels"] = list(set(item[0] for item in toc))
                
                # Analyze sample pages
                total_pages = doc.page_count
                if total_pages <= sample_pages:
                    pages_to_analyze = list(range(total_pages))
                else:
                    # Sample pages evenly distributed
                    step = total_pages // sample_pages
                    pages_to_analyze = [i * step for i in range(sample_pages)]
                
                for page_num in pages_to_analyze:
                    page = doc[page_num]
                    
                    # Analyze page layout
                    layout = self._analyze_page_layout(page)
                    structure["page_layouts"].append({
                        "page": page_num + 1,
                        "layout": layout
                    })
                    
                    # Extract text blocks
                    if depth in ["detailed", "comprehensive"]:
                        blocks = page.get_text("dict")
                        text_blocks = self._extract_text_blocks(blocks)
                        structure["text_blocks"].extend(text_blocks)
                    
                    # Detect images
                    images = page.get_images()
                    if images:
                        structure["images"].append({
                            "page": page_num + 1,
                            "image_count": len(images)
                        })
                    
                    # Detect tables (basic detection)
                    tables = self._detect_tables_in_page(page)
                    if tables:
                        structure["tables"].append({
                            "page": page_num + 1,
                            "table_count": len(tables)
                        })
                
                doc.close()
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze structure: {str(e)}")
        
        return structure
    
    def _analyze_page_layout(self, page) -> Dict[str, Any]:
        """Analyze the layout of a single page."""
        rect = page.rect
        
        layout = {
            "width": rect.width,
            "height": rect.height,
            "orientation": "portrait" if rect.height > rect.width else "landscape"
        }
        
        # Get text blocks to analyze layout
        blocks = page.get_text("dict")
        
        if blocks and "blocks" in blocks:
            text_blocks = [b for b in blocks["blocks"] if "lines" in b]
            
            if text_blocks:
                # Calculate column detection
                x_positions = []
                for block in text_blocks:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            x_positions.append(span["bbox"][0])  # left x coordinate
                
                if x_positions:
                    # Simple column detection based on x-position clustering
                    x_positions.sort()
                    columns = self._detect_columns(x_positions, rect.width)
                    layout["columns"] = columns
                    layout["column_count"] = len(columns)
        
        return layout
    
    def _detect_columns(self, x_positions: List[float], page_width: float) -> List[Dict[str, float]]:
        """Detect columns based on x-positions of text."""
        if not x_positions:
            return []
        
        # Group x-positions into clusters (potential column starts)
        tolerance = page_width * 0.05  # 5% of page width
        clusters = []
        
        for x in sorted(set(x_positions)):
            added_to_cluster = False
            for cluster in clusters:
                if abs(x - cluster["center"]) <= tolerance:
                    cluster["positions"].append(x)
                    cluster["center"] = sum(cluster["positions"]) / len(cluster["positions"])
                    added_to_cluster = True
                    break
            
            if not added_to_cluster:
                clusters.append({
                    "center": x,
                    "positions": [x]
                })
        
        # Filter clusters with significant number of text starts
        min_occurrences = max(1, len(x_positions) // 20)  # At least 5% of all text starts
        significant_clusters = [c for c in clusters if len(c["positions"]) >= min_occurrences]
        
        # Convert to column definitions
        columns = []
        for i, cluster in enumerate(sorted(significant_clusters, key=lambda c: c["center"])):
            column = {
                "index": i + 1,
                "left": cluster["center"],
                "width": None  # Will be calculated if there's a next column
            }
            
            if i < len(significant_clusters) - 1:
                next_cluster = sorted(significant_clusters, key=lambda c: c["center"])[i + 1]
                column["width"] = next_cluster["center"] - cluster["center"]
            else:
                column["width"] = page_width - cluster["center"]
            
            columns.append(column)
        
        return columns
    
    def _extract_text_blocks(self, blocks_dict: Dict) -> List[Dict[str, Any]]:
        """Extract and analyze text blocks."""
        text_blocks = []
        
        if "blocks" not in blocks_dict:
            return text_blocks
        
        for block in blocks_dict["blocks"]:
            if "lines" not in block:
                continue
            
            block_text = ""
            font_sizes = []
            
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                    font_sizes.append(span["size"])
                block_text += line_text + "\n"
            
            if block_text.strip():
                text_block = {
                    "text": block_text.strip(),
                    "bbox": block["bbox"],
                    "font_size_avg": statistics.mean(font_sizes) if font_sizes else 0,
                    "font_size_max": max(font_sizes) if font_sizes else 0,
                    "line_count": len(block["lines"]),
                    "char_count": len(block_text.strip())
                }
                
                # Classify block type
                text_block["type"] = self._classify_text_block(text_block)
                
                text_blocks.append(text_block)
        
        return text_blocks
    
    def _classify_text_block(self, block: Dict[str, Any]) -> str:
        """Classify a text block as heading, paragraph, etc."""
        text = block["text"]
        font_size = block["font_size_avg"]
        char_count = block["char_count"]
        
        # Simple heuristics for classification
        if char_count < 100 and font_size > 14:
            return "heading"
        elif char_count < 50:
            return "caption"
        elif re.match(r'^\d+\.\s', text.strip()):
            return "numbered_item"
        elif re.match(r'^[•\-\*]\s', text.strip()):
            return "bullet_item"
        elif char_count > 200:
            return "paragraph"
        else:
            return "text"
    
    def _detect_tables_in_page(self, page) -> List[Dict[str, Any]]:
        """Basic table detection in a page."""
        tables = []
        
        # Simple table detection based on text alignment
        blocks = page.get_text("dict")
        
        if "blocks" not in blocks:
            return tables
        
        # Look for patterns that suggest tabular data
        for block in blocks["blocks"]:
            if "lines" not in block:
                continue
            
            lines = block["lines"]
            if len(lines) < 3:  # Need at least 3 lines for a table
                continue
            
            # Check for consistent column alignment
            x_positions = []
            for line in lines:
                line_x_positions = []
                for span in line["spans"]:
                    line_x_positions.append(span["bbox"][0])
                x_positions.append(line_x_positions)
            
            # If multiple lines have similar x-positions, it might be a table
            if self._has_consistent_alignment(x_positions):
                tables.append({
                    "bbox": block["bbox"],
                    "estimated_rows": len(lines),
                    "estimated_columns": len(x_positions[0]) if x_positions else 0
                })
        
        return tables
    
    def _has_consistent_alignment(self, x_positions: List[List[float]]) -> bool:
        """Check if text has consistent column alignment."""
        if len(x_positions) < 3:
            return False
        
        # Check if at least 3 lines have similar x-positions
        tolerance = 10  # pixels
        
        for i in range(len(x_positions[0])):
            consistent_count = 0
            base_x = x_positions[0][i] if i < len(x_positions[0]) else None
            
            if base_x is None:
                continue
            
            for line_x_pos in x_positions[1:]:
                if i < len(line_x_pos) and abs(line_x_pos[i] - base_x) <= tolerance:
                    consistent_count += 1
            
            if consistent_count >= 2:  # At least 3 lines total (including base)
                return True
        
        return False
    
    async def _detect_document_type(self, file_path: Path, analysis: Dict, sample_pages: int) -> Dict[str, Any]:
        """Detect the type of document."""
        doc_type = {
            "primary_type": "unknown",
            "confidence": 0.0,
            "secondary_types": [],
            "characteristics": []
        }
        
        # Extract text sample for analysis
        text_sample = await self._extract_text_sample(file_path, sample_pages)
        
        if not text_sample:
            return doc_type
        
        # Analyze different document type indicators
        type_scores = {
            "academic_paper": self._score_academic_paper(text_sample, analysis),
            "financial_report": self._score_financial_report(text_sample, analysis),
            "technical_manual": self._score_technical_manual(text_sample, analysis),
            "legal_document": self._score_legal_document(text_sample, analysis),
            "form": self._score_form(text_sample, analysis),
            "presentation": self._score_presentation(text_sample, analysis),
            "book": self._score_book(text_sample, analysis),
            "article": self._score_article(text_sample, analysis)
        }
        
        # Find the highest scoring type
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        doc_type["primary_type"] = best_type
        doc_type["confidence"] = best_score
        
        # Find secondary types (scores > 0.3)
        secondary = [(t, s) for t, s in type_scores.items() if s > 0.3 and t != best_type]
        doc_type["secondary_types"] = sorted(secondary, key=lambda x: x[1], reverse=True)
        
        # Add characteristics based on analysis
        characteristics = []
        
        if analysis.get("structure", {}).get("has_toc"):
            characteristics.append("has_table_of_contents")
        
        if analysis.get("structure", {}).get("images"):
            characteristics.append("contains_images")
        
        if analysis.get("structure", {}).get("tables"):
            characteristics.append("contains_tables")
        
        page_count = analysis.get("document_info", {}).get("page_count", 0)
        if page_count > 50:
            characteristics.append("long_document")
        elif page_count < 5:
            characteristics.append("short_document")
        
        doc_type["characteristics"] = characteristics
        
        return doc_type
    
    async def _extract_text_sample(self, file_path: Path, sample_pages: int) -> str:
        """Extract a text sample for analysis."""
        text_sample = ""
        
        if fitz:
            try:
                doc = fitz.open(file_path)
                total_pages = doc.page_count
                
                # Sample first few pages and some middle pages
                pages_to_sample = []
                pages_to_sample.extend(range(min(3, total_pages)))  # First 3 pages
                
                if total_pages > 10:
                    # Add some middle pages
                    middle_start = total_pages // 3
                    pages_to_sample.extend(range(middle_start, min(middle_start + 2, total_pages)))
                
                pages_to_sample = list(set(pages_to_sample))[:sample_pages]
                
                for page_num in pages_to_sample:
                    page = doc[page_num]
                    text_sample += page.get_text() + "\n\n"
                
                doc.close()
                
            except Exception as e:
                self.logger.warning(f"Failed to extract text sample: {str(e)}")
        
        return text_sample[:10000]  # Limit to first 10k characters
    
    def _score_academic_paper(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being an academic paper."""
        score = 0.0
        
        # Check for academic keywords
        academic_keywords = [
            "abstract", "introduction", "methodology", "results", "conclusion",
            "references", "bibliography", "doi:", "arxiv", "journal",
            "university", "research", "study", "analysis", "experiment"
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in academic_keywords if keyword in text_lower)
        score += min(keyword_matches / len(academic_keywords), 0.4)
        
        # Check for citation patterns
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\w+,?\s*\d{4}\)',  # (Author, 2023)
            r'et al\.',  # et al.
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, text):
                score += 0.1
        
        # Check structure
        if analysis.get("structure", {}).get("has_toc"):
            score += 0.1
        
        # Check metadata
        metadata = analysis.get("metadata", {})
        if any(keyword in metadata.get("keywords", "").lower() for keyword in ["research", "study", "analysis"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_financial_report(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being a financial report."""
        score = 0.0
        
        financial_keywords = [
            "revenue", "profit", "loss", "balance sheet", "income statement",
            "cash flow", "assets", "liabilities", "equity", "earnings",
            "financial", "fiscal", "quarter", "annual report", "sec filing"
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in financial_keywords if keyword in text_lower)
        score += min(keyword_matches / len(financial_keywords), 0.5)
        
        # Check for financial patterns
        financial_patterns = [
            r'\$[\d,]+',  # Dollar amounts
            r'\d+\.\d+%',  # Percentages
            r'Q[1-4]\s*\d{4}',  # Quarters
        ]
        
        for pattern in financial_patterns:
            if re.search(pattern, text):
                score += 0.1
        
        # Check for tables (financial reports often have many tables)
        if len(analysis.get("structure", {}).get("tables", [])) > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_technical_manual(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being a technical manual."""
        score = 0.0
        
        technical_keywords = [
            "manual", "guide", "instructions", "procedure", "step",
            "configuration", "installation", "setup", "troubleshooting",
            "specification", "technical", "documentation", "api", "software"
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in technical_keywords if keyword in text_lower)
        score += min(keyword_matches / len(technical_keywords), 0.4)
        
        # Check for numbered steps
        if re.search(r'^\d+\.\s', text, re.MULTILINE):
            score += 0.2
        
        # Check for code blocks or technical formatting
        if re.search(r'```|<code>|\{.*\}', text):
            score += 0.1
        
        # Check structure
        if analysis.get("structure", {}).get("has_toc"):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_legal_document(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being a legal document."""
        score = 0.0
        
        legal_keywords = [
            "whereas", "therefore", "hereby", "pursuant", "contract",
            "agreement", "legal", "court", "plaintiff", "defendant",
            "clause", "section", "article", "law", "statute"
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in legal_keywords if keyword in text_lower)
        score += min(keyword_matches / len(legal_keywords), 0.5)
        
        # Check for legal formatting
        if re.search(r'\b[A-Z]{2,}\b', text):  # ALL CAPS words
            score += 0.1
        
        if re.search(r'Section\s+\d+', text, re.IGNORECASE):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_form(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being a form."""
        score = 0.0
        
        form_keywords = [
            "name:", "address:", "phone:", "email:", "date:",
            "signature", "form", "application", "please fill",
            "check box", "select", "enter", "provide"
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in form_keywords if keyword in text_lower)
        score += min(keyword_matches / len(form_keywords), 0.4)
        
        # Check for form patterns
        if re.search(r'_{3,}', text):  # Underlines for filling
            score += 0.2
        
        if re.search(r'\[\s*\]', text):  # Checkboxes
            score += 0.2
        
        # Forms are usually short
        page_count = analysis.get("document_info", {}).get("page_count", 0)
        if page_count <= 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_presentation(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being a presentation."""
        score = 0.0
        
        # Presentations often have short text blocks
        structure = analysis.get("structure", {})
        text_blocks = structure.get("text_blocks", [])
        
        if text_blocks:
            avg_block_length = sum(block["char_count"] for block in text_blocks) / len(text_blocks)
            if avg_block_length < 200:  # Short text blocks
                score += 0.3
        
        # Check for slide-like content
        if re.search(r'slide\s+\d+', text.lower()):
            score += 0.2
        
        # Check page layout (presentations often have landscape orientation)
        layouts = structure.get("page_layouts", [])
        landscape_count = sum(1 for layout in layouts if layout.get("layout", {}).get("orientation") == "landscape")
        if layouts and landscape_count / len(layouts) > 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_book(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being a book."""
        score = 0.0
        
        # Books are usually long
        page_count = analysis.get("document_info", {}).get("page_count", 0)
        if page_count > 50:
            score += 0.3
        elif page_count > 100:
            score += 0.5
        
        # Check for book-like structure
        if analysis.get("structure", {}).get("has_toc"):
            score += 0.2
        
        # Check for chapters
        if re.search(r'chapter\s+\d+', text.lower()):
            score += 0.2
        
        # Check metadata
        metadata = analysis.get("metadata", {})
        if metadata.get("author") and metadata.get("title"):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_article(self, text: str, analysis: Dict) -> float:
        """Score likelihood of being an article."""
        score = 0.0
        
        # Articles are medium length
        page_count = analysis.get("document_info", {}).get("page_count", 0)
        if 3 <= page_count <= 20:
            score += 0.2
        
        # Check for article-like keywords
        article_keywords = ["article", "news", "report", "story", "author", "published"]
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in article_keywords if keyword in text_lower)
        score += min(keyword_matches / len(article_keywords), 0.3)
        
        # Check for byline patterns
        if re.search(r'by\s+[A-Z][a-z]+\s+[A-Z][a-z]+', text):
            score += 0.2
        
        return min(score, 1.0)
    
    async def _detect_language(self, file_path: Path, sample_pages: int) -> Dict[str, Any]:
        """Detect document language."""
        language_info = {
            "primary_language": "unknown",
            "confidence": 0.0,
            "detected_languages": []
        }
        
        # Extract text sample
        text_sample = await self._extract_text_sample(file_path, sample_pages)
        
        if not text_sample:
            return language_info
        
        # Simple language detection based on character patterns
        # This is a basic implementation - in practice, you'd use a proper language detection library
        
        # Count character types
        char_counts = {
            "latin": 0,
            "chinese": 0,
            "arabic": 0,
            "cyrillic": 0,
            "japanese": 0
        }
        
        for char in text_sample:
            if ord(char) < 128:  # ASCII (mostly English)
                char_counts["latin"] += 1
            elif 0x4e00 <= ord(char) <= 0x9fff:  # Chinese
                char_counts["chinese"] += 1
            elif 0x0600 <= ord(char) <= 0x06ff:  # Arabic
                char_counts["arabic"] += 1
            elif 0x0400 <= ord(char) <= 0x04ff:  # Cyrillic
                char_counts["cyrillic"] += 1
            elif 0x3040 <= ord(char) <= 0x309f or 0x30a0 <= ord(char) <= 0x30ff:  # Japanese
                char_counts["japanese"] += 1
        
        total_chars = sum(char_counts.values())
        if total_chars > 0:
            # Find dominant script
            dominant_script = max(char_counts, key=char_counts.get)
            confidence = char_counts[dominant_script] / total_chars
            
            # Map script to language (simplified)
            script_to_language = {
                "latin": "english",
                "chinese": "chinese",
                "arabic": "arabic",
                "cyrillic": "russian",
                "japanese": "japanese"
            }
            
            language_info["primary_language"] = script_to_language.get(dominant_script, "unknown")
            language_info["confidence"] = confidence
        
        return language_info
    
    async def _analyze_content(self, file_path: Path, sample_pages: int, depth: str) -> Dict[str, Any]:
        """Analyze document content characteristics."""
        content_analysis = {
            "text_density": 0.0,
            "image_density": 0.0,
            "table_density": 0.0,
            "readability": {},
            "topics": [],
            "complexity": "unknown"
        }
        
        # Extract text sample
        text_sample = await self._extract_text_sample(file_path, sample_pages)
        
        if text_sample:
            # Calculate text density (characters per page)
            page_count = max(1, sample_pages)
            content_analysis["text_density"] = len(text_sample) / page_count
            
            # Basic readability analysis
            content_analysis["readability"] = self._analyze_readability(text_sample)
            
            # Extract topics (simple keyword extraction)
            if depth in ["detailed", "comprehensive"]:
                content_analysis["topics"] = self._extract_topics(text_sample)
            
            # Assess complexity
            content_analysis["complexity"] = self._assess_complexity(text_sample)
        
        return content_analysis
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        
        if not sentences or not words:
            return {"avg_sentence_length": 0, "avg_word_length": 0}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2),
            "total_sentences": len(sentences),
            "total_words": len(words)
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text (simple keyword extraction)."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "this", "that", "these", "those", "it", "its", "they", "them", "their"
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words and count frequency
        word_freq = Counter(word for word in words if word not in stop_words)
        
        # Return top 10 most frequent words as topics
        return [word for word, count in word_freq.most_common(10)]
    
    def _assess_complexity(self, text: str) -> str:
        """Assess document complexity based on text characteristics."""
        words = text.split()
        
        if not words:
            return "unknown"
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Count complex words (more than 6 characters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_ratio = complex_words / len(words)
        
        # Assess complexity
        if avg_word_length > 6 and complex_ratio > 0.3:
            return "high"
        elif avg_word_length > 5 and complex_ratio > 0.2:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tool recommendations based on analysis."""
        recommendations = {
            "primary_tools": [],
            "secondary_tools": [],
            "processing_order": [],
            "special_considerations": []
        }
        
        doc_type = analysis.get("document_type", {}).get("primary_type", "unknown")
        structure = analysis.get("structure", {})
        content = analysis.get("content_analysis", {})
        
        # Recommend tools based on document type
        if doc_type == "academic_paper":
            recommendations["primary_tools"] = ["extract_text", "extract_formulas"]
            recommendations["secondary_tools"] = ["extract_tables"]
            recommendations["processing_order"] = ["analyze_document", "extract_text", "extract_formulas", "extract_tables"]
        
        elif doc_type == "financial_report":
            recommendations["primary_tools"] = ["extract_tables", "extract_text"]
            recommendations["secondary_tools"] = ["extract_text_ocr"]
            recommendations["processing_order"] = ["analyze_document", "extract_tables", "extract_text"]
        
        elif doc_type == "technical_manual":
            recommendations["primary_tools"] = ["extract_text"]
            recommendations["secondary_tools"] = ["extract_tables", "extract_text_ocr"]
            recommendations["processing_order"] = ["analyze_document", "extract_text", "extract_tables"]
        
        elif doc_type == "form":
            recommendations["primary_tools"] = ["extract_text_ocr", "extract_text"]
            recommendations["secondary_tools"] = []
            recommendations["processing_order"] = ["analyze_document", "extract_text_ocr", "extract_text"]
        
        else:  # Unknown or general document
            recommendations["primary_tools"] = ["extract_text"]
            recommendations["secondary_tools"] = ["extract_tables", "extract_text_ocr"]
            recommendations["processing_order"] = ["analyze_document", "extract_text"]
        
        # Add recommendations based on structure
        if structure.get("tables"):
            if "extract_tables" not in recommendations["primary_tools"]:
                recommendations["secondary_tools"].append("extract_tables")
        
        if structure.get("images"):
            if "extract_text_ocr" not in recommendations["primary_tools"]:
                recommendations["secondary_tools"].append("extract_text_ocr")
        
        # Add special considerations
        if content.get("complexity") == "high":
            recommendations["special_considerations"].append("Document has high complexity - consider manual review")
        
        if analysis.get("document_info", {}).get("page_count", 0) > 100:
            recommendations["special_considerations"].append("Large document - consider batch processing")
        
        if analysis.get("language", {}).get("primary_language") != "english":
            recommendations["special_considerations"].append("Non-English document - OCR accuracy may vary")
        
        return recommendations
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the analysis."""
        doc_info = analysis.get("document_info", {})
        doc_type = analysis.get("document_type", {})
        language = analysis.get("language", {})
        structure = analysis.get("structure", {})
        
        summary_parts = []
        
        # Basic info
        file_name = doc_info.get("file_name", "Unknown")
        page_count = doc_info.get("page_count", 0)
        file_size_mb = doc_info.get("file_size_mb", 0)
        
        summary_parts.append(f"Document: {file_name} ({page_count} pages, {file_size_mb} MB)")
        
        # Document type
        primary_type = doc_type.get("primary_type", "unknown")
        confidence = doc_type.get("confidence", 0)
        summary_parts.append(f"Type: {primary_type.replace('_', ' ').title()} (confidence: {confidence:.1%})")
        
        # Language
        primary_lang = language.get("primary_language", "unknown")
        lang_confidence = language.get("confidence", 0)
        summary_parts.append(f"Language: {primary_lang.title()} (confidence: {lang_confidence:.1%})")
        
        # Structure highlights
        structure_features = []
        if structure.get("has_toc"):
            structure_features.append("table of contents")
        if structure.get("tables"):
            table_count = sum(page.get("table_count", 0) for page in structure["tables"])
            structure_features.append(f"{table_count} tables")
        if structure.get("images"):
            image_count = sum(page.get("image_count", 0) for page in structure["images"])
            structure_features.append(f"{image_count} images")
        
        if structure_features:
            summary_parts.append(f"Contains: {', '.join(structure_features)}")
        
        return "; ".join(summary_parts)


class SmartRoutingTool(PDFAnalysisTool):
    """Smart routing tool that automatically selects optimal processing pipeline."""
    
    def __init__(self):
        super().__init__(
            name="smart_routing",
            description="Automatically analyze document and route to optimal processing pipeline",
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
                "processing_goals": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["text_extraction", "table_extraction", "formula_extraction", "ocr", "full_analysis"]
                    },
                    "default": ["text_extraction"],
                    "description": "Processing goals to optimize for"
                },
                "quality_preference": {
                    "type": "string",
                    "enum": ["speed", "balanced", "quality"],
                    "default": "balanced",
                    "description": "Processing quality vs speed preference"
                },
                "auto_execute": {
                    "type": "boolean",
                    "default": False,
                    "description": "Automatically execute recommended processing pipeline"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute smart routing analysis."""
        try:
            file_path = Path(kwargs["file_path"])
            processing_goals = kwargs.get("processing_goals", ["text_extraction"])
            quality_preference = kwargs.get("quality_preference", "balanced")
            auto_execute = kwargs.get("auto_execute", False)
            
            if not file_path.exists():
                raise MCPResourceException(f"PDF file not found: {file_path}")
            
            # First, analyze the document
            analyzer = AnalyzeDocumentTool()
            analysis_result = await analyzer.execute(
                file_path=str(file_path),
                analysis_depth="detailed",
                recommend_tools=True
            )
            
            if analysis_result.isError:
                return analysis_result
            
            analysis = json.loads(analysis_result.content[0].text)
            
            # Generate optimized processing pipeline
            pipeline = self._generate_pipeline(
                analysis, processing_goals, quality_preference
            )
            
            result = {
                "analysis_summary": analysis.get("summary", ""),
                "document_type": analysis.get("document_type", {}),
                "recommended_pipeline": pipeline,
                "processing_goals": processing_goals,
                "quality_preference": quality_preference
            }
            
            # Auto-execute if requested
            if auto_execute:
                execution_results = await self._execute_pipeline(file_path, pipeline)
                result["execution_results"] = execution_results
            
            return MCPToolResult(
                content=[create_text_content(json.dumps(result, indent=2, ensure_ascii=False))],
                isError=False
            )
            
        except Exception as e:
            self.logger.error(f"Smart routing failed: {str(e)}")
            return MCPToolResult(
                content=[create_error_content(f"Smart routing failed: {str(e)}")],
                isError=True
            )
    
    def _generate_pipeline(self, analysis: Dict, goals: List[str], quality: str) -> Dict[str, Any]:
        """Generate optimized processing pipeline."""
        doc_type = analysis.get("document_type", {}).get("primary_type", "unknown")
        structure = analysis.get("structure", {})
        
        pipeline = {
            "steps": [],
            "parallel_steps": [],
            "estimated_time": 0,
            "confidence": 0.8
        }
        
        # Base pipeline on document type and goals
        if "text_extraction" in goals:
            if doc_type in ["academic_paper", "technical_manual", "book", "article"]:
                pipeline["steps"].append({
                    "tool": "extract_text",
                    "method": "pymupdf" if quality == "speed" else "pdfplumber",
                    "priority": 1
                })
            elif doc_type == "form" or structure.get("images"):
                pipeline["steps"].append({
                    "tool": "extract_text_ocr",
                    "engine": "tesseract" if quality == "speed" else "auto",
                    "priority": 1
                })
            else:
                pipeline["steps"].append({
                    "tool": "extract_text",
                    "method": "auto",
                    "priority": 1
                })
        
        if "table_extraction" in goals and structure.get("tables"):
            pipeline["steps"].append({
                "tool": "extract_tables",
                "method": "camelot" if quality != "speed" else "auto",
                "priority": 2
            })
        
        if "formula_extraction" in goals:
            if doc_type == "academic_paper" or "formula" in analysis.get("content_analysis", {}).get("topics", []):
                pipeline["steps"].append({
                    "tool": "extract_formulas",
                    "engine": "pix2tex",
                    "priority": 3
                })
        
        if "ocr" in goals:
            pipeline["steps"].append({
                "tool": "extract_text_ocr",
                "engine": "auto",
                "priority": 4
            })
        
        if "full_analysis" in goals:
            pipeline["steps"].insert(0, {
                "tool": "analyze_document",
                "analysis_depth": "comprehensive",
                "priority": 0
            })
        
        # Optimize for parallel execution
        if quality != "speed" and len(pipeline["steps"]) > 2:
            # Group independent steps for parallel execution
            text_steps = [s for s in pipeline["steps"] if "text" in s["tool"]]
            other_steps = [s for s in pipeline["steps"] if "text" not in s["tool"]]
            
            if len(text_steps) > 1:
                pipeline["parallel_steps"].append(text_steps)
                pipeline["steps"] = other_steps
        
        # Estimate processing time
        page_count = analysis.get("document_info", {}).get("page_count", 1)
        base_time = page_count * 2  # 2 seconds per page base
        
        for step in pipeline["steps"]:
            if step["tool"] == "extract_text_ocr":
                base_time += page_count * 5  # OCR is slower
            elif step["tool"] == "extract_formulas":
                base_time += page_count * 3  # Formula extraction is slower
        
        pipeline["estimated_time"] = base_time
        
        return pipeline
    
    async def _execute_pipeline(self, file_path: Path, pipeline: Dict) -> Dict[str, Any]:
        """Execute the processing pipeline."""
        results = {
            "executed_steps": [],
            "execution_time": 0,
            "success": True,
            "errors": []
        }
        
        start_time = datetime.now()
        
        # Execute sequential steps
        for step in sorted(pipeline["steps"], key=lambda s: s["priority"]):
            try:
                step_result = await self._execute_step(file_path, step)
                results["executed_steps"].append({
                    "step": step,
                    "success": not step_result.isError,
                    "result_preview": step_result.content[0].text[:200] + "..." if step_result.content else ""
                })
                
                if step_result.isError:
                    results["errors"].append(f"Step {step['tool']} failed: {step_result.content[0].text}")
                    
            except Exception as e:
                results["errors"].append(f"Step {step['tool']} failed: {str(e)}")
                results["success"] = False
        
        # Execute parallel steps
        for parallel_group in pipeline.get("parallel_steps", []):
            try:
                # In a real implementation, you'd use asyncio.gather for true parallelism
                for step in parallel_group:
                    step_result = await self._execute_step(file_path, step)
                    results["executed_steps"].append({
                        "step": step,
                        "success": not step_result.isError,
                        "result_preview": step_result.content[0].text[:200] + "..." if step_result.content else ""
                    })
                    
            except Exception as e:
                results["errors"].append(f"Parallel execution failed: {str(e)}")
                results["success"] = False
        
        end_time = datetime.now()
        results["execution_time"] = (end_time - start_time).total_seconds()
        
        return results
    
    async def _execute_step(self, file_path: Path, step: Dict) -> MCPToolResult:
        """Execute a single pipeline step."""
        tool_name = step["tool"]
        
        # Import and execute the appropriate tool
        # This is a simplified implementation - in practice, you'd have a proper tool registry
        
        if tool_name == "analyze_document":
            from .document_analysis import AnalyzeDocumentTool
            tool = AnalyzeDocumentTool()
            return await tool.execute(
                file_path=str(file_path),
                analysis_depth=step.get("analysis_depth", "detailed")
            )
        
        elif tool_name == "extract_text":
            from .text_extraction import ExtractTextTool
            tool = ExtractTextTool()
            return await tool.execute(
                file_path=str(file_path),
                method=step.get("method", "auto")
            )
        
        elif tool_name == "extract_tables":
            from .table_extraction import ExtractTablesTool
            tool = ExtractTablesTool()
            return await tool.execute(
                file_path=str(file_path),
                method=step.get("method", "auto")
            )
        
        elif tool_name == "extract_formulas":
            from .formula_extraction import ExtractFormulasTool
            tool = ExtractFormulasTool()
            return await tool.execute(
                file_path=str(file_path),
                engine=step.get("engine", "auto")
            )
        
        elif tool_name == "extract_text_ocr":
            from .ocr_extraction import ExtractTextOCRTool
            tool = ExtractTextOCRTool()
            return await tool.execute(
                file_path=str(file_path),
                engine=step.get("engine", "auto")
            )
        
        else:
            raise ToolExecutionException(f"Unknown tool: {tool_name}")