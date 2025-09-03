#!/usr/bin/env python3
"""
Document analysis tools for PDF documents.
Provides intelligent document type detection and processing pipeline recommendations.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import re

from pydantic import Field, ConfigDict

from .base import PDFTool, PDFToolInput, PDFToolOutput

logger = logging.getLogger(__name__)


class AnalyzeDocumentInput(PDFToolInput):
    """Input schema for document analysis."""
    
    model_config = ConfigDict(extra='forbid')
    
    analysis_depth: str = Field(
        default="standard",
        description="Analysis depth: 'basic', 'standard', 'comprehensive'"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include document metadata in analysis"
    )
    include_structure: bool = Field(
        default=True,
        description="Include document structure analysis"
    )
    include_content: bool = Field(
        default=True,
        description="Include content analysis"
    )
    language_detection: bool = Field(
        default=True,
        description="Perform language detection"
    )
    extract_keywords: bool = Field(
        default=False,
        description="Extract keywords and topics"
    )
    max_pages_sample: int = Field(
        default=10,
        description="Maximum pages to sample for analysis",
        ge=1,
        le=100
    )


class AnalyzeDocumentOutput(PDFToolOutput):
    """Output schema for document analysis."""
    
    document_info: Dict[str, Any] = Field(
        description="Basic document information"
    )
    document_type: str = Field(
        description="Detected document type"
    )
    confidence: float = Field(
        description="Confidence score for document type detection",
        ge=0.0,
        le=1.0
    )
    structure_analysis: Dict[str, Any] = Field(
        description="Document structure analysis"
    )
    content_analysis: Dict[str, Any] = Field(
        description="Content analysis results"
    )
    recommended_tools: List[str] = Field(
        description="Recommended tools for processing this document"
    )
    processing_pipeline: List[Dict[str, Any]] = Field(
        description="Recommended processing pipeline"
    )


class SmartRoutingInput(PDFToolInput):
    """Input schema for smart routing."""
    
    model_config = ConfigDict(extra='forbid')
    
    task_description: str = Field(
        description="Description of the task to be performed"
    )
    preferred_methods: Optional[List[str]] = Field(
        default=None,
        description="Preferred extraction methods"
    )
    quality_priority: str = Field(
        default="balanced",
        description="Quality priority: 'speed', 'balanced', 'accuracy'"
    )
    output_format: str = Field(
        default="json",
        description="Desired output format"
    )


class SmartRoutingOutput(PDFToolOutput):
    """Output schema for smart routing."""
    
    recommended_tool: str = Field(
        description="Recommended tool for the task"
    )
    tool_parameters: Dict[str, Any] = Field(
        description="Recommended parameters for the tool"
    )
    alternative_tools: List[Dict[str, Any]] = Field(
        description="Alternative tools and their parameters"
    )
    reasoning: str = Field(
        description="Explanation of the recommendation"
    )
    estimated_processing_time: str = Field(
        description="Estimated processing time"
    )


class AnalyzeDocumentTool(PDFTool):
    """Tool for analyzing PDF documents and detecting their type and characteristics."""
    
    name = "analyze_document"
    description = "Analyze PDF document structure, content, and type for intelligent processing"
    input_schema = AnalyzeDocumentInput
    output_schema = AnalyzeDocumentOutput
    
    def execute(self, input_data: AnalyzeDocumentInput) -> AnalyzeDocumentOutput:
        """Execute document analysis."""
        try:
            # Validate file exists
            if not Path(input_data.file_path).exists():
                raise FileNotFoundError(f"PDF file not found: {input_data.file_path}")
            
            # Extract basic document information
            doc_info = self._extract_document_info(input_data.file_path)
            
            # Analyze document structure
            structure_analysis = {}
            if input_data.include_structure:
                structure_analysis = self._analyze_structure(
                    input_data.file_path, 
                    input_data.max_pages_sample
                )
            
            # Analyze content
            content_analysis = {}
            if input_data.include_content:
                content_analysis = self._analyze_content(
                    input_data.file_path,
                    input_data.language_detection,
                    input_data.extract_keywords,
                    input_data.max_pages_sample
                )
            
            # Detect document type
            doc_type, confidence = self._detect_document_type(
                doc_info, structure_analysis, content_analysis
            )
            
            # Generate tool recommendations
            recommended_tools = self._recommend_tools(doc_type, structure_analysis, content_analysis)
            
            # Generate processing pipeline
            processing_pipeline = self._generate_pipeline(doc_type, recommended_tools)
            
            return AnalyzeDocumentOutput(
                success=True,
                document_info=doc_info,
                document_type=doc_type,
                confidence=confidence,
                structure_analysis=structure_analysis,
                content_analysis=content_analysis,
                recommended_tools=recommended_tools,
                processing_pipeline=processing_pipeline,
                metadata={
                    "file_path": input_data.file_path,
                    "analysis_depth": input_data.analysis_depth
                }
            )
            
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            return AnalyzeDocumentOutput(
                success=False,
                error=str(e),
                document_info={},
                document_type="unknown",
                confidence=0.0,
                structure_analysis={},
                content_analysis={},
                recommended_tools=[],
                processing_pipeline=[],
                metadata={"file_path": input_data.file_path}
            )
    
    def _extract_document_info(self, file_path: str) -> Dict[str, Any]:
        """Extract basic document information."""
        info = {
            "file_size": 0,
            "page_count": 0,
            "title": "",
            "author": "",
            "subject": "",
            "creator": "",
            "producer": "",
            "creation_date": None,
            "modification_date": None,
            "encrypted": False
        }
        
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            
            info["page_count"] = len(doc)
            info["encrypted"] = doc.needs_pass
            
            # Extract metadata
            metadata = doc.metadata
            info["title"] = metadata.get("title", "")
            info["author"] = metadata.get("author", "")
            info["subject"] = metadata.get("subject", "")
            info["creator"] = metadata.get("creator", "")
            info["producer"] = metadata.get("producer", "")
            info["creation_date"] = metadata.get("creationDate", "")
            info["modification_date"] = metadata.get("modDate", "")
            
            doc.close()
            
            # Get file size
            info["file_size"] = Path(file_path).stat().st_size
            
        except ImportError:
            logger.warning("PyMuPDF not available for metadata extraction")
        except Exception as e:
            logger.error(f"Error extracting document info: {str(e)}")
        
        return info
    
    def _analyze_structure(self, file_path: str, max_pages: int) -> Dict[str, Any]:
        """Analyze document structure."""
        structure = {
            "has_toc": False,
            "heading_levels": 0,
            "text_blocks": 0,
            "images": 0,
            "tables": 0,
            "links": 0,
            "annotations": 0,
            "fonts": [],
            "page_layouts": []
        }
        
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            
            # Check for table of contents
            toc = doc.get_toc()
            structure["has_toc"] = len(toc) > 0
            if toc:
                structure["heading_levels"] = max([item[0] for item in toc])
            
            # Analyze sample pages
            pages_to_analyze = min(max_pages, len(doc))
            fonts_set = set()
            
            for page_num in range(pages_to_analyze):
                page = doc[page_num]
                
                # Count text blocks
                text_dict = page.get_text("dict")
                structure["text_blocks"] += len(text_dict.get("blocks", []))
                
                # Count images
                image_list = page.get_images()
                structure["images"] += len(image_list)
                
                # Count links
                links = page.get_links()
                structure["links"] += len(links)
                
                # Count annotations
                annots = page.annots()
                structure["annotations"] += len(list(annots))
                
                # Collect fonts
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                font_info = f"{span.get('font', 'Unknown')}_{span.get('size', 0)}"
                                fonts_set.add(font_info)
                
                # Analyze page layout
                layout_info = {
                    "page": page_num + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "text_blocks": len([b for b in text_dict.get("blocks", []) if "lines" in b]),
                    "image_blocks": len([b for b in text_dict.get("blocks", []) if "lines" not in b])
                }
                structure["page_layouts"].append(layout_info)
            
            structure["fonts"] = list(fonts_set)
            
            # Estimate table count (simple heuristic)
            structure["tables"] = self._estimate_table_count(doc, pages_to_analyze)
            
            doc.close()
            
        except ImportError:
            logger.warning("PyMuPDF not available for structure analysis")
        except Exception as e:
            logger.error(f"Error analyzing structure: {str(e)}")
        
        return structure
    
    def _analyze_content(self, file_path: str, detect_language: bool, 
                        extract_keywords: bool, max_pages: int) -> Dict[str, Any]:
        """Analyze document content."""
        content = {
            "text_density": 0.0,
            "average_words_per_page": 0,
            "language": "unknown",
            "readability_score": 0.0,
            "keywords": [],
            "topics": [],
            "has_formulas": False,
            "has_code": False,
            "complexity": "medium"
        }
        
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            
            total_text = ""
            total_words = 0
            pages_analyzed = min(max_pages, len(doc))
            
            for page_num in range(pages_analyzed):
                page = doc[page_num]
                page_text = page.get_text()
                total_text += page_text + " "
                
                # Count words
                words = len(page_text.split())
                total_words += words
            
            doc.close()
            
            # Calculate metrics
            if pages_analyzed > 0:
                content["average_words_per_page"] = total_words / pages_analyzed
                content["text_density"] = len(total_text.strip()) / pages_analyzed if pages_analyzed > 0 else 0
            
            # Language detection
            if detect_language and total_text.strip():
                content["language"] = self._detect_language(total_text)
            
            # Check for formulas (simple heuristic)
            formula_patterns = [r'\$.*\$', r'\\[a-zA-Z]+', r'[∫∑∏√±≤≥≠∞]']
            content["has_formulas"] = any(re.search(pattern, total_text) for pattern in formula_patterns)
            
            # Check for code (simple heuristic)
            code_patterns = [r'def\s+\w+\(', r'class\s+\w+:', r'import\s+\w+', r'#include', r'function\s+\w+']
            content["has_code"] = any(re.search(pattern, total_text) for pattern in code_patterns)
            
            # Extract keywords
            if extract_keywords:
                content["keywords"] = self._extract_keywords(total_text)
            
            # Determine complexity
            content["complexity"] = self._assess_complexity(total_text, content)
            
        except ImportError:
            logger.warning("PyMuPDF not available for content analysis")
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
        
        return content
    
    def _detect_document_type(self, doc_info: Dict[str, Any], 
                             structure: Dict[str, Any], 
                             content: Dict[str, Any]) -> tuple[str, float]:
        """Detect document type based on analysis."""
        scores = {
            "academic_paper": 0.0,
            "financial_report": 0.0,
            "technical_manual": 0.0,
            "legal_document": 0.0,
            "form": 0.0,
            "presentation": 0.0,
            "book": 0.0,
            "article": 0.0
        }
        
        # Academic paper indicators
        if structure.get("has_toc", False):
            scores["academic_paper"] += 0.2
            scores["book"] += 0.3
        
        if content.get("has_formulas", False):
            scores["academic_paper"] += 0.3
            scores["technical_manual"] += 0.2
        
        if doc_info.get("page_count", 0) > 50:
            scores["book"] += 0.3
            scores["technical_manual"] += 0.2
        elif doc_info.get("page_count", 0) < 10:
            scores["article"] += 0.3
            scores["form"] += 0.2
        
        # Financial report indicators
        if structure.get("tables", 0) > 5:
            scores["financial_report"] += 0.4
        
        # Technical manual indicators
        if content.get("has_code", False):
            scores["technical_manual"] += 0.4
        
        # Presentation indicators
        if structure.get("images", 0) > structure.get("text_blocks", 1):
            scores["presentation"] += 0.3
        
        # Form indicators
        if structure.get("annotations", 0) > 0:
            scores["form"] += 0.3
        
        # Find best match
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # If no clear winner, default to article
        if confidence < 0.3:
            best_type = "article"
            confidence = 0.5
        
        return best_type, min(confidence, 1.0)
    
    def _recommend_tools(self, doc_type: str, structure: Dict[str, Any], 
                        content: Dict[str, Any]) -> List[str]:
        """Recommend tools based on document analysis."""
        tools = []
        
        # Always recommend basic text extraction
        tools.append("extract_text")
        
        # Conditional recommendations
        if structure.get("tables", 0) > 0:
            tools.append("extract_tables")
        
        if structure.get("images", 0) > 0:
            tools.append("extract_images")
        
        if content.get("has_formulas", False):
            tools.append("extract_formulas")
        
        if structure.get("annotations", 0) > 0:
            tools.append("extract_annotations")
        
        # Type-specific recommendations
        if doc_type == "financial_report":
            tools.extend(["extract_tables", "extract_charts"])
        elif doc_type == "academic_paper":
            tools.extend(["extract_references", "extract_formulas"])
        elif doc_type == "technical_manual":
            tools.extend(["extract_code", "extract_diagrams"])
        
        return list(set(tools))  # Remove duplicates
    
    def _generate_pipeline(self, doc_type: str, tools: List[str]) -> List[Dict[str, Any]]:
        """Generate processing pipeline."""
        pipeline = []
        
        # Start with document analysis
        pipeline.append({
            "step": 1,
            "tool": "analyze_document",
            "description": "Analyze document structure and content",
            "required": True
        })
        
        # Add recommended tools in logical order
        step = 2
        tool_order = ["extract_text", "extract_tables", "extract_images", 
                     "extract_formulas", "extract_annotations"]
        
        for tool in tool_order:
            if tool in tools:
                pipeline.append({
                    "step": step,
                    "tool": tool,
                    "description": f"Extract {tool.split('_')[1]} from document",
                    "required": tool == "extract_text"
                })
                step += 1
        
        return pipeline
    
    def _estimate_table_count(self, doc, max_pages: int) -> int:
        """Estimate number of tables in document."""
        # Simple heuristic based on text layout patterns
        table_count = 0
        
        try:
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                
                # Look for table-like patterns
                lines = text.split('\n')
                for line in lines:
                    # Count lines with multiple tab-separated or space-separated columns
                    if '\t' in line and len(line.split('\t')) > 2:
                        table_count += 0.1
                    elif len(re.findall(r'\s{3,}', line)) > 1:
                        table_count += 0.1
        
        except Exception:
            pass
        
        return int(table_count)
    
    def _detect_language(self, text: str) -> str:
        """Detect document language."""
        # Simple language detection based on common words
        text_lower = text.lower()
        
        english_indicators = ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        chinese_indicators = ['的', '是', '在', '有', '和', '了', '不', '我', '你', '他']
        
        english_score = sum(1 for word in english_indicators if word in text_lower)
        chinese_score = sum(1 for word in chinese_indicators if word in text_lower)
        
        if english_score > chinese_score:
            return "english"
        elif chinese_score > 0:
            return "chinese"
        else:
            return "unknown"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 10 most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _assess_complexity(self, text: str, content: Dict[str, Any]) -> str:
        """Assess document complexity."""
        complexity_score = 0
        
        # Factors that increase complexity
        if content.get("has_formulas", False):
            complexity_score += 2
        
        if content.get("has_code", False):
            complexity_score += 2
        
        avg_words = content.get("average_words_per_page", 0)
        if avg_words > 500:
            complexity_score += 1
        elif avg_words > 1000:
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"


class SmartRoutingTool(PDFTool):
    """Tool for intelligent routing of PDF processing tasks."""
    
    name = "smart_routing"
    description = "Intelligently route PDF processing tasks to optimal tools and parameters"
    input_schema = SmartRoutingInput
    output_schema = SmartRoutingOutput
    
    def execute(self, input_data: SmartRoutingInput) -> SmartRoutingOutput:
        """Execute smart routing."""
        try:
            # Analyze the task description
            task_analysis = self._analyze_task(input_data.task_description)
            
            # Analyze the document first
            analyzer = AnalyzeDocumentTool()
            doc_analysis = analyzer.execute(AnalyzeDocumentInput(
                file_path=input_data.file_path,
                analysis_depth="standard"
            ))
            
            if not doc_analysis.success:
                raise Exception(f"Document analysis failed: {doc_analysis.error}")
            
            # Determine best tool and parameters
            recommendation = self._generate_recommendation(
                task_analysis,
                doc_analysis,
                input_data.preferred_methods,
                input_data.quality_priority,
                input_data.output_format
            )
            
            return SmartRoutingOutput(
                success=True,
                recommended_tool=recommendation["tool"],
                tool_parameters=recommendation["parameters"],
                alternative_tools=recommendation["alternatives"],
                reasoning=recommendation["reasoning"],
                estimated_processing_time=recommendation["time_estimate"],
                metadata={
                    "file_path": input_data.file_path,
                    "task_description": input_data.task_description,
                    "quality_priority": input_data.quality_priority
                }
            )
            
        except Exception as e:
            logger.error(f"Smart routing failed: {str(e)}")
            return SmartRoutingOutput(
                success=False,
                error=str(e),
                recommended_tool="extract_text",
                tool_parameters={},
                alternative_tools=[],
                reasoning="Fallback to basic text extraction due to error",
                estimated_processing_time="unknown",
                metadata={"file_path": input_data.file_path}
            )
    
    def _analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze task description to understand requirements."""
        task_lower = task_description.lower()
        
        analysis = {
            "primary_goal": "text_extraction",
            "data_types": [],
            "quality_requirements": "standard",
            "output_preferences": []
        }
        
        # Detect primary goal
        if any(word in task_lower for word in ['table', 'tabular', 'data']):
            analysis["primary_goal"] = "table_extraction"
            analysis["data_types"].append("tables")
        
        if any(word in task_lower for word in ['image', 'figure', 'chart', 'diagram']):
            analysis["data_types"].append("images")
        
        if any(word in task_lower for word in ['formula', 'equation', 'math']):
            analysis["data_types"].append("formulas")
        
        if any(word in task_lower for word in ['text', 'content', 'extract']):
            analysis["data_types"].append("text")
        
        # Detect quality requirements
        if any(word in task_lower for word in ['accurate', 'precise', 'high quality']):
            analysis["quality_requirements"] = "high"
        elif any(word in task_lower for word in ['fast', 'quick', 'speed']):
            analysis["quality_requirements"] = "fast"
        
        return analysis
    
    def _generate_recommendation(self, task_analysis: Dict[str, Any], 
                               doc_analysis: AnalyzeDocumentOutput,
                               preferred_methods: Optional[List[str]],
                               quality_priority: str,
                               output_format: str) -> Dict[str, Any]:
        """Generate tool recommendation based on analysis."""
        
        # Start with document analysis recommendations
        recommended_tools = doc_analysis.recommended_tools
        
        # Refine based on task analysis
        if task_analysis["primary_goal"] == "table_extraction":
            primary_tool = "extract_tables"
        elif "formulas" in task_analysis["data_types"]:
            primary_tool = "extract_formulas"
        elif "images" in task_analysis["data_types"]:
            primary_tool = "extract_images"
        else:
            primary_tool = "extract_text"
        
        # Determine parameters based on quality priority
        parameters = self._get_optimal_parameters(
            primary_tool, 
            doc_analysis.document_type,
            quality_priority,
            output_format
        )
        
        # Generate alternatives
        alternatives = self._get_alternative_tools(
            primary_tool, 
            recommended_tools,
            preferred_methods
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            primary_tool,
            doc_analysis.document_type,
            task_analysis,
            quality_priority
        )
        
        # Estimate processing time
        time_estimate = self._estimate_processing_time(
            primary_tool,
            doc_analysis.document_info.get("page_count", 1),
            quality_priority
        )
        
        return {
            "tool": primary_tool,
            "parameters": parameters,
            "alternatives": alternatives,
            "reasoning": reasoning,
            "time_estimate": time_estimate
        }
    
    def _get_optimal_parameters(self, tool: str, doc_type: str, 
                              quality_priority: str, output_format: str) -> Dict[str, Any]:
        """Get optimal parameters for the tool."""
        base_params = {"output_format": output_format}
        
        if tool == "extract_text":
            if quality_priority == "speed":
                base_params.update({"method": "pymupdf", "preserve_layout": False})
            elif quality_priority == "accuracy":
                base_params.update({"method": "pdfplumber", "preserve_layout": True})
            else:
                base_params.update({"method": "pymupdf", "preserve_layout": True})
        
        elif tool == "extract_tables":
            if quality_priority == "speed":
                base_params.update({"method": "pdfplumber"})
            elif quality_priority == "accuracy":
                base_params.update({"method": "camelot"})
            else:
                base_params.update({"method": "tabula"})
        
        elif tool == "extract_formulas":
            if quality_priority == "accuracy":
                base_params.update({"method": "mathpix", "confidence_threshold": 0.9})
            else:
                base_params.update({"method": "latex-ocr", "confidence_threshold": 0.8})
        
        return base_params
    
    def _get_alternative_tools(self, primary_tool: str, recommended_tools: List[str],
                             preferred_methods: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get alternative tools and their parameters."""
        alternatives = []
        
        for tool in recommended_tools:
            if tool != primary_tool:
                alt_params = self._get_optimal_parameters(tool, "article", "balanced", "json")
                alternatives.append({
                    "tool": tool,
                    "parameters": alt_params,
                    "reason": f"Alternative for {tool.replace('_', ' ')}"
                })
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    def _generate_reasoning(self, tool: str, doc_type: str, 
                          task_analysis: Dict[str, Any], quality_priority: str) -> str:
        """Generate reasoning for the recommendation."""
        reasoning_parts = [
            f"Recommended '{tool}' based on document type '{doc_type}'",
            f"Task analysis indicates primary goal: {task_analysis['primary_goal']}",
            f"Quality priority set to '{quality_priority}'"
        ]
        
        if task_analysis["data_types"]:
            reasoning_parts.append(f"Detected data types: {', '.join(task_analysis['data_types'])}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _estimate_processing_time(self, tool: str, page_count: int, 
                                quality_priority: str) -> str:
        """Estimate processing time."""
        base_time_per_page = {
            "extract_text": 0.5,
            "extract_tables": 2.0,
            "extract_formulas": 5.0,
            "extract_images": 1.0
        }
        
        time_per_page = base_time_per_page.get(tool, 1.0)
        
        if quality_priority == "speed":
            time_per_page *= 0.5
        elif quality_priority == "accuracy":
            time_per_page *= 2.0
        
        total_time = time_per_page * page_count
        
        if total_time < 10:
            return "< 10 seconds"
        elif total_time < 60:
            return f"~{int(total_time)} seconds"
        elif total_time < 3600:
            return f"~{int(total_time/60)} minutes"
        else:
            return f"~{int(total_time/3600)} hours"