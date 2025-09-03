#!/usr/bin/env python3
"""
Full Pipeline Processing Tools

Implements comprehensive PDF processing pipelines that combine multiple extraction
methods and provide intelligent processing workflows.

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
import shutil

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException

# Import other tools for pipeline processing
from .text_extraction import ReadTextTool, ExtractMetadataTool
from .table_extraction import ExtractTablesTool
from .ocr_processing import ProcessOCRTool
from .formula_recognition import ExtractFormulasTool
from .pdf_analysis import AnalyzePDFTool, DetectPDFTypeTool


class FullPipelineTool(PDFTool):
    """Complete PDF processing pipeline."""
    
    def __init__(self):
        super().__init__(
            name="full_pipeline",
            description="Execute complete PDF processing pipeline with text, tables, OCR, and formula extraction",
            version="1.0.0"
        )
        
        # Initialize sub-tools
        self.text_tool = ReadTextTool()
        self.metadata_tool = ExtractMetadataTool()
        self.table_tool = ExtractTablesTool()
        self.ocr_tool = ProcessOCRTool()
        self.formula_tool = ExtractFormulasTool()
        self.analysis_tool = AnalyzePDFTool()
        self.type_detection_tool = DetectPDFTypeTool()
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to process"
                },
                "processing_mode": {
                    "type": "string",
                    "enum": ["auto", "comprehensive", "fast", "custom"],
                    "default": "auto",
                    "description": "Processing mode: auto (intelligent), comprehensive (all methods), fast (basic only), custom (user-defined)"
                },
                "enable_text_extraction": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable text extraction"
                },
                "enable_table_extraction": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable table extraction"
                },
                "enable_ocr": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable OCR processing"
                },
                "enable_formula_extraction": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable formula extraction"
                },
                "enable_metadata_extraction": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable metadata extraction"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "markdown", "text", "structured"],
                    "default": "structured",
                    "description": "Output format for results"
                },
                "save_intermediate_results": {
                    "type": "boolean",
                    "default": False,
                    "description": "Save intermediate processing results to files"
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to save output files (required if save_intermediate_results is true)"
                },
                "quality_threshold": {
                    "type": "number",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Quality threshold for processing decisions"
                },
                "parallel_processing": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable parallel processing where possible"
                },
                "include_analysis": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include document analysis and recommendations"
                },
                "custom_config": {
                    "type": "object",
                    "description": "Custom configuration for individual tools",
                    "properties": {
                        "text_extraction": {"type": "object"},
                        "table_extraction": {"type": "object"},
                        "ocr_processing": {"type": "object"},
                        "formula_extraction": {"type": "object"}
                    }
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        processing_mode = kwargs.get("processing_mode", "auto")
        output_format = kwargs.get("output_format", "structured")
        save_intermediate = kwargs.get("save_intermediate_results", False)
        output_directory = kwargs.get("output_directory")
        quality_threshold = kwargs.get("quality_threshold", 0.7)
        parallel_processing = kwargs.get("parallel_processing", True)
        include_analysis = kwargs.get("include_analysis", True)
        custom_config = kwargs.get("custom_config", {})
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Setup output directory if needed
            if save_intermediate:
                if not output_directory:
                    output_directory = tempfile.mkdtemp(prefix="pdf_pipeline_")
                else:
                    output_dir = Path(output_directory)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_directory = str(output_dir)
            
            start_time = datetime.now()
            
            # Initialize pipeline results
            pipeline_results = {
                "file_info": {
                    "path": str(pdf_path),
                    "name": pdf_path.name,
                    "size_mb": pdf_path.stat().st_size / (1024 * 1024)
                },
                "processing_config": {
                    "mode": processing_mode,
                    "output_format": output_format,
                    "quality_threshold": quality_threshold,
                    "parallel_processing": parallel_processing
                },
                "start_time": start_time.isoformat(),
                "results": {},
                "errors": [],
                "warnings": []
            }
            
            # Step 1: Document analysis and type detection
            if include_analysis or processing_mode == "auto":
                self.logger.info("Starting document analysis...")
                analysis_result = await self._run_analysis(str(pdf_path))
                pipeline_results["results"]["analysis"] = analysis_result
                
                # Determine processing strategy based on analysis
                if processing_mode == "auto":
                    processing_strategy = self._determine_processing_strategy(
                        analysis_result, quality_threshold
                    )
                    pipeline_results["processing_strategy"] = processing_strategy
                else:
                    processing_strategy = self._get_manual_strategy(kwargs)
                    pipeline_results["processing_strategy"] = processing_strategy
            else:
                processing_strategy = self._get_manual_strategy(kwargs)
                pipeline_results["processing_strategy"] = processing_strategy
            
            # Step 2: Execute processing pipeline
            self.logger.info("Executing processing pipeline...")
            processing_results = await self._execute_processing_pipeline(
                str(pdf_path), processing_strategy, custom_config, parallel_processing
            )
            pipeline_results["results"].update(processing_results)
            
            # Step 3: Post-processing and integration
            self.logger.info("Post-processing results...")
            integrated_results = await self._integrate_results(
                pipeline_results["results"], processing_strategy
            )
            pipeline_results["integrated_results"] = integrated_results
            
            # Step 4: Save intermediate results if requested
            if save_intermediate:
                await self._save_intermediate_results(
                    pipeline_results, output_directory
                )
                pipeline_results["output_directory"] = output_directory
            
            end_time = datetime.now()
            pipeline_results["end_time"] = end_time.isoformat()
            pipeline_results["total_processing_time"] = (end_time - start_time).total_seconds()
            
            # Format output
            content = await self._format_pipeline_output(
                pipeline_results, output_format
            )
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            content = [create_error_content(f"Pipeline processing failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    async def _run_analysis(self, file_path: str) -> Dict[str, Any]:
        """Run document analysis."""
        try:
            # Run type detection
            type_result = await self.type_detection_tool.execute(
                file_path=file_path,
                quick_analysis=True,
                include_confidence=True
            )
            
            # Run comprehensive analysis
            analysis_result = await self.analysis_tool.execute(
                file_path=file_path,
                analysis_depth="detailed",
                sample_pages=5
            )
            
            return {
                "type_detection": self._extract_tool_result_data(type_result),
                "comprehensive_analysis": self._extract_tool_result_data(analysis_result)
            }
        
        except Exception as e:
            self.logger.warning(f"Analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_tool_result_data(self, tool_result: MCPToolResult) -> Any:
        """Extract data from tool result."""
        if tool_result.isError:
            return {"error": "Tool execution failed"}
        
        # Try to extract JSON data from content
        for content_item in tool_result.content:
            if hasattr(content_item, 'text') and content_item.text:
                # Look for JSON blocks in text
                text = content_item.text
                if '```json' in text:
                    start = text.find('```json') + 7
                    end = text.find('```', start)
                    if end > start:
                        try:
                            return json.loads(text[start:end].strip())
                        except json.JSONDecodeError:
                            pass
        
        return {"raw_content": [item.__dict__ for item in tool_result.content]}
    
    def _determine_processing_strategy(self, analysis_result: Dict[str, Any], quality_threshold: float) -> Dict[str, Any]:
        """Determine processing strategy based on analysis."""
        strategy = {
            "enable_text_extraction": True,
            "enable_table_extraction": False,
            "enable_ocr": False,
            "enable_formula_extraction": False,
            "enable_metadata_extraction": True,
            "processing_order": [],
            "reasoning": []
        }
        
        try:
            # Extract type detection info
            type_detection = analysis_result.get("type_detection", {})
            comprehensive = analysis_result.get("comprehensive_analysis", {})
            
            # Determine document type
            doc_type = "unknown"
            if isinstance(type_detection, dict):
                doc_type_info = type_detection.get("document_type", {})
                if isinstance(doc_type_info, dict):
                    doc_type = doc_type_info.get("primary_type", "unknown")
            
            # Text extraction strategy
            if doc_type in ["text_document", "mixed_content"]:
                strategy["enable_text_extraction"] = True
                strategy["processing_order"].append("text_extraction")
                strategy["reasoning"].append("Document contains extractable text")
            
            # OCR strategy
            if doc_type in ["scanned_document", "image_document"]:
                strategy["enable_ocr"] = True
                strategy["processing_order"].append("ocr_processing")
                strategy["reasoning"].append("Document requires OCR for text extraction")
            
            # Table extraction strategy
            if isinstance(comprehensive, dict):
                content_analysis = comprehensive.get("content_analysis", {})
                if isinstance(content_analysis, dict):
                    text_analysis = content_analysis.get("text_analysis", {})
                    if isinstance(text_analysis, dict):
                        table_indicators = text_analysis.get("table_indicators", {})
                        if isinstance(table_indicators, dict) and any(table_indicators.values()):
                            strategy["enable_table_extraction"] = True
                            strategy["processing_order"].append("table_extraction")
                            strategy["reasoning"].append("Document contains tabular data")
            
            # Formula extraction strategy
            if isinstance(comprehensive, dict):
                content_analysis = comprehensive.get("content_analysis", {})
                if isinstance(content_analysis, dict):
                    text_analysis = content_analysis.get("text_analysis", {})
                    if isinstance(text_analysis, dict):
                        math_content = text_analysis.get("mathematical_content", {})
                        if isinstance(math_content, dict) and any(math_content.values()):
                            strategy["enable_formula_extraction"] = True
                            strategy["processing_order"].append("formula_extraction")
                            strategy["reasoning"].append("Document contains mathematical formulas")
            
            # Ensure processing order
            if not strategy["processing_order"]:
                strategy["processing_order"] = ["text_extraction"]
                strategy["reasoning"].append("Default text extraction fallback")
        
        except Exception as e:
            self.logger.warning(f"Strategy determination failed: {e}")
            # Fallback strategy
            strategy = {
                "enable_text_extraction": True,
                "enable_table_extraction": True,
                "enable_ocr": True,
                "enable_formula_extraction": True,
                "enable_metadata_extraction": True,
                "processing_order": ["text_extraction", "table_extraction", "ocr_processing", "formula_extraction"],
                "reasoning": ["Fallback to comprehensive processing due to analysis error"]
            }
        
        return strategy
    
    def _get_manual_strategy(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get manual processing strategy from user parameters."""
        strategy = {
            "enable_text_extraction": kwargs.get("enable_text_extraction", True),
            "enable_table_extraction": kwargs.get("enable_table_extraction", True),
            "enable_ocr": kwargs.get("enable_ocr", True),
            "enable_formula_extraction": kwargs.get("enable_formula_extraction", True),
            "enable_metadata_extraction": kwargs.get("enable_metadata_extraction", True),
            "processing_order": [],
            "reasoning": ["Manual configuration by user"]
        }
        
        # Build processing order
        if strategy["enable_text_extraction"]:
            strategy["processing_order"].append("text_extraction")
        if strategy["enable_table_extraction"]:
            strategy["processing_order"].append("table_extraction")
        if strategy["enable_ocr"]:
            strategy["processing_order"].append("ocr_processing")
        if strategy["enable_formula_extraction"]:
            strategy["processing_order"].append("formula_extraction")
        if strategy["enable_metadata_extraction"]:
            strategy["processing_order"].append("metadata_extraction")
        
        return strategy
    
    async def _execute_processing_pipeline(
        self,
        file_path: str,
        strategy: Dict[str, Any],
        custom_config: Dict[str, Any],
        parallel_processing: bool
    ) -> Dict[str, Any]:
        """Execute the processing pipeline."""
        results = {}
        
        if parallel_processing:
            # Execute compatible operations in parallel
            tasks = []
            
            if strategy.get("enable_text_extraction", False):
                tasks.append(self._run_text_extraction(file_path, custom_config.get("text_extraction", {})))
            
            if strategy.get("enable_metadata_extraction", False):
                tasks.append(self._run_metadata_extraction(file_path))
            
            if strategy.get("enable_table_extraction", False):
                tasks.append(self._run_table_extraction(file_path, custom_config.get("table_extraction", {})))
            
            if strategy.get("enable_ocr", False):
                tasks.append(self._run_ocr_processing(file_path, custom_config.get("ocr_processing", {})))
            
            if strategy.get("enable_formula_extraction", False):
                tasks.append(self._run_formula_extraction(file_path, custom_config.get("formula_extraction", {})))
            
            # Execute all tasks in parallel
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                task_names = []
                if strategy.get("enable_text_extraction", False):
                    task_names.append("text_extraction")
                if strategy.get("enable_metadata_extraction", False):
                    task_names.append("metadata_extraction")
                if strategy.get("enable_table_extraction", False):
                    task_names.append("table_extraction")
                if strategy.get("enable_ocr", False):
                    task_names.append("ocr_processing")
                if strategy.get("enable_formula_extraction", False):
                    task_names.append("formula_extraction")
                
                for i, result in enumerate(task_results):
                    task_name = task_names[i] if i < len(task_names) else f"task_{i}"
                    if isinstance(result, Exception):
                        results[task_name] = {"error": str(result)}
                        self.logger.error(f"{task_name} failed: {result}")
                    else:
                        results[task_name] = result
        
        else:
            # Execute operations sequentially
            for operation in strategy.get("processing_order", []):
                try:
                    if operation == "text_extraction" and strategy.get("enable_text_extraction", False):
                        results["text_extraction"] = await self._run_text_extraction(
                            file_path, custom_config.get("text_extraction", {})
                        )
                    
                    elif operation == "metadata_extraction" and strategy.get("enable_metadata_extraction", False):
                        results["metadata_extraction"] = await self._run_metadata_extraction(file_path)
                    
                    elif operation == "table_extraction" and strategy.get("enable_table_extraction", False):
                        results["table_extraction"] = await self._run_table_extraction(
                            file_path, custom_config.get("table_extraction", {})
                        )
                    
                    elif operation == "ocr_processing" and strategy.get("enable_ocr", False):
                        results["ocr_processing"] = await self._run_ocr_processing(
                            file_path, custom_config.get("ocr_processing", {})
                        )
                    
                    elif operation == "formula_extraction" and strategy.get("enable_formula_extraction", False):
                        results["formula_extraction"] = await self._run_formula_extraction(
                            file_path, custom_config.get("formula_extraction", {})
                        )
                
                except Exception as e:
                    results[operation] = {"error": str(e)}
                    self.logger.error(f"{operation} failed: {e}")
        
        return results
    
    async def _run_text_extraction(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run text extraction."""
        try:
            result = await self.text_tool.execute(
                file_path=file_path,
                method=config.get("method", "pymupdf"),
                preserve_layout=config.get("preserve_layout", True),
                include_metadata=config.get("include_metadata", True),
                extract_annotations=config.get("extract_annotations", True)
            )
            return self._extract_tool_result_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def _run_metadata_extraction(self, file_path: str) -> Dict[str, Any]:
        """Run metadata extraction."""
        try:
            result = await self.metadata_tool.execute(
                file_path=file_path,
                include_technical_details=True,
                extract_xmp=True
            )
            return self._extract_tool_result_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def _run_table_extraction(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run table extraction."""
        try:
            result = await self.table_tool.execute(
                file_path=file_path,
                method=config.get("method", "camelot"),
                output_format=config.get("output_format", "json"),
                pages=config.get("pages", "all")
            )
            return self._extract_tool_result_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def _run_ocr_processing(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR processing."""
        try:
            result = await self.ocr_tool.execute(
                file_path=file_path,
                ocr_engine=config.get("ocr_engine", "tesseract"),
                language=config.get("language", "eng"),
                output_format=config.get("output_format", "text")
            )
            return self._extract_tool_result_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def _run_formula_extraction(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run formula extraction."""
        try:
            result = await self.formula_tool.execute(
                file_path=file_path,
                recognition_method=config.get("recognition_method", "latex_ocr"),
                output_format=config.get("output_format", "latex")
            )
            return self._extract_tool_result_data(result)
        except Exception as e:
            return {"error": str(e)}
    
    async def _integrate_results(self, results: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate and cross-reference results from different processing methods."""
        integrated = {
            "summary": {},
            "content": {},
            "statistics": {},
            "quality_metrics": {}
        }
        
        # Collect text content from all sources
        all_text = []
        text_sources = []
        
        if "text_extraction" in results and not results["text_extraction"].get("error"):
            text_data = results["text_extraction"]
            if isinstance(text_data, dict) and "text" in text_data:
                all_text.append(text_data["text"])
                text_sources.append("native_extraction")
        
        if "ocr_processing" in results and not results["ocr_processing"].get("error"):
            ocr_data = results["ocr_processing"]
            if isinstance(ocr_data, dict) and "text" in ocr_data:
                all_text.append(ocr_data["text"])
                text_sources.append("ocr")
        
        # Combine text content
        if all_text:
            # Use the longest text as primary
            primary_text = max(all_text, key=len)
            integrated["content"]["text"] = primary_text
            integrated["content"]["text_sources"] = text_sources
            integrated["statistics"]["total_characters"] = len(primary_text)
            integrated["statistics"]["total_words"] = len(primary_text.split())
        
        # Collect tables
        if "table_extraction" in results and not results["table_extraction"].get("error"):
            table_data = results["table_extraction"]
            if isinstance(table_data, dict):
                integrated["content"]["tables"] = table_data.get("tables", [])
                integrated["statistics"]["table_count"] = len(table_data.get("tables", []))
        
        # Collect formulas
        if "formula_extraction" in results and not results["formula_extraction"].get("error"):
            formula_data = results["formula_extraction"]
            if isinstance(formula_data, dict):
                integrated["content"]["formulas"] = formula_data.get("formulas", [])
                integrated["statistics"]["formula_count"] = len(formula_data.get("formulas", []))
        
        # Collect metadata
        if "metadata_extraction" in results and not results["metadata_extraction"].get("error"):
            metadata = results["metadata_extraction"]
            if isinstance(metadata, dict):
                integrated["content"]["metadata"] = metadata
        
        # Calculate quality metrics
        integrated["quality_metrics"] = self._calculate_integration_quality(results)
        
        # Create summary
        integrated["summary"] = {
            "processing_methods_used": [method for method in results.keys() if not results[method].get("error")],
            "processing_methods_failed": [method for method in results.keys() if results[method].get("error")],
            "content_types_found": list(integrated["content"].keys()),
            "overall_success": len([r for r in results.values() if not r.get("error")]) > 0
        }
        
        return integrated
    
    def _calculate_integration_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for integrated results."""
        metrics = {
            "extraction_success_rate": 0.0,
            "content_completeness": 0.0,
            "method_agreement": 0.0
        }
        
        # Calculate success rate
        total_methods = len(results)
        successful_methods = len([r for r in results.values() if not r.get("error")])
        metrics["extraction_success_rate"] = successful_methods / total_methods if total_methods > 0 else 0.0
        
        # Calculate content completeness (simplified)
        content_types = 0
        if any("text" in str(r) for r in results.values() if not r.get("error")):
            content_types += 1
        if any("table" in str(r) for r in results.values() if not r.get("error")):
            content_types += 1
        if any("formula" in str(r) for r in results.values() if not r.get("error")):
            content_types += 1
        
        metrics["content_completeness"] = content_types / 3.0  # Assuming 3 main content types
        
        # Method agreement (simplified - would need more sophisticated comparison)
        metrics["method_agreement"] = 0.8  # Placeholder
        
        return metrics
    
    async def _save_intermediate_results(self, pipeline_results: Dict[str, Any], output_directory: str) -> None:
        """Save intermediate results to files."""
        output_dir = Path(output_directory)
        
        # Save full pipeline results
        with open(output_dir / "pipeline_results.json", "w", encoding="utf-8") as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save individual results
        results = pipeline_results.get("results", {})
        
        for method, result in results.items():
            if not result.get("error"):
                result_file = output_dir / f"{method}_result.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        # Save integrated results
        integrated = pipeline_results.get("integrated_results", {})
        if integrated:
            with open(output_dir / "integrated_results.json", "w", encoding="utf-8") as f:
                json.dump(integrated, f, indent=2, ensure_ascii=False, default=str)
            
            # Save text content separately
            content = integrated.get("content", {})
            if "text" in content:
                with open(output_dir / "extracted_text.txt", "w", encoding="utf-8") as f:
                    f.write(content["text"])
            
            # Save tables as CSV
            if "tables" in content:
                tables_dir = output_dir / "tables"
                tables_dir.mkdir(exist_ok=True)
                
                for i, table in enumerate(content["tables"]):
                    if isinstance(table, dict) and "data" in table:
                        table_file = tables_dir / f"table_{i+1}.json"
                        with open(table_file, "w", encoding="utf-8") as f:
                            json.dump(table, f, indent=2, ensure_ascii=False)
    
    async def _format_pipeline_output(self, pipeline_results: Dict[str, Any], output_format: str) -> List[Any]:
        """Format pipeline output according to specified format."""
        content = []
        
        if output_format == "json":
            import json
            content.append(create_text_content(json.dumps(pipeline_results, indent=2)))
        
        elif output_format == "text":
            text_output = await self._create_text_summary(pipeline_results)
            content.append(create_text_content(text_output))
        
        elif output_format == "markdown":
            markdown_output = await self._create_markdown_summary(pipeline_results)
            content.append(create_text_content(markdown_output))
        
        else:  # structured
            # Create structured output with summary and details
            summary = await self._create_executive_summary(pipeline_results)
            content.append(create_text_content(f"Pipeline Processing Summary:\n{summary}"))
            
            # Add integrated results
            integrated = pipeline_results.get("integrated_results", {})
            if integrated:
                import json
                content.append(create_text_content(json.dumps(integrated, indent=2)))
            
            # Add processing details
            processing_details = await self._create_processing_details(pipeline_results)
            content.append(create_text_content(f"Processing Details:\n{processing_details}"))
        
        return content
    
    async def _create_executive_summary(self, pipeline_results: Dict[str, Any]) -> str:
        """Create executive summary of pipeline processing."""
        file_info = pipeline_results.get("file_info", {})
        config = pipeline_results.get("processing_config", {})
        integrated = pipeline_results.get("integrated_results", {})
        
        summary_parts = []
        
        # File information
        summary_parts.append(f"**File:** {file_info.get('name', 'Unknown')}")
        summary_parts.append(f"**Size:** {file_info.get('size_mb', 0):.1f} MB")
        summary_parts.append(f"**Processing Mode:** {config.get('mode', 'unknown')}")
        
        # Processing results
        summary = integrated.get("summary", {})
        if summary:
            methods_used = summary.get("processing_methods_used", [])
            methods_failed = summary.get("processing_methods_failed", [])
            
            if methods_used:
                summary_parts.append(f"**Successful Methods:** {', '.join(methods_used)}")
            if methods_failed:
                summary_parts.append(f"**Failed Methods:** {', '.join(methods_failed)}")
        
        # Content statistics
        stats = integrated.get("statistics", {})
        if stats:
            content_stats = []
            if "total_characters" in stats:
                content_stats.append(f"{stats['total_characters']:,} characters")
            if "total_words" in stats:
                content_stats.append(f"{stats['total_words']:,} words")
            if "table_count" in stats:
                content_stats.append(f"{stats['table_count']} tables")
            if "formula_count" in stats:
                content_stats.append(f"{stats['formula_count']} formulas")
            
            if content_stats:
                summary_parts.append(f"**Content Extracted:** {', '.join(content_stats)}")
        
        # Quality metrics
        quality = integrated.get("quality_metrics", {})
        if quality:
            success_rate = quality.get("extraction_success_rate", 0)
            summary_parts.append(f"**Success Rate:** {success_rate:.1%}")
        
        # Processing time
        processing_time = pipeline_results.get("total_processing_time", 0)
        summary_parts.append(f"**Processing Time:** {processing_time:.1f} seconds")
        
        return "\n".join(summary_parts)
    
    async def _create_processing_details(self, pipeline_results: Dict[str, Any]) -> str:
        """Create detailed processing information."""
        details = []
        
        # Processing strategy
        strategy = pipeline_results.get("processing_strategy", {})
        if strategy:
            details.append("**Processing Strategy:**")
            reasoning = strategy.get("reasoning", [])
            for reason in reasoning:
                details.append(f"- {reason}")
            
            order = strategy.get("processing_order", [])
            if order:
                details.append(f"\n**Processing Order:** {' → '.join(order)}")
        
        # Individual method results
        results = pipeline_results.get("results", {})
        if results:
            details.append("\n**Method Results:**")
            for method, result in results.items():
                if method == "analysis":
                    continue  # Skip analysis in details
                
                if result.get("error"):
                    details.append(f"- {method}: ❌ Failed ({result['error']})")
                else:
                    details.append(f"- {method}: ✅ Success")
        
        # Output information
        output_dir = pipeline_results.get("output_directory")
        if output_dir:
            details.append(f"\n**Output Directory:** {output_dir}")
        
        return "\n".join(details)
    
    async def _create_text_summary(self, pipeline_results: Dict[str, Any]) -> str:
        """Create plain text summary."""
        summary = await self._create_executive_summary(pipeline_results)
        details = await self._create_processing_details(pipeline_results)
        
        return f"{summary}\n\n{details}"
    
    async def _create_markdown_summary(self, pipeline_results: Dict[str, Any]) -> str:
        """Create markdown summary."""
        summary = await self._create_executive_summary(pipeline_results)
        details = await self._create_processing_details(pipeline_results)
        
        markdown = f"# PDF Processing Pipeline Results\n\n{summary}\n\n{details}"
        
        # Add content preview
        integrated = pipeline_results.get("integrated_results", {})
        content = integrated.get("content", {})
        
        if "text" in content:
            text_preview = content["text"][:500] + "..." if len(content["text"]) > 500 else content["text"]
            markdown += f"\n\n## Text Content Preview\n\n```\n{text_preview}\n```"
        
        if "tables" in content and content["tables"]:
            markdown += f"\n\n## Tables Found\n\n{len(content['tables'])} tables extracted"
        
        if "formulas" in content and content["formulas"]:
            markdown += f"\n\n## Formulas Found\n\n{len(content['formulas'])} formulas extracted"
        
        return markdown


class SmartProcessingTool(PDFTool):
    """Smart PDF processing with adaptive strategies."""
    
    def __init__(self):
        super().__init__(
            name="smart_processing",
            description="Intelligent PDF processing that adapts strategy based on document characteristics",
            version="1.0.0"
        )
        
        self.full_pipeline = FullPipelineTool()
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to process"
                },
                "target_content": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["text", "tables", "formulas", "images", "metadata", "structure"]
                    },
                    "default": ["text"],
                    "description": "Types of content to extract"
                },
                "quality_preference": {
                    "type": "string",
                    "enum": ["speed", "balanced", "accuracy"],
                    "default": "balanced",
                    "description": "Processing quality preference"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "markdown", "text"],
                    "default": "markdown",
                    "description": "Output format"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence threshold for results"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        target_content = kwargs.get("target_content", ["text"])
        quality_preference = kwargs.get("quality_preference", "balanced")
        output_format = kwargs.get("output_format", "markdown")
        confidence_threshold = kwargs.get("confidence_threshold", 0.8)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Create adaptive processing configuration
            processing_config = self._create_adaptive_config(
                target_content, quality_preference, confidence_threshold
            )
            
            # Execute full pipeline with adaptive configuration
            result = await self.full_pipeline.execute(
                file_path=str(pdf_path),
                processing_mode="auto",
                output_format=output_format,
                quality_threshold=confidence_threshold,
                **processing_config
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Smart processing failed: {e}")
            content = [create_error_content(f"Smart processing failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _create_adaptive_config(self, target_content: List[str], quality_preference: str, confidence_threshold: float) -> Dict[str, Any]:
        """Create adaptive processing configuration."""
        config = {
            "enable_text_extraction": "text" in target_content,
            "enable_table_extraction": "tables" in target_content,
            "enable_formula_extraction": "formulas" in target_content,
            "enable_metadata_extraction": "metadata" in target_content,
            "enable_ocr": True,  # Always enable for adaptive processing
            "parallel_processing": quality_preference != "speed",
            "include_analysis": True
        }
        
        # Quality-based adjustments
        if quality_preference == "speed":
            config["custom_config"] = {
                "text_extraction": {"method": "pymupdf"},
                "table_extraction": {"method": "pdfplumber"},
                "ocr_processing": {"ocr_engine": "tesseract"}
            }
        elif quality_preference == "accuracy":
            config["custom_config"] = {
                "text_extraction": {"preserve_layout": True, "include_metadata": True},
                "table_extraction": {"method": "camelot"},
                "ocr_processing": {"ocr_engine": "easyocr"}
            }
        else:  # balanced
            config["custom_config"] = {
                "text_extraction": {"method": "pymupdf", "preserve_layout": True},
                "table_extraction": {"method": "camelot"},
                "ocr_processing": {"ocr_engine": "tesseract"}
            }
        
        return config