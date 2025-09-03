#!/usr/bin/env python3
"""
PDF-MCP Server Advanced Usage Example

This script demonstrates advanced usage patterns for the PDF-MCP server,
including batch processing, custom configurations, and error handling.

Author: PDF-MCP Team
License: MIT
"""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_mcp_server.main import PDFMCPServer
from src.pdf_mcp_server.mcp.protocol import MCPProtocolHandler
from src.pdf_mcp_server.core.exceptions import PDFProcessingError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pdf_mcp_advanced')


class AdvancedPDFProcessor:
    """Advanced PDF processing with custom configurations and batch operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.server = None
        self.config_path = config_path
        self.results_cache = {}
    
    async def initialize(self):
        """Initialize the server with custom configuration."""
        logger.info("Initializing advanced PDF processor...")
        
        # Create custom configuration if needed
        if self.config_path:
            self.server = PDFMCPServer(config_file=self.config_path)
        else:
            self.server = PDFMCPServer()
        
        await self.server.initialize()
        logger.info(f"Server initialized with {len(self.server.server.tools)} tools")
    
    async def process_document_by_type(self, pdf_path: str, document_type: str = "auto") -> Dict[str, Any]:
        """Process a PDF document based on its detected or specified type."""
        logger.info(f"Processing document: {pdf_path} (type: {document_type})")
        
        try:
            # First, analyze the document if type is auto
            if document_type == "auto":
                document_type = await self._detect_document_type(pdf_path)
                logger.info(f"Detected document type: {document_type}")
            
            # Process based on document type
            if document_type == "academic_paper":
                return await self._process_academic_paper(pdf_path)
            elif document_type == "financial_report":
                return await self._process_financial_report(pdf_path)
            elif document_type == "technical_manual":
                return await self._process_technical_manual(pdf_path)
            elif document_type == "form":
                return await self._process_form(pdf_path)
            else:
                return await self._process_general_document(pdf_path)
        
        except Exception as e:
            logger.error(f"Failed to process document {pdf_path}: {e}")
            return {"error": str(e), "file_path": pdf_path}
    
    async def _detect_document_type(self, pdf_path: str) -> str:
        """Detect the type of document based on analysis."""
        # Find analysis tool
        analysis_tool = self._find_tool("analyze")
        if not analysis_tool:
            return "general"
        
        try:
            result = await analysis_tool.execute({
                "file_path": pdf_path,
                "analyze_structure": True,
                "detect_content_types": True
            })
            
            # Simple heuristics for document type detection
            if isinstance(result, dict):
                if result.get("has_formulas", False) and result.get("has_references", False):
                    return "academic_paper"
                elif result.get("has_tables", False) and "financial" in str(result).lower():
                    return "financial_report"
                elif result.get("has_forms", False):
                    return "form"
                elif result.get("has_technical_diagrams", False):
                    return "technical_manual"
            
            return "general"
        
        except Exception:
            return "general"
    
    async def _process_academic_paper(self, pdf_path: str) -> Dict[str, Any]:
        """Process an academic paper with focus on text, formulas, and references."""
        logger.info("Processing as academic paper")
        
        pipeline_tool = self._find_tool("pipeline")
        if not pipeline_tool:
            raise PDFProcessingError("Pipeline tool not available")
        
        return await pipeline_tool.execute({
            "file_path": pdf_path,
            "mode": "comprehensive",
            "extract_text": True,
            "extract_tables": True,
            "apply_ocr": False,  # Usually not needed for academic papers
            "extract_formulas": True,
            "analyze_document": True,
            "output_format": "json",
            "preserve_structure": True,
            "extract_references": True
        })
    
    async def _process_financial_report(self, pdf_path: str) -> Dict[str, Any]:
        """Process a financial report with focus on tables and numbers."""
        logger.info("Processing as financial report")
        
        pipeline_tool = self._find_tool("pipeline")
        if not pipeline_tool:
            raise PDFProcessingError("Pipeline tool not available")
        
        return await pipeline_tool.execute({
            "file_path": pdf_path,
            "mode": "comprehensive",
            "extract_text": True,
            "extract_tables": True,
            "apply_ocr": False,
            "extract_formulas": False,
            "analyze_document": True,
            "output_format": "json",
            "table_extraction_method": "camelot",
            "preserve_table_formatting": True
        })
    
    async def _process_technical_manual(self, pdf_path: str) -> Dict[str, Any]:
        """Process a technical manual with focus on structure and diagrams."""
        logger.info("Processing as technical manual")
        
        pipeline_tool = self._find_tool("pipeline")
        if not pipeline_tool:
            raise PDFProcessingError("Pipeline tool not available")
        
        return await pipeline_tool.execute({
            "file_path": pdf_path,
            "mode": "comprehensive",
            "extract_text": True,
            "extract_tables": True,
            "apply_ocr": True,  # May have diagrams that need OCR
            "extract_formulas": True,
            "analyze_document": True,
            "output_format": "json",
            "preserve_layout": True,
            "extract_images": True
        })
    
    async def _process_form(self, pdf_path: str) -> Dict[str, Any]:
        """Process a form with focus on OCR and field extraction."""
        logger.info("Processing as form")
        
        pipeline_tool = self._find_tool("pipeline")
        if not pipeline_tool:
            raise PDFProcessingError("Pipeline tool not available")
        
        return await pipeline_tool.execute({
            "file_path": pdf_path,
            "mode": "comprehensive",
            "extract_text": True,
            "extract_tables": False,
            "apply_ocr": True,
            "extract_formulas": False,
            "analyze_document": True,
            "output_format": "json",
            "ocr_language": "eng",
            "extract_form_fields": True
        })
    
    async def _process_general_document(self, pdf_path: str) -> Dict[str, Any]:
        """Process a general document with balanced extraction."""
        logger.info("Processing as general document")
        
        pipeline_tool = self._find_tool("pipeline")
        if not pipeline_tool:
            raise PDFProcessingError("Pipeline tool not available")
        
        return await pipeline_tool.execute({
            "file_path": pdf_path,
            "mode": "auto",
            "extract_text": True,
            "extract_tables": True,
            "apply_ocr": False,
            "extract_formulas": False,
            "analyze_document": True,
            "output_format": "json"
        })
    
    def _find_tool(self, tool_type: str):
        """Find a tool by type/name."""
        for tool in self.server.server.tools:
            if tool_type.lower() in tool.name.lower():
                return tool
        return None
    
    async def batch_process(self, pdf_paths: List[str], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Process multiple PDF files concurrently."""
        logger.info(f"Starting batch processing of {len(pdf_paths)} files (max concurrent: {max_concurrent})")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(pdf_path: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.process_document_by_type(pdf_path)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    return {"error": str(e), "file_path": pdf_path}
        
        # Process all files concurrently
        tasks = [process_single(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "file_path": pdf_paths[i]
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processing completed. {len([r for r in processed_results if 'error' not in r])} successful, {len([r for r in processed_results if 'error' in r])} failed")
        return processed_results
    
    async def compare_extraction_methods(self, pdf_path: str) -> Dict[str, Any]:
        """Compare different extraction methods on the same document."""
        logger.info(f"Comparing extraction methods for: {pdf_path}")
        
        results = {}
        
        # Text extraction comparison
        text_tool = self._find_tool("text")
        if text_tool:
            try:
                # Method 1: PyMuPDF
                results["text_pymupdf"] = await text_tool.execute({
                    "file_path": pdf_path,
                    "method": "pymupdf",
                    "preserve_layout": False
                })
                
                # Method 2: pdfplumber
                results["text_pdfplumber"] = await text_tool.execute({
                    "file_path": pdf_path,
                    "method": "pdfplumber",
                    "preserve_layout": True
                })
            except Exception as e:
                results["text_error"] = str(e)
        
        # Table extraction comparison (if tables are detected)
        table_tool = self._find_tool("table")
        if table_tool:
            try:
                # Method 1: Camelot
                results["tables_camelot"] = await table_tool.execute({
                    "file_path": pdf_path,
                    "method": "camelot",
                    "pages": "all"
                })
                
                # Method 2: Tabula
                results["tables_tabula"] = await table_tool.execute({
                    "file_path": pdf_path,
                    "method": "tabula",
                    "pages": "all"
                })
            except Exception as e:
                results["tables_error"] = str(e)
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        if self.server:
            await self.server.stop()
        logger.info("Advanced PDF processor cleaned up")


async def create_custom_config() -> str:
    """Create a custom configuration for advanced processing."""
    config = {
        "server": {
            "name": "PDF-MCP-Advanced",
            "version": "1.0.0",
            "description": "Advanced PDF processing server",
            "max_concurrent_requests": 5,
            "request_timeout": 300,
            "temp_directory": "./temp_advanced",
            "cleanup_interval": 3600
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "./logs/pdf_mcp_advanced.log"
        },
        "tools": {
            "text_extraction": {
                "enabled": True,
                "default_method": "pdfplumber",
                "preserve_layout": True,
                "extract_metadata": True
            },
            "table_extraction": {
                "enabled": True,
                "default_method": "camelot",
                "lattice_detection": True,
                "stream_detection": True
            },
            "ocr": {
                "enabled": True,
                "default_engine": "tesseract",
                "languages": ["eng", "chi_sim"],
                "dpi": 300
            },
            "formula_recognition": {
                "enabled": True,
                "default_model": "pix2tex",
                "confidence_threshold": 0.8
            },
            "analysis": {
                "enabled": True,
                "detect_document_type": True,
                "analyze_structure": True,
                "extract_metadata": True
            }
        },
        "performance": {
            "max_file_size_mb": 100,
            "max_pages": 1000,
            "enable_parallel_processing": True,
            "enable_caching": True,
            "memory_limit_mb": 2048
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name


async def main():
    """Main advanced example function."""
    logger.info("Starting PDF-MCP Server Advanced Usage Examples")
    
    # Create custom configuration
    config_path = await create_custom_config()
    logger.info(f"Created custom configuration: {config_path}")
    
    # Sample PDF paths - replace with actual PDF files
    sample_pdfs = [
        "/path/to/academic_paper.pdf",
        "/path/to/financial_report.pdf",
        "/path/to/technical_manual.pdf",
        "/path/to/form.pdf"
    ]
    
    processor = None
    
    try:
        # Initialize advanced processor
        processor = AdvancedPDFProcessor(config_path)
        await processor.initialize()
        
        # Example 1: Process documents by type
        logger.info("\n=== Example 1: Document Type-based Processing ===")
        for pdf_path in sample_pdfs[:1]:  # Process first one for demo
            if Path(pdf_path).exists():
                result = await processor.process_document_by_type(pdf_path)
                logger.info(f"Processed {pdf_path}: {type(result)} result")
            else:
                logger.info(f"Sample file not found: {pdf_path}")
        
        # Example 2: Batch processing
        logger.info("\n=== Example 2: Batch Processing ===")
        existing_pdfs = [pdf for pdf in sample_pdfs if Path(pdf).exists()]
        if existing_pdfs:
            batch_results = await processor.batch_process(existing_pdfs, max_concurrent=2)
            logger.info(f"Batch processing completed: {len(batch_results)} results")
        else:
            logger.info("No existing sample files for batch processing")
        
        # Example 3: Method comparison
        logger.info("\n=== Example 3: Extraction Method Comparison ===")
        if existing_pdfs:
            comparison = await processor.compare_extraction_methods(existing_pdfs[0])
            logger.info(f"Method comparison completed: {list(comparison.keys())}")
        else:
            logger.info("No sample files for method comparison")
        
        logger.info("\nAll advanced examples completed successfully")
        
    except Exception as e:
        logger.error(f"Advanced example execution failed: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Cleanup
        if processor:
            await processor.cleanup()
        
        # Remove temporary config file
        try:
            Path(config_path).unlink()
            logger.info("Temporary configuration file removed")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())