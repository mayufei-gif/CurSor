#!/usr/bin/env python3
"""
PDF Processing Tools

Implementations of various PDF processing tools for the MCP server.
Includes text extraction, table extraction, OCR, formula recognition, and more.

Author: PDF-MCP Team
License: MIT
"""

from .text_extraction import ReadTextTool, ExtractMetadataTool
from .table_extraction import ExtractTablesTool, ExtractTablesAdvancedTool
from .ocr_processing import ProcessOCRTool, OCRWithLayoutTool
from .formula_recognition import ExtractFormulasTool, FormulaToLatexTool
from .pdf_analysis import AnalyzePDFTool, DetectPDFTypeTool
from .full_pipeline import FullPipelineTool, SmartProcessingTool

# Tool classes for easy import
__all__ = [
    # Text extraction tools
    "ReadTextTool",
    "ExtractMetadataTool",
    
    # Table extraction tools
    "ExtractTablesTool",
    "ExtractTablesAdvancedTool",
    
    # OCR tools
    "ProcessOCRTool",
    "OCRWithLayoutTool",
    
    # Formula recognition tools
    "ExtractFormulasTool",
    "FormulaToLatexTool",
    
    # Analysis tools
    "AnalyzePDFTool",
    "DetectPDFTypeTool",
    
    # Pipeline tools
    "FullPipelineTool",
    "SmartProcessingTool",
    
    # Utility functions
    "register_all_tools",
    "get_available_tools",
    "create_tool_instances"
]

# Tool metadata
TOOL_CATEGORIES = {
    "text": ["ReadTextTool", "ExtractMetadataTool"],
    "table": ["ExtractTablesTool", "ExtractTablesAdvancedTool"],
    "ocr": ["ProcessOCRTool", "OCRWithLayoutTool"],
    "formula": ["ExtractFormulasTool", "FormulaToLatexTool"],
    "analysis": ["AnalyzePDFTool", "DetectPDFTypeTool"],
    "pipeline": ["FullPipelineTool", "SmartProcessingTool"]
}

TOOL_DESCRIPTIONS = {
    "ReadTextTool": "Extract text content from PDF files with various options",
    "ExtractMetadataTool": "Extract metadata and document information from PDF files",
    "ExtractTablesTool": "Extract tables from PDF files using multiple detection methods",
    "ExtractTablesAdvancedTool": "Advanced table extraction with custom formatting and filtering",
    "ProcessOCRTool": "Perform OCR on PDF files to extract text from images",
    "OCRWithLayoutTool": "OCR with layout preservation and text positioning",
    "ExtractFormulasTool": "Extract mathematical formulas and equations from PDF files",
    "FormulaToLatexTool": "Convert extracted formulas to LaTeX format",
    "AnalyzePDFTool": "Analyze PDF structure, content types, and characteristics",
    "DetectPDFTypeTool": "Detect the type and category of PDF document",
    "FullPipelineTool": "Complete PDF processing pipeline with all features",
    "SmartProcessingTool": "Intelligent PDF processing with automatic feature detection"
}


def register_all_tools(registry=None):
    """Register all available PDF tools with the registry."""
    from ..mcp.tools import register_tool, get_registry
    
    if registry is None:
        registry = get_registry()
    
    # Create and register all tool instances
    tools = create_tool_instances()
    
    for tool in tools:
        registry.register(tool)
    
    return len(tools)


def create_tool_instances():
    """Create instances of all available tools."""
    tools = []
    
    # Text extraction tools
    tools.append(ReadTextTool())
    tools.append(ExtractMetadataTool())
    
    # Table extraction tools
    tools.append(ExtractTablesTool())
    tools.append(ExtractTablesAdvancedTool())
    
    # OCR tools
    tools.append(ProcessOCRTool())
    tools.append(OCRWithLayoutTool())
    
    # Formula recognition tools
    tools.append(ExtractFormulasTool())
    tools.append(FormulaToLatexTool())
    
    # Analysis tools
    tools.append(AnalyzePDFTool())
    tools.append(DetectPDFTypeTool())
    
    # Pipeline tools
    tools.append(FullPipelineTool())
    tools.append(SmartProcessingTool())
    
    return tools


def get_available_tools():
    """Get information about all available tools."""
    return {
        "categories": TOOL_CATEGORIES,
        "descriptions": TOOL_DESCRIPTIONS,
        "total_tools": len(TOOL_DESCRIPTIONS)
    }


def get_tools_by_category(category: str):
    """Get tools in a specific category."""
    return TOOL_CATEGORIES.get(category, [])


def get_tool_info(tool_name: str):
    """Get information about a specific tool."""
    if tool_name not in TOOL_DESCRIPTIONS:
        return None
    
    # Find category
    category = None
    for cat, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            category = cat
            break
    
    return {
        "name": tool_name,
        "description": TOOL_DESCRIPTIONS[tool_name],
        "category": category
    }