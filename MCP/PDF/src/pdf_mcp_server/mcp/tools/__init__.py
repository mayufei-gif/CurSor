#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF MCP Server Tools Module

This module provides all the PDF processing tools for the MCP server.
"""

from .text_extraction import ExtractTextTool, ExtractTextAdvancedTool
from .table_extraction import ExtractTablesTool
from .ocr_extraction import ExtractTextOCRTool, ExtractTextOCRAdvancedTool
from .formula_extraction import ExtractFormulasTool, ExtractFormulasAdvancedTool
from .document_analysis import AnalyzeDocumentTool, SmartRoutingTool
from .base import PDFTool

__all__ = [
    'PDFTool',
    'ExtractTextTool',
    'ExtractTextAdvancedTool',
    'ExtractTablesTool',
    'ExtractTextOCRTool',
    'ExtractTextOCRAdvancedTool',
    'ExtractFormulasTool',
    'ExtractFormulasAdvancedTool',
    'AnalyzeDocumentTool',
    'SmartRoutingTool',
]

# Tool registry for easy access
TOOL_REGISTRY = {
    'extract_text': ExtractTextTool,
    'extract_text_advanced': ExtractTextAdvancedTool,
    'extract_tables': ExtractTablesTool,
    'extract_text_ocr': ExtractTextOCRTool,
    'extract_text_ocr_advanced': ExtractTextOCRAdvancedTool,
    'extract_formulas': ExtractFormulasTool,
    'extract_formulas_advanced': ExtractFormulasAdvancedTool,
    'analyze_document': AnalyzeDocumentTool,
    'smart_routing': SmartRoutingTool,
}

def get_tool(tool_name: str) -> type:
    """Get a tool class by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool class
        
    Raises:
        KeyError: If tool not found
    """
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{tool_name}' not found. Available tools: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[tool_name]

def list_tools() -> list:
    """List all available tools.
    
    Returns:
        List of tool names
    """
    return list(TOOL_REGISTRY.keys())

def register_tools_to_server(server):
    """Register all tools to the MCP server.
    
    Args:
        server: MCP server instance
    """
    for tool_name, tool_class in TOOL_REGISTRY.items():
        # Create tool instance
        tool_instance = tool_class()
        
        # Get input schema from tool
        input_schema = tool_instance.get_input_schema() if hasattr(tool_instance, 'get_input_schema') else {}
        
        # Register tool with server
        server.register_tool(
            name=tool_name,
            description=tool_instance.description,
            input_schema=input_schema,
            handler=tool_instance.execute
        )