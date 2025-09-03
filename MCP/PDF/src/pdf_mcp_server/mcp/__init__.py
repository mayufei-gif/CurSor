#!/usr/bin/env python3
"""
PDF-MCP Server MCP Protocol Implementation

This module provides the Model Context Protocol (MCP) implementation for the PDF processing server.
It defines the MCP tools, handles tool calls, and manages the communication protocol.

Author: PDF-MCP Team
License: MIT
"""

from .server import MCPServer
from .tools import (
    PDFTool,
    ExtractTextTool,
    ExtractTextAdvancedTool,
    ExtractTablesTool,
    ExtractTextOCRTool,
    ExtractTextOCRAdvancedTool,
    ExtractFormulasTool,
    ExtractFormulasAdvancedTool,
    AnalyzeDocumentTool,
    SmartRoutingTool,
    get_tool,
    list_tools
)
from .protocol import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPToolCall,
    MCPToolResult,
    MCPMessage,
    MCPProtocolHandler
)
from .exceptions import (
    MCPException,
    ToolNotFoundException,
    InvalidToolCallException,
    MCPProtocolException
)

__all__ = [
    # Server
    'MCPServer',
    
    # Tools
    'PDFTool',
    'ReadTextTool',
    'ExtractTablesTool', 
    'ExtractFormulasTool',
    'ProcessOCRTool',
    'FullPipelineTool',
    'get_available_tools',
    'create_tool_registry',
    
    # Protocol
    'MCPRequest',
    'MCPResponse',
    'MCPError',
    'ToolCall',
    'ToolResult',
    'MCPMessage',
    'MCPProtocolHandler',
    
    # Exceptions
    'MCPException',
    'ToolNotFoundException',
    'InvalidToolCallException',
    'MCPProtocolException'
]

# Version information
__version__ = '1.0.0'
__author__ = 'PDF-MCP Team'
__email__ = 'team@pdf-mcp.com'
__license__ = 'MIT'
__description__ = 'MCP Protocol Implementation for PDF Processing Server'