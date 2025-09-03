#!/usr/bin/env python3
"""
PDF-MCP Server Test Script

This script tests the PDF-MCP server functionality by sending various MCP requests
and validating the responses.

Author: PDF-MCP Team
License: MIT
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_mcp_server.mcp.protocol import MCPProtocolHandler, MCPRequest, MCPMethod
from src.pdf_mcp_server.main import PDFMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pdf_mcp_test')

# Sample PDF path - replace with an actual PDF file for testing
SAMPLE_PDF_PATH = Path(__file__).parent / 'samples' / 'sample.pdf'


@pytest.mark.asyncio
async def test_server_initialization():
    """Test server initialization."""
    logger.info("Testing server initialization...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        config = {
            "server": {
                "name": "pdf-mcp-test-server",
                "version": "1.0.0",
                "description": "Test PDF processing server",
                "max_concurrent_requests": 5,
                "request_timeout": 60,
                "temp_dir": "./temp_test",
                "cleanup_interval": 3600
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None
            },
            "tools": {
                "text_extraction": {"enabled": True},
                "table_extraction": {"enabled": False},
                "ocr": {"enabled": False},
                "formula_recognition": {"enabled": False},
                "analysis": {"enabled": True}
            }
        }
        json.dump(config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Initialize server
        server = PDFMCPServer(config_file=temp_file_path)
        await server.initialize()
        
        # Check if server was initialized correctly
        assert server.server is not None, "Server was not initialized"
        assert server.protocol_handler is not None, "Protocol handler was not initialized"
        
        # Check if tools were registered correctly
        tool_names = server.server.list_tool_names()
        logger.info(f"Registered tools: {tool_names}")
        
        # Verify text extraction tools are registered
        assert any("text" in tool.lower() for tool in tool_names), "Text extraction tools not registered"
        
        # Verify analysis tools are registered
        assert any("analyze" in tool.lower() for tool in tool_names), "Analysis tools not registered"
        
        # Verify table extraction tools are not registered (disabled in config)
        assert not any("table" in tool.lower() for tool in tool_names), "Table extraction tools should not be registered"
        
        logger.info("Server initialization test passed")
        
    except Exception as e:
        logger.error(f"Server initialization test failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up temporary config file
        os.unlink(temp_file_path)


@pytest.mark.asyncio
async def test_tool_registration():
    """Test tool registration."""
    logger.info("Testing tool registration...")
    
    # Create server with default config
    server = PDFMCPServer()
    await server.initialize()
    
    # Get registered tools
    tools = server.server.list_tools()
    
    # Verify tools were registered
    assert len(tools) > 0, "No tools were registered"
    
    # Check tool structure
    for tool in tools:
        assert "name" in tool, "Tool missing 'name' field"
        assert "description" in tool, "Tool missing 'description' field"
        assert "inputSchema" in tool, "Tool missing 'inputSchema' field"
    
    logger.info(f"Found {len(tools)} registered tools")
    logger.info("Tool registration test passed")


@pytest.mark.asyncio
async def test_protocol_handler():
    """Test protocol handler functionality."""
    logger.info("Testing protocol handler...")
    
    # Create protocol handler
    protocol = MCPProtocolHandler()
    
    # Create a sample request
    request = MCPRequest(
        id="test-request-1",
        method=MCPMethod.INITIALIZE,
        params={"capabilities": ["tools", "resources"]}
    )
    
    # Serialize and deserialize the request
    serialized = protocol.serialize_message(request)
    deserialized = protocol.parse_message(serialized)
    
    # Verify the deserialized request matches the original
    assert deserialized.id == request.id, "Request ID mismatch"
    assert deserialized.method == request.method, "Request method mismatch"
    assert deserialized.params == request.params, "Request params mismatch"
    
    logger.info("Protocol handler test passed")


@pytest.mark.asyncio
async def test_text_extraction():
    """Test text extraction functionality."""
    logger.info("Testing text extraction...")
    
    # Skip test if sample PDF doesn't exist
    if not SAMPLE_PDF_PATH.exists():
        logger.warning(f"Sample PDF not found at {SAMPLE_PDF_PATH}, skipping text extraction test")
        return
    
    # Create server
    server = PDFMCPServer()
    await server.initialize()
    
    # Create protocol handler
    protocol = MCPProtocolHandler()
    
    # Create a tool call request for text extraction
    request = MCPRequest(
        id="test-text-extraction",
        method=MCPMethod.TOOLS_CALL,
        params={
            "name": "read_text",
            "arguments": {
                "file_path": str(SAMPLE_PDF_PATH),
                "pages": [1],
                "include_metadata": True
            }
        }
    )
    
    # Serialize the request
    serialized_request = protocol.serialize_message(request)
    
    # Process the request (in a real scenario, this would be sent to the server)
    # For testing, we'll directly call the tool handler
    tool_name = request.params["name"]
    tool_args = request.params["arguments"]
    
    # Get the tool
    tool = next((t for t in server.server.tools if t.name == tool_name), None)
    assert tool is not None, f"Tool '{tool_name}' not found"
    
    # Call the tool
    result = await tool.execute(tool_args)
    
    # Verify the result
    assert result is not None, "Tool execution returned None"
    assert "text" in result or "content" in result, "Text extraction result missing text content"
    
    logger.info("Text extraction test passed")


async def run_tests():
    """Run all tests."""
    logger.info("Starting PDF-MCP server tests")
    
    try:
        await test_server_initialization()
        await test_tool_registration()
        await test_protocol_handler()
        await test_text_extraction()
        
        logger.info("All tests passed successfully")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()