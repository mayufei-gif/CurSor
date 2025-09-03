#!/usr/bin/env python3
"""
PDF-MCP Server Basic Usage Example

This script demonstrates how to use the PDF-MCP server to process PDF files
using the Model Context Protocol (MCP).

Author: PDF-MCP Team
License: MIT
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_mcp_server.main import PDFMCPServer
from src.pdf_mcp_server.mcp.protocol import MCPProtocolHandler, MCPRequest, MCPMethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pdf_mcp_example')


async def example_text_extraction(server: PDFMCPServer, pdf_path: str):
    """Example: Extract text from a PDF file."""
    logger.info("Example: Text Extraction")
    
    # Find the text extraction tool
    text_tool = None
    for tool in server.server.tools:
        if "text" in tool.name.lower() and "read" in tool.name.lower():
            text_tool = tool
            break
    
    if not text_tool:
        logger.error("Text extraction tool not found")
        return
    
    # Prepare arguments
    args = {
        "file_path": pdf_path,
        "pages": [1, 2],  # Extract first two pages
        "include_metadata": True,
        "preserve_layout": True
    }
    
    try:
        # Execute the tool
        result = await text_tool.execute(args)
        
        # Display results
        logger.info("Text extraction completed successfully")
        logger.info(f"Extracted text preview: {str(result)[:200]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return None


async def example_pdf_analysis(server: PDFMCPServer, pdf_path: str):
    """Example: Analyze PDF structure and content."""
    logger.info("Example: PDF Analysis")
    
    # Find the analysis tool
    analysis_tool = None
    for tool in server.server.tools:
        if "analyze" in tool.name.lower():
            analysis_tool = tool
            break
    
    if not analysis_tool:
        logger.error("PDF analysis tool not found")
        return
    
    # Prepare arguments
    args = {
        "file_path": pdf_path,
        "analyze_structure": True,
        "detect_content_types": True,
        "generate_recommendations": True
    }
    
    try:
        # Execute the tool
        result = await analysis_tool.execute(args)
        
        # Display results
        logger.info("PDF analysis completed successfully")
        if isinstance(result, dict):
            logger.info(f"Document type: {result.get('document_type', 'Unknown')}")
            logger.info(f"Total pages: {result.get('page_count', 'Unknown')}")
            logger.info(f"Has text: {result.get('has_text', 'Unknown')}")
            logger.info(f"Has images: {result.get('has_images', 'Unknown')}")
            logger.info(f"Has tables: {result.get('has_tables', 'Unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"PDF analysis failed: {e}")
        return None


async def example_full_pipeline(server: PDFMCPServer, pdf_path: str):
    """Example: Run full processing pipeline."""
    logger.info("Example: Full Processing Pipeline")
    
    # Find the full pipeline tool
    pipeline_tool = None
    for tool in server.server.tools:
        if "pipeline" in tool.name.lower() or "full" in tool.name.lower():
            pipeline_tool = tool
            break
    
    if not pipeline_tool:
        logger.error("Full pipeline tool not found")
        return
    
    # Prepare arguments
    args = {
        "file_path": pdf_path,
        "mode": "comprehensive",
        "extract_text": True,
        "extract_tables": False,  # Disable for this example
        "apply_ocr": False,      # Disable for this example
        "extract_formulas": False,  # Disable for this example
        "analyze_document": True,
        "output_format": "json"
    }
    
    try:
        # Execute the tool
        result = await pipeline_tool.execute(args)
        
        # Display results
        logger.info("Full pipeline processing completed successfully")
        logger.info(f"Processing result keys: {list(result.keys()) if isinstance(result, dict) else 'Non-dict result'}")
        
        return result
        
    except Exception as e:
        logger.error(f"Full pipeline processing failed: {e}")
        return None


async def demonstrate_mcp_protocol(server: PDFMCPServer):
    """Demonstrate MCP protocol usage."""
    logger.info("Example: MCP Protocol Communication")
    
    # Create protocol handler
    protocol = MCPProtocolHandler()
    
    # Example 1: List available tools
    list_tools_request = MCPRequest(
        id="list-tools-1",
        method=MCPMethod.TOOLS_LIST,
        params={}
    )
    
    # Serialize the request (this would be sent over the wire)
    serialized_request = protocol.serialize_message(list_tools_request)
    logger.info(f"Serialized list tools request: {serialized_request[:100]}...")
    
    # Parse the request (this would be done by the server)
    parsed_request = protocol.parse_message(serialized_request)
    logger.info(f"Parsed request method: {parsed_request.method}")
    
    # Get tools list
    tools = server.server.list_tools()
    logger.info(f"Available tools: {[tool['name'] for tool in tools]}")
    
    # Example 2: Create a tool call request
    if tools:
        first_tool = tools[0]
        tool_call_request = MCPRequest(
            id="tool-call-1",
            method=MCPMethod.TOOLS_CALL,
            params={
                "name": first_tool["name"],
                "arguments": {
                    "file_path": "/path/to/sample.pdf"
                }
            }
        )
        
        serialized_call = protocol.serialize_message(tool_call_request)
        logger.info(f"Serialized tool call request: {serialized_call[:100]}...")


async def main():
    """Main example function."""
    logger.info("Starting PDF-MCP Server Basic Usage Examples")
    
    # Sample PDF path - replace with an actual PDF file
    sample_pdf = Path(__file__).parent / "samples" / "sample.pdf"
    
    if not sample_pdf.exists():
        logger.warning(f"Sample PDF not found at {sample_pdf}")
        logger.info("Creating a placeholder path for demonstration")
        sample_pdf = "/path/to/your/sample.pdf"
    
    try:
        # Initialize the PDF-MCP server
        logger.info("Initializing PDF-MCP server...")
        server = PDFMCPServer()
        await server.initialize()
        
        logger.info(f"Server initialized with {len(server.server.tools)} tools")
        
        # Demonstrate MCP protocol
        await demonstrate_mcp_protocol(server)
        
        # Only run file processing examples if sample PDF exists
        if isinstance(sample_pdf, Path) and sample_pdf.exists():
            # Example 1: Text extraction
            await example_text_extraction(server, str(sample_pdf))
            
            # Example 2: PDF analysis
            await example_pdf_analysis(server, str(sample_pdf))
            
            # Example 3: Full pipeline
            await example_full_pipeline(server, str(sample_pdf))
        else:
            logger.info("Skipping file processing examples (no sample PDF available)")
        
        logger.info("All examples completed successfully")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())