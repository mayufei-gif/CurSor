#!/usr/bin/env python3
"""
MCP PDF Tools

Base classes and utilities for PDF processing tools in the MCP framework.
Provides a unified interface for all PDF operations.

Author: PDF-MCP Team
License: MIT
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
from datetime import datetime

from .protocol import (
    MCPTool,
    MCPToolResult,
    MCPToolInputSchema,
    create_text_content,
    create_image_content,
    create_resource_content,
    create_progress_content,
    create_error_content
)
from .exceptions import (
    ToolExecutionException,
    MCPValidationException,
    MCPResourceException
)


class PDFToolBase(ABC):
    """Base class for all PDF processing tools."""
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.logger = logging.getLogger(f"pdf_tools.{name}")
        
        # Tool metadata
        self.metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "category": self.get_category(),
            "tags": self.get_tags()
        }
        
        # Execution statistics
        self.stats = {
            "calls_total": 0,
            "calls_success": 0,
            "calls_failed": 0,
            "total_execution_time": 0.0,
            "last_called": None
        }
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool input parameters."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def get_category(self) -> str:
        """Get the tool category."""
        return "pdf"
    
    def get_tags(self) -> List[str]:
        """Get tool tags for categorization."""
        return []
    
    def get_tool_definition(self) -> MCPTool:
        """Get the MCP tool definition."""
        schema = MCPToolInputSchema(**self.get_input_schema())
        return MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=schema
        )
    
    async def __call__(self, **kwargs) -> MCPToolResult:
        """Execute the tool and track statistics."""
        start_time = time.time()
        self.stats["calls_total"] += 1
        self.stats["last_called"] = datetime.now().isoformat()
        
        try:
            # Validate input parameters
            self._validate_input(kwargs)
            
            # Execute the tool
            result = await self.execute(**kwargs)
            
            # Update success statistics
            self.stats["calls_success"] += 1
            execution_time = time.time() - start_time
            self.stats["total_execution_time"] += execution_time
            
            self.logger.info(f"Tool {self.name} executed successfully in {execution_time:.2f}s")
            return result
        
        except Exception as e:
            # Update failure statistics
            self.stats["calls_failed"] += 1
            execution_time = time.time() - start_time
            self.stats["total_execution_time"] += execution_time
            
            self.logger.error(f"Tool {self.name} failed after {execution_time:.2f}s: {e}")
            
            # Return error result
            content = [create_error_content(str(e), type(e).__name__)]
            return MCPToolResult(content=content, isError=True)
    
    def _validate_input(self, kwargs: Dict[str, Any]):
        """Validate input parameters against schema."""
        schema = self.get_input_schema()
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required fields
        for field in required_fields:
            if field not in kwargs:
                raise MCPValidationException(f"Missing required parameter: {field}")
        
        # Validate field types (basic validation)
        for field, value in kwargs.items():
            if field in properties:
                field_schema = properties[field]
                field_type = field_schema.get("type")
                
                if field_type == "string" and not isinstance(value, str):
                    raise MCPValidationException(f"Parameter {field} must be a string")
                elif field_type == "integer" and not isinstance(value, int):
                    raise MCPValidationException(f"Parameter {field} must be an integer")
                elif field_type == "number" and not isinstance(value, (int, float)):
                    raise MCPValidationException(f"Parameter {field} must be a number")
                elif field_type == "boolean" and not isinstance(value, bool):
                    raise MCPValidationException(f"Parameter {field} must be a boolean")
                elif field_type == "array" and not isinstance(value, list):
                    raise MCPValidationException(f"Parameter {field} must be an array")
                elif field_type == "object" and not isinstance(value, dict):
                    raise MCPValidationException(f"Parameter {field} must be an object")
    
    def _validate_file_path(self, file_path: str, must_exist: bool = True) -> Path:
        """Validate and normalize file path."""
        try:
            path = Path(file_path).resolve()
            
            if must_exist and not path.exists():
                raise MCPResourceException(f"File not found: {file_path}")
            
            if must_exist and not path.is_file():
                raise MCPResourceException(f"Path is not a file: {file_path}")
            
            return path
        
        except Exception as e:
            if isinstance(e, MCPResourceException):
                raise
            raise MCPValidationException(f"Invalid file path: {file_path}")
    
    def _validate_pdf_file(self, file_path: str) -> Path:
        """Validate that the file is a PDF."""
        path = self._validate_file_path(file_path)
        
        if path.suffix.lower() != '.pdf':
            raise MCPValidationException(f"File is not a PDF: {file_path}")
        
        return path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        avg_time = 0.0
        if self.stats["calls_success"] > 0:
            avg_time = self.stats["total_execution_time"] / self.stats["calls_success"]
        
        return {
            **self.stats,
            "average_execution_time": avg_time,
            "success_rate": (
                self.stats["calls_success"] / self.stats["calls_total"]
                if self.stats["calls_total"] > 0 else 0.0
            )
        }


class PDFTextTool(PDFToolBase):
    """Base class for PDF text processing tools."""
    
    def get_category(self) -> str:
        return "pdf_text"
    
    def get_tags(self) -> List[str]:
        return ["text", "extraction", "pdf"]


class PDFTableTool(PDFToolBase):
    """Base class for PDF table processing tools."""
    
    def get_category(self) -> str:
        return "pdf_table"
    
    def get_tags(self) -> List[str]:
        return ["table", "extraction", "pdf", "data"]


class PDFOCRTool(PDFToolBase):
    """Base class for PDF OCR tools."""
    
    def get_category(self) -> str:
        return "pdf_ocr"
    
    def get_tags(self) -> List[str]:
        return ["ocr", "text", "recognition", "pdf"]


class PDFFormulaTool(PDFToolBase):
    """Base class for PDF formula processing tools."""
    
    def get_category(self) -> str:
        return "pdf_formula"
    
    def get_tags(self) -> List[str]:
        return ["formula", "math", "latex", "pdf"]


class PDFAnalysisTool(PDFToolBase):
    """Base class for PDF analysis tools."""
    
    def get_category(self) -> str:
        return "pdf_analysis"
    
    def get_tags(self) -> List[str]:
        return ["analysis", "metadata", "pdf"]


class ToolRegistry:
    """Registry for managing PDF tools."""
    
    def __init__(self):
        self.tools: Dict[str, PDFToolBase] = {}
        self.categories: Dict[str, List[str]] = {}
        self.tags: Dict[str, List[str]] = {}
        self.logger = logging.getLogger("tool_registry")
    
    def register(self, tool: PDFToolBase):
        """Register a tool."""
        if tool.name in self.tools:
            self.logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self.tools[tool.name] = tool
        
        # Update categories
        category = tool.get_category()
        if category not in self.categories:
            self.categories[category] = []
        if tool.name not in self.categories[category]:
            self.categories[category].append(tool.name)
        
        # Update tags
        for tag in tool.get_tags():
            if tag not in self.tags:
                self.tags[tag] = []
            if tool.name not in self.tags[tag]:
                self.tags[tag].append(tool.name)
        
        self.logger.info(f"Registered tool: {tool.name} (category: {category})")
    
    def unregister(self, name: str):
        """Unregister a tool."""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        
        tool = self.tools[name]
        
        # Remove from categories
        category = tool.get_category()
        if category in self.categories and name in self.categories[category]:
            self.categories[category].remove(name)
            if not self.categories[category]:
                del self.categories[category]
        
        # Remove from tags
        for tag in tool.get_tags():
            if tag in self.tags and name in self.tags[tag]:
                self.tags[tag].remove(name)
                if not self.tags[tag]:
                    del self.tags[tag]
        
        del self.tools[name]
        self.logger.info(f"Unregistered tool: {name}")
    
    def get(self, name: str) -> Optional[PDFToolBase]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
        """List tools, optionally filtered by category or tag."""
        if category:
            return self.categories.get(category, [])
        elif tag:
            return self.tags.get(tag, [])
        else:
            return list(self.tools.keys())
    
    def get_tool_definitions(self) -> List[MCPTool]:
        """Get MCP tool definitions for all registered tools."""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self.categories.keys())
    
    def get_tags(self) -> List[str]:
        """Get all available tags."""
        return list(self.tags.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_calls = sum(tool.stats["calls_total"] for tool in self.tools.values())
        total_success = sum(tool.stats["calls_success"] for tool in self.tools.values())
        total_failed = sum(tool.stats["calls_failed"] for tool in self.tools.values())
        
        return {
            "tools_registered": len(self.tools),
            "categories": len(self.categories),
            "tags": len(self.tags),
            "total_calls": total_calls,
            "total_success": total_success,
            "total_failed": total_failed,
            "overall_success_rate": total_success / total_calls if total_calls > 0 else 0.0
        }


# Global tool registry instance
_global_registry = ToolRegistry()


def register_tool(tool: PDFToolBase):
    """Register a tool with the global registry."""
    _global_registry.register(tool)


def unregister_tool(name: str):
    """Unregister a tool from the global registry."""
    _global_registry.unregister(name)


def get_tool(name: str) -> Optional[PDFToolBase]:
    """Get a tool from the global registry."""
    return _global_registry.get(name)


def list_tools(category: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
    """List tools from the global registry."""
    return _global_registry.list_tools(category, tag)


def get_tool_definitions() -> List[MCPTool]:
    """Get all tool definitions from the global registry."""
    return _global_registry.get_tool_definitions()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def register_tools_with_server(server, registry: Optional[ToolRegistry] = None):
    """Register all tools from registry with an MCP server."""
    if registry is None:
        registry = _global_registry
    
    for tool in registry.tools.values():
        server.register_tool(
            name=tool.name,
            description=tool.description,
            input_schema=tool.get_input_schema(),
            handler=tool,
            **tool.metadata
        )


# Utility functions for creating common tool results

def create_text_result(text: str, metadata: Optional[Dict[str, Any]] = None) -> MCPToolResult:
    """Create a text result."""
    content = [create_text_content(text)]
    if metadata:
        content.append(create_text_content(f"Metadata: {json.dumps(metadata, indent=2)}"))
    return MCPToolResult(content=content)


def create_json_result(data: Any, description: str = "Result") -> MCPToolResult:
    """Create a JSON result."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    content = [
        create_text_content(f"{description}:\n```json\n{json_str}\n```")
    ]
    return MCPToolResult(content=content)


def create_table_result(data: List[List[Any]], headers: Optional[List[str]] = None) -> MCPToolResult:
    """Create a table result."""
    # Convert to markdown table
    if headers:
        table_lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
        table_lines.append("|" + "---|" * len(headers))
    else:
        table_lines = []
    
    for row in data:
        table_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    table_str = "\n".join(table_lines)
    content = [create_text_content(f"Table:\n{table_str}")]
    return MCPToolResult(content=content)


def create_progress_result(current: float, total: float, message: str = "") -> MCPToolResult:
    """Create a progress result."""
    content = [create_progress_content(current, total)]
    if message:
        content.append(create_text_content(message))
    return MCPToolResult(content=content)


def create_error_result(error: str, details: Optional[Dict[str, Any]] = None) -> MCPToolResult:
    """Create an error result."""
    content = [create_error_content(error)]
    if details:
        content.append(create_text_content(f"Details: {json.dumps(details, indent=2)}"))
    return MCPToolResult(content=content, isError=True)