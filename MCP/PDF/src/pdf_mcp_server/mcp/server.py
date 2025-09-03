#!/usr/bin/env python3
"""
MCP Server Implementation

Core MCP server that handles protocol communication, tool registration,
and request routing for PDF processing operations.

Author: PDF-MCP Team
License: MIT
"""

import asyncio
import json
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Callable, Union, Type
from datetime import datetime
from pathlib import Path

from .protocol import (
    MCPProtocolHandler,
    MCPRequest,
    MCPResponse,
    MCPNotification,
    MCPTool,
    MCPToolCall,
    MCPToolResult,
    MCPInitializeParams,
    MCPInitializeResult,
    MCPServerInfo,
    MCPCapabilities,
    MCPError,
    MCPErrorCode,
    MCPMethod,
    MCPLogLevel,
    create_text_content,
    create_error_content
)
from .exceptions import (
    MCPException,
    MCPProtocolException,
    ToolNotFoundException,
    InvalidToolCallException,
    ToolExecutionException,
    MCPTimeoutException,
    MCPAuthenticationException,
    MCPAuthorizationException,
    MCPRateLimitException,
    MCPValidationException
)


class MCPServer:
    """MCP Server implementation for PDF processing."""
    
    def __init__(
        self,
        name: str = "pdf-mcp-server",
        version: str = "1.0.0",
        protocol_version: str = "2024-11-05",
        description: Optional[str] = None,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        request_timeout: float = 300.0,  # 5 minutes
        enable_logging: bool = True
    ):
        self.name = name
        self.version = version
        self.description = description or "PDF processing MCP server"
        self.max_request_size = max_request_size
        self.request_timeout = request_timeout
        self.enable_logging = enable_logging
        
        # Protocol handler
        self.protocol = MCPProtocolHandler(protocol_version)
        
        # Server state
        self.initialized = False
        self.client_info = None
        self.client_capabilities = None
        
        # Tool registry
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Resource registry
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.resource_handlers: Dict[str, Callable] = {}
        
        # Prompt registry
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.prompt_handlers: Dict[str, Callable] = {}
        
        # Logging
        self.logger = self._setup_logging()
        self.log_level = MCPLogLevel.INFO
        
        # Statistics
        self.stats = {
            "requests_received": 0,
            "requests_processed": 0,
            "requests_failed": 0,
            "tools_called": 0,
            "start_time": datetime.now()
        }
        
        # Register built-in handlers
        self._register_builtin_handlers()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the server."""
        logger = logging.getLogger(f"mcp.{self.name}")
        
        if not logger.handlers and self.enable_logging:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _register_builtin_handlers(self):
        """Register built-in MCP handlers."""
        # Core protocol handlers
        self.register_method_handler(MCPMethod.INITIALIZE, self._handle_initialize)
        self.register_method_handler(MCPMethod.PING, self._handle_ping)
        
        # Tool handlers
        self.register_method_handler(MCPMethod.LIST_TOOLS, self._handle_list_tools)
        self.register_method_handler(MCPMethod.CALL_TOOL, self._handle_call_tool)
        
        # Resource handlers
        self.register_method_handler(MCPMethod.LIST_RESOURCES, self._handle_list_resources)
        self.register_method_handler(MCPMethod.READ_RESOURCE, self._handle_read_resource)
        
        # Prompt handlers
        self.register_method_handler(MCPMethod.LIST_PROMPTS, self._handle_list_prompts)
        self.register_method_handler(MCPMethod.GET_PROMPT, self._handle_get_prompt)
        
        # Logging handlers
        self.register_method_handler(MCPMethod.SET_LOG_LEVEL, self._handle_set_log_level)
    
    def register_method_handler(self, method: str, handler: Callable):
        """Register a method handler."""
        if not hasattr(self, '_method_handlers'):
            self._method_handlers = {}
        self._method_handlers[method] = handler
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
        **metadata
    ):
        """Register a tool with the server."""
        tool_def = self.protocol.create_tool_definition(name, description, input_schema)
        
        self.tools[name] = {
            "definition": tool_def,
            "metadata": metadata
        }
        self.tool_handlers[name] = handler
        
        self.logger.info(f"Registered tool: {name}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [tool_info["definition"].dict() for tool_info in self.tools.values()]
    
    def list_tool_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str,
        handler: Callable,
        **metadata
    ):
        """Register a resource with the server."""
        self.resources[uri] = {
            "name": name,
            "description": description,
            "mimeType": mime_type,
            "metadata": metadata
        }
        self.resource_handlers[uri] = handler
        
        self.logger.info(f"Registered resource: {uri}")
    
    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: List[Dict[str, Any]],
        handler: Callable,
        **metadata
    ):
        """Register a prompt with the server."""
        self.prompts[name] = {
            "description": description,
            "arguments": arguments,
            "metadata": metadata
        }
        self.prompt_handlers[name] = handler
        
        self.logger.info(f"Registered prompt: {name}")
    
    async def handle_message(self, data: Union[str, bytes, Dict[str, Any]]) -> Optional[str]:
        """Handle an incoming MCP message."""
        self.stats["requests_received"] += 1
        
        try:
            # Parse message
            message = self.protocol.parse_message(data)
            
            # Handle different message types
            if isinstance(message, MCPRequest):
                response = await self._handle_request(message)
                if response:
                    return self.protocol.serialize_message(response)
            elif isinstance(message, MCPNotification):
                await self._handle_notification(message)
            else:
                raise MCPProtocolException("Unexpected message type")
            
            self.stats["requests_processed"] += 1
            return None
        
        except Exception as e:
            self.stats["requests_failed"] += 1
            self.logger.error(f"Error handling message: {e}")
            
            # Try to create error response if we have a request ID
            if isinstance(data, dict) and 'id' in data:
                error_response = self.protocol.create_error_response(
                    data['id'],
                    MCPErrorCode.INTERNAL_ERROR,
                    str(e)
                )
                return self.protocol.serialize_message(error_response)
            
            raise
    
    async def _handle_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Handle an MCP request."""
        try:
            # Check if server is initialized for non-initialize requests
            if request.method != MCPMethod.INITIALIZE and not self.initialized:
                return self.protocol.create_error_response(
                    request.id,
                    MCPErrorCode.INVALID_REQUEST,
                    "Server not initialized"
                )
            
            # Find handler
            handler = getattr(self, '_method_handlers', {}).get(request.method)
            if not handler:
                return self.protocol.create_error_response(
                    request.id,
                    MCPErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {request.method}"
                )
            
            # Call handler with timeout
            try:
                result = await asyncio.wait_for(
                    handler(request.params or {}),
                    timeout=self.request_timeout
                )
                return self.protocol.create_response(request.id, result=result)
            
            except asyncio.TimeoutError:
                return self.protocol.create_error_response(
                    request.id,
                    MCPErrorCode.TIMEOUT,
                    "Request timeout"
                )
        
        except MCPException as e:
            return self.protocol.create_error_response(
                request.id,
                e.error_code,
                e.message,
                e.details
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in request handler: {e}")
            self.logger.error(traceback.format_exc())
            return self.protocol.create_error_response(
                request.id,
                MCPErrorCode.INTERNAL_ERROR,
                "Internal server error"
            )
    
    async def _handle_notification(self, notification: MCPNotification):
        """Handle an MCP notification."""
        try:
            # Find handler
            handler = getattr(self, '_method_handlers', {}).get(notification.method)
            if handler:
                await handler(notification.params or {})
            else:
                self.logger.warning(f"No handler for notification: {notification.method}")
        
        except Exception as e:
            self.logger.error(f"Error handling notification: {e}")
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        try:
            init_params = MCPInitializeParams(**params)
        except Exception as e:
            raise MCPValidationException(f"Invalid initialize parameters: {e}")
        
        # Validate protocol version
        if not self.protocol.validate_protocol_version(init_params.protocolVersion):
            raise MCPProtocolException(
                f"Unsupported protocol version: {init_params.protocolVersion}"
            )
        
        # Store client info
        self.client_info = init_params.clientInfo
        self.client_capabilities = init_params.capabilities
        
        # Create server capabilities
        capabilities = MCPCapabilities(
            tools={} if self.tools else None,
            resources={} if self.resources else None,
            prompts={} if self.prompts else None,
            logging={}
        )
        
        # Create server info
        server_info = MCPServerInfo(
            name=self.name,
            version=self.version
        )
        
        # Mark as initialized
        self.initialized = True
        
        result = MCPInitializeResult(
            protocolVersion=self.protocol.protocol_version,
            capabilities=capabilities,
            serverInfo=server_info,
            instructions=self.description
        )
        
        self.logger.info(f"Initialized with client: {self.client_info.name} v{self.client_info.version}")
        
        return result.dict(exclude_none=True)
    
    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request."""
        return {}
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list tools request."""
        tools = [tool_info["definition"].dict() for tool_info in self.tools.values()]
        return {"tools": tools}
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call tool request."""
        try:
            tool_call = MCPToolCall(**params)
        except Exception as e:
            raise MCPValidationException(f"Invalid tool call parameters: {e}")
        
        # Find tool handler
        handler = self.tool_handlers.get(tool_call.name)
        if not handler:
            raise ToolNotFoundException(f"Tool not found: {tool_call.name}")
        
        try:
            # Call tool handler
            self.stats["tools_called"] += 1
            result = await handler(**tool_call.arguments)
            
            # Ensure result is in correct format
            if isinstance(result, MCPToolResult):
                return result.dict()
            elif isinstance(result, dict) and "content" in result:
                return result
            else:
                # Wrap simple results
                content = [create_text_content(str(result))]
                return MCPToolResult(content=content).dict()
        
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return error result
            content = [create_error_content(str(e))]
            return MCPToolResult(content=content, isError=True).dict()
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list resources request."""
        resources = [
            {
                "uri": uri,
                "name": info["name"],
                "description": info["description"],
                "mimeType": info["mimeType"]
            }
            for uri, info in self.resources.items()
        ]
        return {"resources": resources}
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read resource request."""
        uri = params.get("uri")
        if not uri:
            raise MCPValidationException("Resource URI is required")
        
        handler = self.resource_handlers.get(uri)
        if not handler:
            raise MCPValidationException(f"Resource not found: {uri}")
        
        try:
            result = await handler(uri)
            return result
        except Exception as e:
            self.logger.error(f"Resource read error: {e}")
            raise ToolExecutionException(f"Failed to read resource: {e}")
    
    async def _handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list prompts request."""
        prompts = [
            {
                "name": name,
                "description": info["description"],
                "arguments": info["arguments"]
            }
            for name, info in self.prompts.items()
        ]
        return {"prompts": prompts}
    
    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get prompt request."""
        name = params.get("name")
        if not name:
            raise MCPValidationException("Prompt name is required")
        
        handler = self.prompt_handlers.get(name)
        if not handler:
            raise MCPValidationException(f"Prompt not found: {name}")
        
        try:
            arguments = params.get("arguments", {})
            result = await handler(arguments)
            return result
        except Exception as e:
            self.logger.error(f"Prompt execution error: {e}")
            raise ToolExecutionException(f"Failed to execute prompt: {e}")
    
    async def _handle_set_log_level(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set log level request."""
        level = params.get("level")
        if not level:
            raise MCPValidationException("Log level is required")
        
        try:
            # Validate log level
            mcp_level = MCPLogLevel(level)
            self.log_level = mcp_level
            
            # Update logger level
            python_level = {
                MCPLogLevel.DEBUG: logging.DEBUG,
                MCPLogLevel.INFO: logging.INFO,
                MCPLogLevel.NOTICE: logging.INFO,
                MCPLogLevel.WARNING: logging.WARNING,
                MCPLogLevel.ERROR: logging.ERROR,
                MCPLogLevel.CRITICAL: logging.CRITICAL,
                MCPLogLevel.ALERT: logging.CRITICAL,
                MCPLogLevel.EMERGENCY: logging.CRITICAL
            }.get(mcp_level, logging.INFO)
            
            self.logger.setLevel(python_level)
            
            self.logger.info(f"Log level set to: {level}")
            return {}
        
        except ValueError:
            raise MCPValidationException(f"Invalid log level: {level}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        uptime = datetime.now() - self.stats["start_time"]
        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "tools_registered": len(self.tools),
            "resources_registered": len(self.resources),
            "prompts_registered": len(self.prompts),
            "initialized": self.initialized
        }
    
    async def shutdown(self):
        """Shutdown the server gracefully."""
        self.logger.info("Shutting down MCP server")
        self.initialized = False
        
        # Log final stats
        stats = self.get_stats()
        self.logger.info(f"Final stats: {stats}")