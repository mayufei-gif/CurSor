#!/usr/bin/env python3
"""
MCP Protocol Data Models

Defines the data structures and message formats for the Model Context Protocol (MCP).
This module handles serialization, validation, and protocol compliance.

Author: PDF-MCP Team
License: MIT
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, Field, field_validator, model_validator

from .exceptions import MCPProtocolException, MCPValidationException


class MCPMessageType(str, Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPMethod(str, Enum):
    """MCP method names."""
    # Core protocol methods
    INITIALIZE = "initialize"
    PING = "ping"
    
    # Tool methods
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    
    # Resource methods
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    
    # Prompt methods
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"
    
    # Logging methods
    SET_LOG_LEVEL = "logging/setLevel"


class MCPCapability(str, Enum):
    """MCP server capabilities."""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    LOGGING = "logging"


class MCPLogLevel(str, Enum):
    """MCP log levels."""
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"


class MCPToolInputSchema(BaseModel):
    """Schema for tool input parameters."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    additionalProperties: bool = False
    
    class Config:
        extra = "allow"


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    inputSchema: MCPToolInputSchema = Field(..., description="Input schema")
    
    class Config:
        extra = "forbid"


class MCPResource(BaseModel):
    """MCP resource definition."""
    uri: str = Field(..., description="Resource URI")
    name: str = Field(..., description="Resource name")
    description: Optional[str] = Field(None, description="Resource description")
    mimeType: Optional[str] = Field(None, description="MIME type")
    
    class Config:
        extra = "forbid"


class MCPPrompt(BaseModel):
    """MCP prompt definition."""
    name: str = Field(..., description="Prompt name")
    description: str = Field(..., description="Prompt description")
    arguments: List[Dict[str, Any]] = Field(default_factory=list, description="Prompt arguments")
    
    class Config:
        extra = "forbid"


class MCPServerInfo(BaseModel):
    """MCP server information."""
    name: str = Field(..., description="Server name")
    version: str = Field(..., description="Server version")
    
    class Config:
        extra = "forbid"


class MCPClientInfo(BaseModel):
    """MCP client information."""
    name: str = Field(..., description="Client name")
    version: str = Field(..., description="Client version")
    
    class Config:
        extra = "forbid"


class MCPImplementation(BaseModel):
    """MCP implementation details."""
    name: str = Field(..., description="Implementation name")
    version: str = Field(..., description="Implementation version")
    
    class Config:
        extra = "forbid"


class MCPCapabilities(BaseModel):
    """MCP capabilities."""
    tools: Optional[Dict[str, Any]] = Field(None, description="Tool capabilities")
    resources: Optional[Dict[str, Any]] = Field(None, description="Resource capabilities")
    prompts: Optional[Dict[str, Any]] = Field(None, description="Prompt capabilities")
    logging: Optional[Dict[str, Any]] = Field(None, description="Logging capabilities")
    
    class Config:
        extra = "allow"


class MCPInitializeParams(BaseModel):
    """Parameters for initialize request."""
    protocolVersion: str = Field(..., description="Protocol version")
    capabilities: MCPCapabilities = Field(..., description="Client capabilities")
    clientInfo: MCPClientInfo = Field(..., description="Client information")
    
    class Config:
        extra = "forbid"


class MCPInitializeResult(BaseModel):
    """Result for initialize request."""
    protocolVersion: str = Field(..., description="Protocol version")
    capabilities: MCPCapabilities = Field(..., description="Server capabilities")
    serverInfo: MCPServerInfo = Field(..., description="Server information")
    instructions: Optional[str] = Field(None, description="Server instructions")
    
    class Config:
        extra = "forbid"


class MCPToolCall(BaseModel):
    """MCP tool call request."""
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    
    class Config:
        extra = "forbid"


class MCPToolResult(BaseModel):
    """MCP tool call result."""
    content: List[Dict[str, Any]] = Field(..., description="Result content")
    isError: bool = Field(False, description="Whether result is an error")
    
    class Config:
        extra = "forbid"


class MCPError(BaseModel):
    """MCP error object."""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional error data")
    
    class Config:
        extra = "forbid"


class MCPMessage(BaseModel):
    """Base MCP message."""
    jsonrpc: str = Field("2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(None, description="Message ID")
    
    class Config:
        extra = "allow"
    
    @field_validator('jsonrpc')
    @classmethod
    def validate_jsonrpc(cls, v):
        if v != "2.0":
            raise ValueError("jsonrpc must be '2.0'")
        return v


class MCPRequest(MCPMessage):
    """MCP request message."""
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")
    
    class Config:
        extra = "forbid"
    
    @field_validator('id')
    @classmethod
    def validate_id_required(cls, v):
        if v is None:
            raise ValueError("Request ID is required")
        return v


class MCPResponse(MCPMessage):
    """MCP response message."""
    result: Optional[Any] = Field(None, description="Method result")
    error: Optional[MCPError] = Field(None, description="Error object")
    
    class Config:
        extra = "forbid"
    
    @model_validator(mode='before')
    @classmethod
    def validate_result_or_error(cls, values):
        if isinstance(values, dict):
            result = values.get('result')
            error = values.get('error')
            
            if result is not None and error is not None:
                raise ValueError("Response cannot have both result and error")
            if result is None and error is None:
                raise ValueError("Response must have either result or error")
        
        return values
    
    @field_validator('id')
    @classmethod
    def validate_id_required(cls, v):
        if v is None:
            raise ValueError("Response ID is required")
        return v


class MCPNotification(MCPMessage):
    """MCP notification message."""
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")
    
    class Config:
        extra = "forbid"
    
    @field_validator('id')
    @classmethod
    def validate_id_forbidden(cls, v):
        if v is not None:
            raise ValueError("Notification cannot have ID")
        return v


class MCPProtocolHandler:
    """Handles MCP protocol operations."""
    
    def __init__(self, protocol_version: str = "2024-11-05"):
        self.protocol_version = protocol_version
        self.request_counter = 0
    
    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self.request_counter += 1
        return f"{uuid.uuid4().hex[:8]}-{self.request_counter}"
    
    def create_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> MCPRequest:
        """Create an MCP request."""
        if request_id is None:
            request_id = self.generate_request_id()
        
        return MCPRequest(
            id=request_id,
            method=method,
            params=params
        )
    
    def create_response(
        self,
        request_id: Union[str, int],
        result: Optional[Any] = None,
        error: Optional[MCPError] = None
    ) -> MCPResponse:
        """Create an MCP response."""
        return MCPResponse(
            id=request_id,
            result=result,
            error=error
        )
    
    def create_error_response(
        self,
        request_id: Union[str, int],
        code: int,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """Create an MCP error response."""
        error = MCPError(code=code, message=message, data=data)
        return self.create_response(request_id, error=error)
    
    def create_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> MCPNotification:
        """Create an MCP notification."""
        return MCPNotification(
            method=method,
            params=params
        )
    
    def parse_message(self, data: Union[str, bytes, Dict[str, Any]]) -> Union[MCPRequest, MCPResponse, MCPNotification]:
        """Parse an MCP message from JSON data."""
        try:
            if isinstance(data, (str, bytes)):
                message_dict = json.loads(data)
            else:
                message_dict = data
            
            # Determine message type
            if 'method' in message_dict:
                if 'id' in message_dict and message_dict['id'] is not None:
                    return MCPRequest(**message_dict)
                else:
                    return MCPNotification(**message_dict)
            elif 'result' in message_dict or 'error' in message_dict:
                return MCPResponse(**message_dict)
            else:
                raise MCPProtocolException("Invalid message format")
        
        except json.JSONDecodeError as e:
            raise MCPProtocolException(f"Invalid JSON: {e}")
        except Exception as e:
            raise MCPProtocolException(f"Failed to parse message: {e}")
    
    def serialize_message(self, message: Union[MCPRequest, MCPResponse, MCPNotification]) -> str:
        """Serialize an MCP message to JSON."""
        try:
            return message.json(exclude_none=True, ensure_ascii=False)
        except Exception as e:
            raise MCPProtocolException(f"Failed to serialize message: {e}")
    
    def validate_protocol_version(self, version: str) -> bool:
        """Validate protocol version compatibility."""
        # For now, we only support the current version
        return version == self.protocol_version
    
    def create_tool_definition(self, name: str, description: str, input_schema: Dict[str, Any]) -> MCPTool:
        """Create a tool definition."""
        schema = MCPToolInputSchema(**input_schema)
        return MCPTool(
            name=name,
            description=description,
            inputSchema=schema
        )
    
    def create_tool_result(
        self,
        content: List[Dict[str, Any]],
        is_error: bool = False
    ) -> MCPToolResult:
        """Create a tool result."""
        return MCPToolResult(
            content=content,
            isError=is_error
        )


# Standard MCP error codes
class MCPErrorCode:
    """Standard MCP error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes (application-specific)
    TOOL_NOT_FOUND = -32000
    TOOL_EXECUTION_ERROR = -32001
    RESOURCE_NOT_FOUND = -32002
    UNAUTHORIZED = -32003
    RATE_LIMITED = -32004
    TIMEOUT = -32005


# Content types for tool results
class MCPContentType:
    """Standard content types for MCP messages."""
    TEXT = "text"
    IMAGE = "image"
    RESOURCE = "resource"
    PROGRESS = "progress"
    ERROR = "error"


def create_text_content(text: str) -> Dict[str, Any]:
    """Create text content for tool results."""
    return {
        "type": MCPContentType.TEXT,
        "text": text
    }


def create_image_content(data: str, mime_type: str) -> Dict[str, Any]:
    """Create image content for tool results."""
    return {
        "type": MCPContentType.IMAGE,
        "data": data,
        "mimeType": mime_type
    }


def create_resource_content(uri: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
    """Create resource content for tool results."""
    content = {
        "type": MCPContentType.RESOURCE,
        "resource": {
            "uri": uri
        }
    }
    if mime_type:
        content["resource"]["mimeType"] = mime_type
    return content


def create_progress_content(progress: float, total: Optional[float] = None) -> Dict[str, Any]:
    """Create progress content for tool results."""
    content = {
        "type": MCPContentType.PROGRESS,
        "progress": progress
    }
    if total is not None:
        content["total"] = total
    return content


def create_error_content(error: str, code: Optional[str] = None) -> Dict[str, Any]:
    """Create error content for tool results."""
    content = {
        "type": MCPContentType.ERROR,
        "error": error
    }
    if code:
        content["code"] = code
    return content