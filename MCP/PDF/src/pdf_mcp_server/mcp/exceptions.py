#!/usr/bin/env python3
"""
MCP Protocol Exceptions

Defines custom exceptions for the MCP protocol implementation.
These exceptions provide specific error handling for MCP-related operations.

Author: PDF-MCP Team
License: MIT
"""

from typing import Optional, Dict, Any


class MCPException(Exception):
    """Base exception for all MCP-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'MCP_ERROR'
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            'error': {
                'code': self.error_code,
                'message': self.message,
                'details': self.details
            }
        }
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Code: {self.error_code}, Details: {self.details})"
        return f"{self.message} (Code: {self.error_code})"


class MCPProtocolException(MCPException):
    """Exception for MCP protocol-level errors."""
    
    def __init__(
        self,
        message: str,
        protocol_version: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, 'PROTOCOL_ERROR', details)
        self.protocol_version = protocol_version
        if protocol_version:
            self.details['protocol_version'] = protocol_version


class ToolNotFoundException(MCPException):
    """Exception raised when a requested tool is not found."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: Optional[list] = None
    ):
        message = f"Tool '{tool_name}' not found"
        details = {'tool_name': tool_name}
        if available_tools:
            details['available_tools'] = available_tools
            message += f". Available tools: {', '.join(available_tools)}"
        
        super().__init__(message, 'TOOL_NOT_FOUND', details)
        self.tool_name = tool_name
        self.available_tools = available_tools


class InvalidToolCallException(MCPException):
    """Exception raised when a tool call is invalid."""
    
    def __init__(
        self,
        tool_name: str,
        reason: str,
        provided_args: Optional[Dict[str, Any]] = None,
        expected_args: Optional[Dict[str, Any]] = None
    ):
        message = f"Invalid call to tool '{tool_name}': {reason}"
        details = {
            'tool_name': tool_name,
            'reason': reason
        }
        
        if provided_args is not None:
            details['provided_args'] = provided_args
        if expected_args is not None:
            details['expected_args'] = expected_args
        
        super().__init__(message, 'INVALID_TOOL_CALL', details)
        self.tool_name = tool_name
        self.reason = reason
        self.provided_args = provided_args
        self.expected_args = expected_args


class ToolExecutionException(MCPException):
    """Exception raised when tool execution fails."""
    
    def __init__(
        self,
        tool_name: str,
        execution_error: str,
        original_exception: Optional[Exception] = None
    ):
        message = f"Tool '{tool_name}' execution failed: {execution_error}"
        details = {
            'tool_name': tool_name,
            'execution_error': execution_error
        }
        
        if original_exception:
            details['original_exception'] = {
                'type': type(original_exception).__name__,
                'message': str(original_exception)
            }
        
        super().__init__(message, 'TOOL_EXECUTION_ERROR', details)
        self.tool_name = tool_name
        self.execution_error = execution_error
        self.original_exception = original_exception


class MCPTimeoutException(MCPException):
    """Exception raised when MCP operations timeout."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float
    ):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        details = {
            'operation': operation,
            'timeout_seconds': timeout_seconds
        }
        
        super().__init__(message, 'TIMEOUT_ERROR', details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class MCPAuthenticationException(MCPException):
    """Exception raised for authentication failures."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 'AUTHENTICATION_ERROR')


class MCPAuthorizationException(MCPException):
    """Exception raised for authorization failures."""
    
    def __init__(
        self,
        resource: str,
        required_permission: str
    ):
        message = f"Access denied to resource '{resource}'. Required permission: {required_permission}"
        details = {
            'resource': resource,
            'required_permission': required_permission
        }
        
        super().__init__(message, 'AUTHORIZATION_ERROR', details)
        self.resource = resource
        self.required_permission = required_permission


class MCPRateLimitException(MCPException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after_seconds: Optional[int] = None
    ):
        message = f"Rate limit exceeded: {limit} requests per {window_seconds} seconds"
        details = {
            'limit': limit,
            'window_seconds': window_seconds
        }
        
        if retry_after_seconds:
            details['retry_after_seconds'] = retry_after_seconds
            message += f". Retry after {retry_after_seconds} seconds"
        
        super().__init__(message, 'RATE_LIMIT_ERROR', details)
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after_seconds = retry_after_seconds


class MCPValidationException(MCPException):
    """Exception raised for validation errors."""
    
    def __init__(
        self,
        field: str,
        value: Any,
        reason: str
    ):
        message = f"Validation failed for field '{field}': {reason}"
        details = {
            'field': field,
            'value': value,
            'reason': reason
        }
        
        super().__init__(message, 'VALIDATION_ERROR', details)
        self.field = field
        self.value = value
        self.reason = reason


class MCPResourceException(MCPException):
    """Exception raised for resource-related errors."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        operation: str,
        reason: str
    ):
        message = f"Failed to {operation} {resource_type} '{resource_id}': {reason}"
        details = {
            'resource_type': resource_type,
            'resource_id': resource_id,
            'operation': operation,
            'reason': reason
        }
        
        super().__init__(message, 'RESOURCE_ERROR', details)
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.operation = operation
        self.reason = reason


class MCPConfigurationException(MCPException):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        config_key: str,
        reason: str
    ):
        message = f"Configuration error for '{config_key}': {reason}"
        details = {
            'config_key': config_key,
            'reason': reason
        }
        
        super().__init__(message, 'CONFIGURATION_ERROR', details)
        self.config_key = config_key
        self.reason = reason


# Exception mapping for HTTP status codes
MCP_EXCEPTION_HTTP_MAPPING = {
    MCPException: 500,
    MCPProtocolException: 400,
    ToolNotFoundException: 404,
    InvalidToolCallException: 400,
    ToolExecutionException: 500,
    MCPTimeoutException: 408,
    MCPAuthenticationException: 401,
    MCPAuthorizationException: 403,
    MCPRateLimitException: 429,
    MCPValidationException: 400,
    MCPResourceException: 404,
    MCPConfigurationException: 500
}


def get_http_status_for_exception(exception: Exception) -> int:
    """Get appropriate HTTP status code for an exception."""
    for exc_type, status_code in MCP_EXCEPTION_HTTP_MAPPING.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500  # Default to internal server error


def create_error_response(
    exception: Exception,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standardized error response from an exception."""
    if isinstance(exception, MCPException):
        error_dict = exception.to_dict()
    else:
        error_dict = {
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(exception),
                'details': {
                    'exception_type': type(exception).__name__
                }
            }
        }
    
    if request_id:
        error_dict['request_id'] = request_id
    
    return error_dict