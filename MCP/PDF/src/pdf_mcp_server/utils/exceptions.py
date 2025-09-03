"""Custom exceptions for PDF-MCP server.

This module defines custom exception classes for different types of errors
that can occur during PDF processing.
"""

from typing import Optional, Any, Dict


class PDFMCPError(Exception):
    """Base exception class for PDF-MCP server."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary.
        
        Returns:
            Exception as dictionary
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class PDFProcessingError(PDFMCPError):
    """Exception raised when PDF processing fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 page_number: Optional[int] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            file_path: Optional path to the PDF file
            page_number: Optional page number where error occurred
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        if page_number is not None:
            details['page_number'] = page_number
        
        super().__init__(message, error_code="PDF_PROCESSING_ERROR", details=details)
        self.file_path = file_path
        self.page_number = page_number


class ConfigurationError(PDFMCPError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Optional[Any] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            config_key: Optional configuration key that caused the error
            config_value: Optional configuration value that caused the error
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = str(config_value)
        
        super().__init__(message, error_code="CONFIGURATION_ERROR", details=details)
        self.config_key = config_key
        self.config_value = config_value


class ModelLoadError(PDFMCPError):
    """Exception raised when a model fails to load."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 model_path: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            model_name: Optional name of the model that failed to load
            model_path: Optional path to the model
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if model_path:
            details['model_path'] = model_path
        
        super().__init__(message, error_code="MODEL_LOAD_ERROR", details=details)
        self.model_name = model_name
        self.model_path = model_path


class ValidationError(PDFMCPError):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 field_value: Optional[Any] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            field_name: Optional name of the field that failed validation
            field_value: Optional value that failed validation
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)
        
        super().__init__(message, error_code="VALIDATION_ERROR", details=details)
        self.field_name = field_name
        self.field_value = field_value


class FileNotFoundError(PDFMCPError):
    """Exception raised when a file is not found."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            file_path: Optional path to the missing file
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        
        super().__init__(message, error_code="FILE_NOT_FOUND", details=details)
        self.file_path = file_path


class FileAccessError(PDFMCPError):
    """Exception raised when file access is denied."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            file_path: Optional path to the file
            operation: Optional operation that was attempted
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        if operation:
            details['operation'] = operation
        
        super().__init__(message, error_code="FILE_ACCESS_ERROR", details=details)
        self.file_path = file_path
        self.operation = operation


class FileSizeError(PDFMCPError):
    """Exception raised when file size exceeds limits."""
    
    def __init__(self, message: str, file_size: Optional[int] = None, 
                 max_size: Optional[int] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            file_size: Optional actual file size
            max_size: Optional maximum allowed size
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if file_size is not None:
            details['file_size'] = file_size
        if max_size is not None:
            details['max_size'] = max_size
        
        super().__init__(message, error_code="FILE_SIZE_ERROR", details=details)
        self.file_size = file_size
        self.max_size = max_size


class TextExtractionError(PDFProcessingError):
    """Exception raised when text extraction fails."""
    
    def __init__(self, message: str, engine: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            engine: Optional extraction engine that failed
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if engine:
            details['engine'] = engine
        
        kwargs['details'] = details
        super().__init__(message, error_code="TEXT_EXTRACTION_ERROR", **kwargs)
        self.engine = engine


class TableExtractionError(PDFProcessingError):
    """Exception raised when table extraction fails."""
    
    def __init__(self, message: str, engine: Optional[str] = None, 
                 table_index: Optional[int] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            engine: Optional extraction engine that failed
            table_index: Optional index of the table that failed
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if engine:
            details['engine'] = engine
        if table_index is not None:
            details['table_index'] = table_index
        
        kwargs['details'] = details
        super().__init__(message, error_code="TABLE_EXTRACTION_ERROR", **kwargs)
        self.engine = engine
        self.table_index = table_index


class OCRError(PDFProcessingError):
    """Exception raised when OCR processing fails."""
    
    def __init__(self, message: str, engine: Optional[str] = None, 
                 language: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            engine: Optional OCR engine that failed
            language: Optional OCR language
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if engine:
            details['engine'] = engine
        if language:
            details['language'] = language
        
        kwargs['details'] = details
        super().__init__(message, error_code="OCR_ERROR", **kwargs)
        self.engine = engine
        self.language = language


class FormulaExtractionError(PDFProcessingError):
    """Exception raised when formula extraction fails."""
    
    def __init__(self, message: str, model: Optional[str] = None, 
                 formula_index: Optional[int] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            model: Optional formula recognition model that failed
            formula_index: Optional index of the formula that failed
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if model:
            details['model'] = model
        if formula_index is not None:
            details['formula_index'] = formula_index
        
        kwargs['details'] = details
        super().__init__(message, error_code="FORMULA_EXTRACTION_ERROR", **kwargs)
        self.model = model
        self.formula_index = formula_index


class GrobidError(PDFProcessingError):
    """Exception raised when GROBID processing fails."""
    
    def __init__(self, message: str, grobid_url: Optional[str] = None, 
                 status_code: Optional[int] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            grobid_url: Optional GROBID service URL
            status_code: Optional HTTP status code
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if grobid_url:
            details['grobid_url'] = grobid_url
        if status_code is not None:
            details['status_code'] = status_code
        
        kwargs['details'] = details
        super().__init__(message, error_code="GROBID_ERROR", **kwargs)
        self.grobid_url = grobid_url
        self.status_code = status_code


class TimeoutError(PDFMCPError):
    """Exception raised when an operation times out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, 
                 operation: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            timeout_seconds: Optional timeout duration
            operation: Optional operation that timed out
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if timeout_seconds is not None:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        
        super().__init__(message, error_code="TIMEOUT_ERROR", details=details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class DependencyError(PDFMCPError):
    """Exception raised when a required dependency is missing."""
    
    def __init__(self, message: str, dependency: Optional[str] = None, 
                 install_command: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            dependency: Optional name of the missing dependency
            install_command: Optional command to install the dependency
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if dependency:
            details['dependency'] = dependency
        if install_command:
            details['install_command'] = install_command
        
        super().__init__(message, error_code="DEPENDENCY_ERROR", details=details)
        self.dependency = dependency
        self.install_command = install_command


class ResourceError(PDFMCPError):
    """Exception raised when system resources are insufficient."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 required: Optional[str] = None, available: Optional[str] = None, **kwargs):
        """Initialize the exception.
        
        Args:
            message: Error message
            resource_type: Optional type of resource (memory, disk, etc.)
            required: Optional required amount
            available: Optional available amount
            **kwargs: Additional arguments for base class
        """
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if required:
            details['required'] = required
        if available:
            details['available'] = available
        
        super().__init__(message, error_code="RESOURCE_ERROR", details=details)
        self.resource_type = resource_type
        self.required = required
        self.available = available