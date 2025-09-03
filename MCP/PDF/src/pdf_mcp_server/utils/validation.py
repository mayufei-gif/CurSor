"""Validation utilities for PDF-MCP server.

This module provides validation functions for PDF files, processing requests,
and other input data.
"""

import re
import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import logging
from urllib.parse import urlparse

from ..models import ProcessingRequest, ProcessingMode, OutputFormat, TableEngine
from .exceptions import ValidationError, FileNotFoundError, FileSizeError
from .file_utils import is_pdf_file, get_file_size, check_file_size


def validate_pdf_file(file_path: Union[str, Path], 
                     max_size: Optional[int] = None,
                     check_readable: bool = True) -> bool:
    """Validate PDF file.
    
    Args:
        file_path: Path to PDF file
        max_size: Optional maximum file size in bytes
        check_readable: Whether to check if file is readable
        
    Returns:
        True if valid
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If file is not a valid PDF
        FileSizeError: If file is too large
    """
    file_path = Path(file_path)
    logger = logging.getLogger(__name__)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}",
            file_path=str(file_path)
        )
    
    # Check if it's a file (not directory)
    if not file_path.is_file():
        raise ValidationError(
            f"Path is not a file: {file_path}",
            field_name="file_path",
            field_value=str(file_path)
        )
    
    # Check file size if specified
    if max_size is not None:
        check_file_size(file_path, max_size)
    
    # Check if it's a PDF file
    if not is_pdf_file(file_path):
        raise ValidationError(
            f"File is not a valid PDF: {file_path}",
            field_name="file_path",
            field_value=str(file_path)
        )
    
    # Check if file is readable
    if check_readable:
        try:
            with open(file_path, 'rb') as f:
                # Try to read first few bytes
                header = f.read(1024)
                if not header:
                    raise ValidationError(
                        f"PDF file is empty: {file_path}",
                        field_name="file_path",
                        field_value=str(file_path)
                    )
        except PermissionError as e:
            raise ValidationError(
                f"Permission denied reading PDF file: {file_path}",
                field_name="file_path",
                field_value=str(file_path)
            ) from e
        except Exception as e:
            raise ValidationError(
                f"Cannot read PDF file: {file_path}",
                field_name="file_path",
                field_value=str(file_path)
            ) from e
    
    logger.debug(f"PDF file validation passed: {file_path}")
    return True


def validate_processing_request(request: ProcessingRequest) -> bool:
    """Validate processing request.
    
    Args:
        request: Processing request to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If request is invalid
    """
    logger = logging.getLogger(__name__)
    
    # Validate file path if provided
    if request.file_path:
        try:
            validate_pdf_file(request.file_path)
        except Exception as e:
            raise ValidationError(
                f"Invalid file_path in request: {e}",
                field_name="file_path",
                field_value=request.file_path
            ) from e
    
    # Validate URL if provided
    if request.file_url:
        if not validate_url(request.file_url):
            raise ValidationError(
                f"Invalid file_url in request: {request.file_url}",
                field_name="file_url",
                field_value=request.file_url
            )
    
    # Ensure either file_path or file_url is provided
    if not request.file_path and not request.file_url:
        raise ValidationError(
            "Either file_path or file_url must be provided",
            field_name="file_path,file_url",
            field_value="None"
        )
    
    # Validate page range if provided
    if request.pages:
        if not validate_page_range(request.pages):
            raise ValidationError(
                f"Invalid page range: {request.pages}",
                field_name="pages",
                field_value=str(request.pages)
            )
    
    # Validate processing modes
    if request.modes:
        for mode in request.modes:
            if mode not in ProcessingMode:
                raise ValidationError(
                    f"Invalid processing mode: {mode}",
                    field_name="modes",
                    field_value=str(mode)
                )
    
    # Validate output format
    if request.output_format and request.output_format not in OutputFormat:
        raise ValidationError(
            f"Invalid output format: {request.output_format}",
            field_name="output_format",
            field_value=str(request.output_format)
        )
    
    # Validate table engine
    if request.table_engine and request.table_engine not in TableEngine:
        raise ValidationError(
            f"Invalid table engine: {request.table_engine}",
            field_name="table_engine",
            field_value=str(request.table_engine)
        )
    
    # Validate options if provided
    if request.options:
        validate_processing_options(request.options)
    
    logger.debug(f"Processing request validation passed")
    return True


def validate_url(url: str) -> bool:
    """Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_page_range(pages: Union[List[int], str]) -> bool:
    """Validate page range specification.
    
    Args:
        pages: Page range as list of integers or string
        
    Returns:
        True if valid page range
    """
    if isinstance(pages, list):
        # List of page numbers
        return all(isinstance(p, int) and p > 0 for p in pages)
    
    elif isinstance(pages, str):
        # String format like "1-5,7,9-12"
        try:
            parts = pages.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Range like "1-5"
                    start, end = part.split('-', 1)
                    start_num = int(start.strip())
                    end_num = int(end.strip())
                    if start_num <= 0 or end_num <= 0 or start_num > end_num:
                        return False
                else:
                    # Single page number
                    page_num = int(part)
                    if page_num <= 0:
                        return False
            return True
        except (ValueError, AttributeError):
            return False
    
    return False


def validate_processing_options(options: Dict[str, Any]) -> bool:
    """Validate processing options.
    
    Args:
        options: Processing options dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If options are invalid
    """
    # Define valid option keys and their types
    valid_options = {
        # Text extraction options
        'text_engine': str,
        'include_coordinates': bool,
        'clean_text': bool,
        'extract_metadata': bool,
        
        # Table extraction options
        'table_areas': list,
        'table_columns': list,
        'table_flavor': str,
        'table_pandas_options': dict,
        
        # OCR options
        'ocr_language': str,
        'ocr_dpi': int,
        'ocr_psm': int,
        'ocr_oem': int,
        
        # Formula extraction options
        'formula_confidence_threshold': float,
        'formula_model': str,
        'formula_device': str,
        
        # General options
        'timeout': int,
        'max_workers': int,
        'temp_dir': str,
        'keep_temp_files': bool,
    }
    
    for key, value in options.items():
        if key not in valid_options:
            raise ValidationError(
                f"Unknown processing option: {key}",
                field_name="options",
                field_value=key
            )
        
        expected_type = valid_options[key]
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Invalid type for option '{key}': expected {expected_type.__name__}, got {type(value).__name__}",
                field_name="options",
                field_value=f"{key}={value}"
            )
        
        # Additional validation for specific options
        if key == 'ocr_dpi' and (value < 72 or value > 600):
            raise ValidationError(
                f"OCR DPI must be between 72 and 600, got {value}",
                field_name="options",
                field_value=f"ocr_dpi={value}"
            )
        
        elif key == 'ocr_psm' and (value < 0 or value > 13):
            raise ValidationError(
                f"OCR PSM must be between 0 and 13, got {value}",
                field_name="options",
                field_value=f"ocr_psm={value}"
            )
        
        elif key == 'ocr_oem' and (value < 0 or value > 3):
            raise ValidationError(
                f"OCR OEM must be between 0 and 3, got {value}",
                field_name="options",
                field_value=f"ocr_oem={value}"
            )
        
        elif key == 'formula_confidence_threshold' and (value < 0.0 or value > 1.0):
            raise ValidationError(
                f"Formula confidence threshold must be between 0.0 and 1.0, got {value}",
                field_name="options",
                field_value=f"formula_confidence_threshold={value}"
            )
        
        elif key == 'timeout' and value <= 0:
            raise ValidationError(
                f"Timeout must be positive, got {value}",
                field_name="options",
                field_value=f"timeout={value}"
            )
        
        elif key == 'max_workers' and value <= 0:
            raise ValidationError(
                f"Max workers must be positive, got {value}",
                field_name="options",
                field_value=f"max_workers={value}"
            )
    
    return True


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    # Windows invalid characters: < > : " | ? * \ /
    # Also remove control characters
    invalid_chars = r'[<>:"|?*\\/\x00-\x1f\x7f]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed_file'
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        max_name_length = max_length - len(ext)
        sanitized = name[:max_name_length] + ext
    
    # Avoid reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = os.path.splitext(sanitized)[0].upper()
    if name_without_ext in reserved_names:
        sanitized = f"_{sanitized}"
    
    return sanitized


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension.
    
    Args:
        filename: Filename to check
        allowed_extensions: List of allowed extensions (with or without dots)
        
    Returns:
        True if extension is allowed
    """
    file_ext = Path(filename).suffix.lower()
    
    # Normalize extensions (ensure they start with dot)
    normalized_extensions = []
    for ext in allowed_extensions:
        if not ext.startswith('.'):
            ext = f'.{ext}'
        normalized_extensions.append(ext.lower())
    
    return file_ext in normalized_extensions


def validate_mime_type(file_path: Union[str, Path], allowed_types: List[str]) -> bool:
    """Validate file MIME type.
    
    Args:
        file_path: Path to file
        allowed_types: List of allowed MIME types
        
    Returns:
        True if MIME type is allowed
    """
    import mimetypes
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type in allowed_types if mime_type else False


def validate_json_structure(data: Dict[str, Any], required_fields: List[str], 
                          optional_fields: Optional[List[str]] = None) -> bool:
    """Validate JSON structure.
    
    Args:
        data: JSON data to validate
        required_fields: List of required field names
        optional_fields: List of optional field names
        
    Returns:
        True if structure is valid
        
    Raises:
        ValidationError: If structure is invalid
    """
    if not isinstance(data, dict):
        raise ValidationError(
            "Data must be a dictionary",
            field_name="data",
            field_value=str(type(data))
        )
    
    # Check required fields
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            field_name="required_fields",
            field_value=str(missing_fields)
        )
    
    # Check for unexpected fields
    if optional_fields is not None:
        allowed_fields = set(required_fields + optional_fields)
        unexpected_fields = []
        for field in data.keys():
            if field not in allowed_fields:
                unexpected_fields.append(field)
        
        if unexpected_fields:
            raise ValidationError(
                f"Unexpected fields: {', '.join(unexpected_fields)}",
                field_name="unexpected_fields",
                field_value=str(unexpected_fields)
            )
    
    return True


def validate_coordinate_bounds(bbox: List[float], page_width: float, page_height: float) -> bool:
    """Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box as [x0, y0, x1, y1]
        page_width: Page width
        page_height: Page height
        
    Returns:
        True if coordinates are valid
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    if len(bbox) != 4:
        raise ValidationError(
            f"Bounding box must have 4 coordinates, got {len(bbox)}",
            field_name="bbox",
            field_value=str(bbox)
        )
    
    x0, y0, x1, y1 = bbox
    
    # Check coordinate types
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        raise ValidationError(
            "All coordinates must be numbers",
            field_name="bbox",
            field_value=str(bbox)
        )
    
    # Check coordinate order
    if x0 >= x1 or y0 >= y1:
        raise ValidationError(
            "Invalid coordinate order: x0 < x1 and y0 < y1 required",
            field_name="bbox",
            field_value=str(bbox)
        )
    
    # Check bounds
    if x0 < 0 or y0 < 0 or x1 > page_width or y1 > page_height:
        raise ValidationError(
            f"Coordinates out of page bounds: page=({page_width}x{page_height}), bbox={bbox}",
            field_name="bbox",
            field_value=str(bbox)
        )
    
    return True


def validate_language_code(language: str) -> bool:
    """Validate ISO language code.
    
    Args:
        language: Language code to validate
        
    Returns:
        True if valid language code
    """
    # Basic validation for ISO 639-1 (2-letter) and ISO 639-2 (3-letter) codes
    # Also support common OCR language codes
    valid_patterns = [
        r'^[a-z]{2}$',  # ISO 639-1 (e.g., 'en', 'zh')
        r'^[a-z]{3}$',  # ISO 639-2 (e.g., 'eng', 'chi')
        r'^[a-z]{2}_[A-Z]{2}$',  # Locale format (e.g., 'en_US', 'zh_CN')
        r'^[a-z]{3}_[A-Z]{3}$',  # Extended locale (e.g., 'eng_USA')
    ]
    
    return any(re.match(pattern, language) for pattern in valid_patterns)