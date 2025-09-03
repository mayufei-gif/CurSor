"""File utility functions for PDF-MCP server.

This module provides utility functions for file operations, validation,
and management.
"""

import os
import shutil
import hashlib
import tempfile
import mimetypes
from pathlib import Path
from typing import Optional, List, Union, Tuple
import logging

from .exceptions import FileNotFoundError, FileAccessError, FileSizeError, ValidationError


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
        
    Raises:
        FileAccessError: If directory cannot be created
    """
    dir_path = Path(directory)
    
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    except PermissionError as e:
        raise FileAccessError(
            f"Permission denied creating directory: {dir_path}",
            file_path=str(dir_path),
            operation="create_directory"
        ) from e
    except Exception as e:
        raise FileAccessError(
            f"Failed to create directory: {dir_path}",
            file_path=str(dir_path),
            operation="create_directory"
        ) from e


def cleanup_temp_files(temp_dir: Union[str, Path], pattern: str = "*", 
                      max_age_hours: Optional[float] = None) -> int:
    """Clean up temporary files.
    
    Args:
        temp_dir: Temporary directory path
        pattern: File pattern to match (default: all files)
        max_age_hours: Optional maximum age in hours for files to keep
        
    Returns:
        Number of files cleaned up
        
    Raises:
        FileAccessError: If cleanup fails
    """
    temp_path = Path(temp_dir)
    
    if not temp_path.exists():
        return 0
    
    logger = logging.getLogger(__name__)
    cleaned_count = 0
    
    try:
        import time
        current_time = time.time()
        
        for file_path in temp_path.glob(pattern):
            try:
                # Check file age if specified
                if max_age_hours is not None:
                    file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                    if file_age_hours < max_age_hours:
                        continue
                
                if file_path.is_file():
                    file_path.unlink()
                    cleaned_count += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    cleaned_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")
                continue
        
        logger.info(f"Cleaned up {cleaned_count} temporary files from {temp_dir}")
        return cleaned_count
        
    except Exception as e:
        raise FileAccessError(
            f"Failed to cleanup temporary files in {temp_dir}",
            file_path=str(temp_dir),
            operation="cleanup"
        ) from e


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Calculate file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256, etc.)
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file cannot be read
        ValidationError: If algorithm is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}",
            file_path=str(file_path)
        )
    
    if not file_path.is_file():
        raise ValidationError(
            f"Path is not a file: {file_path}",
            field_name="file_path",
            field_value=str(file_path)
        )
    
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValidationError(
            f"Invalid hash algorithm: {algorithm}",
            field_name="algorithm",
            field_value=algorithm
        ) from e
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
        
    except PermissionError as e:
        raise FileAccessError(
            f"Permission denied reading file: {file_path}",
            file_path=str(file_path),
            operation="read"
        ) from e
    except Exception as e:
        raise FileAccessError(
            f"Failed to read file: {file_path}",
            file_path=str(file_path),
            operation="read"
        ) from e


def is_pdf_file(file_path: Union[str, Path]) -> bool:
    """Check if file is a PDF.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is a PDF, False otherwise
    """
    file_path = Path(file_path)
    
    # Check file extension
    if file_path.suffix.lower() != '.pdf':
        return False
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type != 'application/pdf':
        return False
    
    # Check file signature (magic bytes)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file cannot be accessed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}",
            file_path=str(file_path)
        )
    
    try:
        return file_path.stat().st_size
    except PermissionError as e:
        raise FileAccessError(
            f"Permission denied accessing file: {file_path}",
            file_path=str(file_path),
            operation="stat"
        ) from e
    except Exception as e:
        raise FileAccessError(
            f"Failed to get file size: {file_path}",
            file_path=str(file_path),
            operation="stat"
        ) from e


def check_file_size(file_path: Union[str, Path], max_size: int) -> bool:
    """Check if file size is within limits.
    
    Args:
        file_path: Path to file
        max_size: Maximum allowed size in bytes
        
    Returns:
        True if file size is within limits
        
    Raises:
        FileSizeError: If file is too large
        FileNotFoundError: If file doesn't exist
        FileAccessError: If file cannot be accessed
    """
    file_size = get_file_size(file_path)
    
    if file_size > max_size:
        raise FileSizeError(
            f"File too large: {file_size} bytes (max: {max_size} bytes)",
            file_size=file_size,
            max_size=max_size
        )
    
    return True


def create_temp_file(suffix: str = ".tmp", prefix: str = "pdf_mcp_", 
                    directory: Optional[Union[str, Path]] = None) -> Tuple[int, str]:
    """Create a temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        directory: Optional directory for temp file
        
    Returns:
        Tuple of (file_descriptor, file_path)
        
    Raises:
        FileAccessError: If temp file cannot be created
    """
    try:
        if directory:
            ensure_directory(directory)
        
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=str(directory) if directory else None
        )
        
        return fd, temp_path
        
    except Exception as e:
        raise FileAccessError(
            f"Failed to create temporary file",
            operation="create_temp_file"
        ) from e


def create_temp_directory(prefix: str = "pdf_mcp_", 
                         directory: Optional[Union[str, Path]] = None) -> str:
    """Create a temporary directory.
    
    Args:
        prefix: Directory prefix
        directory: Optional parent directory
        
    Returns:
        Path to temporary directory
        
    Raises:
        FileAccessError: If temp directory cannot be created
    """
    try:
        if directory:
            ensure_directory(directory)
        
        temp_dir = tempfile.mkdtemp(
            prefix=prefix,
            dir=str(directory) if directory else None
        )
        
        return temp_dir
        
    except Exception as e:
        raise FileAccessError(
            f"Failed to create temporary directory",
            operation="create_temp_directory"
        ) from e


def safe_remove(file_path: Union[str, Path]) -> bool:
    """Safely remove a file or directory.
    
    Args:
        file_path: Path to file or directory
        
    Returns:
        True if removed successfully, False otherwise
    """
    file_path = Path(file_path)
    logger = logging.getLogger(__name__)
    
    try:
        if file_path.is_file():
            file_path.unlink()
            return True
        elif file_path.is_dir():
            shutil.rmtree(file_path)
            return True
        else:
            logger.warning(f"Path does not exist: {file_path}")
            return False
            
    except Exception as e:
        logger.warning(f"Failed to remove {file_path}: {e}")
        return False


def copy_file(src: Union[str, Path], dst: Union[str, Path], 
             preserve_metadata: bool = True) -> Path:
    """Copy a file.
    
    Args:
        src: Source file path
        dst: Destination file path
        preserve_metadata: Whether to preserve file metadata
        
    Returns:
        Path to destination file
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileAccessError: If copy operation fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {src_path}",
            file_path=str(src_path)
        )
    
    try:
        # Ensure destination directory exists
        ensure_directory(dst_path.parent)
        
        if preserve_metadata:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)
        
        return dst_path
        
    except PermissionError as e:
        raise FileAccessError(
            f"Permission denied copying file: {src_path} -> {dst_path}",
            file_path=str(src_path),
            operation="copy"
        ) from e
    except Exception as e:
        raise FileAccessError(
            f"Failed to copy file: {src_path} -> {dst_path}",
            file_path=str(src_path),
            operation="copy"
        ) from e


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Move a file.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        Path to destination file
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileAccessError: If move operation fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {src_path}",
            file_path=str(src_path)
        )
    
    try:
        # Ensure destination directory exists
        ensure_directory(dst_path.parent)
        
        shutil.move(str(src_path), str(dst_path))
        return dst_path
        
    except PermissionError as e:
        raise FileAccessError(
            f"Permission denied moving file: {src_path} -> {dst_path}",
            file_path=str(src_path),
            operation="move"
        ) from e
    except Exception as e:
        raise FileAccessError(
            f"Failed to move file: {src_path} -> {dst_path}",
            file_path=str(src_path),
            operation="move"
        ) from e


def get_available_space(directory: Union[str, Path]) -> int:
    """Get available disk space in bytes.
    
    Args:
        directory: Directory path to check
        
    Returns:
        Available space in bytes
        
    Raises:
        FileAccessError: If space cannot be determined
    """
    try:
        statvfs = os.statvfs(str(directory))
        return statvfs.f_frsize * statvfs.f_bavail
    except AttributeError:
        # Windows doesn't have statvfs
        try:
            import shutil
            _, _, free = shutil.disk_usage(str(directory))
            return free
        except Exception as e:
            raise FileAccessError(
                f"Failed to get disk space for {directory}",
                file_path=str(directory),
                operation="disk_usage"
            ) from e
    except Exception as e:
        raise FileAccessError(
            f"Failed to get disk space for {directory}",
            file_path=str(directory),
            operation="disk_usage"
        ) from e


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    import math
    
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def find_files(directory: Union[str, Path], pattern: str = "*", 
              recursive: bool = True, max_depth: Optional[int] = None) -> List[Path]:
    """Find files matching pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        max_depth: Maximum search depth
        
    Returns:
        List of matching file paths
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        FileAccessError: If directory cannot be accessed
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {dir_path}",
            file_path=str(dir_path)
        )
    
    if not dir_path.is_dir():
        raise ValidationError(
            f"Path is not a directory: {dir_path}",
            field_name="directory",
            field_value=str(dir_path)
        )
    
    try:
        if recursive:
            if max_depth is not None:
                # Limited depth search
                files = []
                for depth in range(max_depth + 1):
                    search_pattern = "/".join(["*"] * depth + [pattern])
                    files.extend(dir_path.glob(search_pattern))
                return [f for f in files if f.is_file()]
            else:
                # Unlimited depth search
                return [f for f in dir_path.rglob(pattern) if f.is_file()]
        else:
            # Non-recursive search
            return [f for f in dir_path.glob(pattern) if f.is_file()]
            
    except PermissionError as e:
        raise FileAccessError(
            f"Permission denied accessing directory: {dir_path}",
            file_path=str(dir_path),
            operation="search"
        ) from e
    except Exception as e:
        raise FileAccessError(
            f"Failed to search directory: {dir_path}",
            file_path=str(dir_path),
            operation="search"
        ) from e