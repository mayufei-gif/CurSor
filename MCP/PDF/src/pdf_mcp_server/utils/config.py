"""Configuration management for PDF-MCP server.

This module handles loading and managing configuration from environment
variables, configuration files, and default values.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for PDF-MCP server."""
    
    # Server settings
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    
    # File processing settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    temp_dir: str = "/tmp/pdf_mcp"
    cleanup_temp_files: bool = True
    
    # Text extraction settings
    text_engine: str = "pymupdf"  # pymupdf, pdfplumber
    extract_coordinates: bool = True
    clean_text: bool = True
    
    # Table extraction settings
    table_engine: str = "camelot"  # camelot, tabula, pdfplumber
    table_flavor: str = "lattice"  # lattice, stream
    table_confidence_threshold: float = 0.8
    
    # OCR settings
    ocr_engine: str = "ocrmypdf"  # ocrmypdf, tesseract
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    ocr_timeout: int = 300  # seconds
    
    # Formula recognition settings
    formula_model: str = "latex_ocr"  # latex_ocr, pix2tex, texify
    formula_confidence_threshold: float = 0.5
    min_formula_area: int = 100
    max_formula_area: int = 50000
    
    # GROBID settings
    grobid_enabled: bool = False
    grobid_url: str = "http://localhost:8070"
    grobid_timeout: int = 60
    
    # GPU settings
    use_gpu: bool = False
    gpu_device: str = "cuda:0"
    
    # Security settings
    allowed_file_types: list = field(default_factory=lambda: [".pdf"])
    max_pages: int = 1000
    enable_file_upload: bool = True
    upload_path_whitelist: list = field(default_factory=list)
    
    # Performance settings
    max_workers: int = 4
    processing_timeout: int = 600  # seconds
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    
    # MCP settings
    mcp_server_name: str = "pdf-mcp-server"
    mcp_server_version: str = "1.0.0"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables.
        
        Returns:
            Config instance with values from environment
        """
        config = cls()
        
        # Server settings
        config.host = os.getenv("PDF_MCP_HOST", config.host)
        config.port = int(os.getenv("PDF_MCP_PORT", str(config.port)))
        config.debug = os.getenv("PDF_MCP_DEBUG", "false").lower() == "true"
        config.log_level = os.getenv("PDF_MCP_LOG_LEVEL", config.log_level)
        
        # File processing settings
        config.max_file_size = int(os.getenv("PDF_MCP_MAX_FILE_SIZE", str(config.max_file_size)))
        config.temp_dir = os.getenv("PDF_MCP_TEMP_DIR", config.temp_dir)
        config.cleanup_temp_files = os.getenv("PDF_MCP_CLEANUP_TEMP", "true").lower() == "true"
        
        # Text extraction settings
        config.text_engine = os.getenv("PDF_MCP_TEXT_ENGINE", config.text_engine)
        config.extract_coordinates = os.getenv("PDF_MCP_EXTRACT_COORDS", "true").lower() == "true"
        config.clean_text = os.getenv("PDF_MCP_CLEAN_TEXT", "true").lower() == "true"
        
        # Table extraction settings
        config.table_engine = os.getenv("PDF_MCP_TABLE_ENGINE", config.table_engine)
        config.table_flavor = os.getenv("PDF_MCP_TABLE_FLAVOR", config.table_flavor)
        config.table_confidence_threshold = float(os.getenv(
            "PDF_MCP_TABLE_CONFIDENCE", str(config.table_confidence_threshold)
        ))
        
        # OCR settings
        config.ocr_engine = os.getenv("PDF_MCP_OCR_ENGINE", config.ocr_engine)
        config.ocr_language = os.getenv("PDF_MCP_OCR_LANGUAGE", config.ocr_language)
        config.ocr_dpi = int(os.getenv("PDF_MCP_OCR_DPI", str(config.ocr_dpi)))
        config.ocr_timeout = int(os.getenv("PDF_MCP_OCR_TIMEOUT", str(config.ocr_timeout)))
        
        # Formula recognition settings
        config.formula_model = os.getenv("PDF_MCP_FORMULA_MODEL", config.formula_model)
        config.formula_confidence_threshold = float(os.getenv(
            "PDF_MCP_FORMULA_CONFIDENCE", str(config.formula_confidence_threshold)
        ))
        config.min_formula_area = int(os.getenv("PDF_MCP_MIN_FORMULA_AREA", str(config.min_formula_area)))
        config.max_formula_area = int(os.getenv("PDF_MCP_MAX_FORMULA_AREA", str(config.max_formula_area)))
        
        # GROBID settings
        config.grobid_enabled = os.getenv("PDF_MCP_GROBID_ENABLED", "false").lower() == "true"
        config.grobid_url = os.getenv("PDF_MCP_GROBID_URL", config.grobid_url)
        config.grobid_timeout = int(os.getenv("PDF_MCP_GROBID_TIMEOUT", str(config.grobid_timeout)))
        
        # GPU settings
        config.use_gpu = os.getenv("PDF_MCP_USE_GPU", "false").lower() == "true"
        config.gpu_device = os.getenv("PDF_MCP_GPU_DEVICE", config.gpu_device)
        
        # Security settings
        allowed_types = os.getenv("PDF_MCP_ALLOWED_TYPES")
        if allowed_types:
            config.allowed_file_types = [t.strip() for t in allowed_types.split(",")]
        
        config.max_pages = int(os.getenv("PDF_MCP_MAX_PAGES", str(config.max_pages)))
        config.enable_file_upload = os.getenv("PDF_MCP_ENABLE_UPLOAD", "true").lower() == "true"
        
        whitelist = os.getenv("PDF_MCP_UPLOAD_WHITELIST")
        if whitelist:
            config.upload_path_whitelist = [p.strip() for p in whitelist.split(",")]
        
        # Performance settings
        config.max_workers = int(os.getenv("PDF_MCP_MAX_WORKERS", str(config.max_workers)))
        config.processing_timeout = int(os.getenv("PDF_MCP_TIMEOUT", str(config.processing_timeout)))
        config.cache_enabled = os.getenv("PDF_MCP_CACHE_ENABLED", "true").lower() == "true"
        config.cache_ttl = int(os.getenv("PDF_MCP_CACHE_TTL", str(config.cache_ttl)))
        
        # MCP settings
        config.mcp_server_name = os.getenv("PDF_MCP_SERVER_NAME", config.mcp_server_name)
        config.mcp_server_version = os.getenv("PDF_MCP_SERVER_VERSION", config.mcp_server_version)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """Create configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config instance with values from file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Start with default config
        config = cls()
        
        # Update with values from file
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> 'Config':
        """Load configuration from file and environment.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Config instance
        """
        # Start with environment variables
        config = cls.from_env()
        
        # Override with file if provided
        if config_path:
            try:
                file_config = cls.from_file(config_path)
                # Merge file config into env config
                for key, value in file_config.__dict__.items():
                    if not key.startswith('_'):
                        setattr(config, key, value)
            except Exception as e:
                logging.warning(f"Failed to load config file {config_path}: {e}")
        
        # Validate configuration
        config.validate()
        
        return config
    
    def validate(self):
        """Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate port
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Invalid port: {self.port}")
        
        # Validate file size
        if self.max_file_size <= 0:
            raise ValueError(f"Invalid max file size: {self.max_file_size}")
        
        # Validate thresholds
        if not (0.0 <= self.table_confidence_threshold <= 1.0):
            raise ValueError(f"Invalid table confidence threshold: {self.table_confidence_threshold}")
        
        if not (0.0 <= self.formula_confidence_threshold <= 1.0):
            raise ValueError(f"Invalid formula confidence threshold: {self.formula_confidence_threshold}")
        
        # Validate areas
        if self.min_formula_area <= 0:
            raise ValueError(f"Invalid min formula area: {self.min_formula_area}")
        
        if self.max_formula_area <= self.min_formula_area:
            raise ValueError(f"Max formula area must be greater than min: {self.max_formula_area} <= {self.min_formula_area}")
        
        # Validate timeouts
        if self.ocr_timeout <= 0:
            raise ValueError(f"Invalid OCR timeout: {self.ocr_timeout}")
        
        if self.processing_timeout <= 0:
            raise ValueError(f"Invalid processing timeout: {self.processing_timeout}")
        
        # Validate workers
        if self.max_workers <= 0:
            raise ValueError(f"Invalid max workers: {self.max_workers}")
        
        # Validate pages
        if self.max_pages <= 0:
            raise ValueError(f"Invalid max pages: {self.max_pages}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")
        
        # Validate engines
        valid_text_engines = ['pymupdf', 'pdfplumber']
        if self.text_engine not in valid_text_engines:
            raise ValueError(f"Invalid text engine: {self.text_engine}")
        
        valid_table_engines = ['camelot', 'tabula', 'pdfplumber']
        if self.table_engine not in valid_table_engines:
            raise ValueError(f"Invalid table engine: {self.table_engine}")
        
        valid_ocr_engines = ['ocrmypdf', 'tesseract']
        if self.ocr_engine not in valid_ocr_engines:
            raise ValueError(f"Invalid OCR engine: {self.ocr_engine}")
        
        valid_formula_models = ['latex_ocr', 'pix2tex', 'texify']
        if self.formula_model not in valid_formula_models:
            raise ValueError(f"Invalid formula model: {self.formula_model}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, config_path: Union[str, Path]):
        """Save configuration to JSON file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(host={self.host}, port={self.port}, debug={self.debug})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Config({self.to_dict()})"


def get_default_config_path() -> Path:
    """Get default configuration file path.
    
    Returns:
        Path to default configuration file
    """
    # Look for config in several locations
    possible_paths = [
        Path.cwd() / "config.json",
        Path.cwd() / "pdf_mcp_config.json",
        Path.home() / ".pdf_mcp" / "config.json",
        Path("/etc/pdf_mcp/config.json")
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Return first path as default
    return possible_paths[0]


def create_default_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Create and save default configuration file.
    
    Args:
        config_path: Optional path to save configuration
        
    Returns:
        Default configuration
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    config = Config()
    config.save(config_path)
    
    return config