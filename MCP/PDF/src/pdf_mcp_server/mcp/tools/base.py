#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base PDF Tool Class

This module provides the base class for all PDF processing tools.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

class PDFToolInput(BaseModel):
    """Base input schema for PDF tools."""
    model_config = ConfigDict(extra="forbid")
    
    file_path: str = Field(
        description="Path to the PDF file to process",
        examples=["/path/to/document.pdf"]
    )

class PDFToolOutput(BaseModel):
    """Base output schema for PDF tools."""
    model_config = ConfigDict(extra="allow")
    
    success: bool = Field(
        description="Whether the operation was successful"
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message or error description"
    )
    file_path: str = Field(
        description="Path to the processed PDF file"
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Time taken to process the file in seconds"
    )

class PDFTool(ABC):
    """Base class for all PDF processing tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Return the input schema for this tool."""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Return the output schema for this tool."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given input data.
        
        Args:
            input_data: Input parameters for the tool
            
        Returns:
            Tool execution results
            
        Raises:
            Exception: If tool execution fails
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> BaseModel:
        """Validate input data against the tool's schema.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Validated input model
            
        Raises:
            ValidationError: If input is invalid
        """
        return self.input_schema(**input_data)
    
    def validate_output(self, output_data: Dict[str, Any]) -> BaseModel:
        """Validate output data against the tool's schema.
        
        Args:
            output_data: Raw output data
            
        Returns:
            Validated output model
            
        Raises:
            ValidationError: If output is invalid
        """
        return self.output_schema(**output_data)
    
    def validate_file_path(self, file_path: str) -> Path:
        """Validate that the file path exists and is a PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Validated Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        return path
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool.
        
        Returns:
            Tool information dictionary
        """
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema.model_json_schema(),
            'output_schema': self.output_schema.model_json_schema(),
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()