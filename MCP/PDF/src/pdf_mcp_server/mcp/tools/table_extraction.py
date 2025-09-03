#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table Extraction Tools

This module provides tools for extracting tables from PDF files using various methods.
"""

import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

from .base import PDFTool, PDFToolOutput

class ExtractTablesInput(BaseModel):
    """Input schema for table extraction."""
    model_config = ConfigDict(extra="forbid")
    
    file_path: str = Field(
        description="Path to the PDF file",
        examples=["/path/to/document.pdf"]
    )
    method: str = Field(
        default="camelot",
        description="Extraction method: 'camelot', 'tabula', or 'pdfplumber'",
        examples=["camelot", "tabula", "pdfplumber"]
    )
    pages: Optional[str] = Field(
        default=None,
        description="Page range (e.g., '1-5', '1,3,5', 'all')",
        examples=["1-5", "1,3,5", "all"]
    )
    output_format: str = Field(
        default="json",
        description="Output format: 'json', 'csv', 'markdown', 'html'",
        examples=["json", "csv", "markdown", "html"]
    )
    table_areas: Optional[List[List[float]]] = Field(
        default=None,
        description="Table areas as [x1, y1, x2, y2] coordinates",
        examples=[[[100, 100, 500, 300]]]
    )
    flavor: str = Field(
        default="lattice",
        description="Camelot flavor: 'lattice' or 'stream'",
        examples=["lattice", "stream"]
    )
    edge_tol: int = Field(
        default=50,
        description="Tolerance for edge detection (Camelot)",
        ge=0,
        le=500
    )
    row_tol: int = Field(
        default=2,
        description="Tolerance for row detection (Camelot)",
        ge=0,
        le=50
    )
    column_tol: int = Field(
        default=0,
        description="Tolerance for column detection (Camelot)",
        ge=0,
        le=50
    )

class ExtractTablesOutput(PDFToolOutput):
    """Output schema for table extraction."""
    
    tables: List[Dict[str, Any]] = Field(
        description="Extracted tables data"
    )
    table_count: int = Field(
        description="Number of tables extracted"
    )
    pages_processed: List[int] = Field(
        description="List of page numbers that were processed"
    )
    method_used: str = Field(
        description="Extraction method that was used"
    )
    extraction_stats: Dict[str, Any] = Field(
        description="Extraction statistics"
    )

class ExtractTablesTool(PDFTool):
    """Table extraction tool."""
    
    def __init__(self):
        super().__init__(
            name="extract_tables",
            description="Extract tables from PDF files using Camelot, Tabula, or pdfplumber"
        )
    
    @property
    def input_schema(self) -> type[BaseModel]:
        return ExtractTablesInput
    
    @property
    def output_schema(self) -> type[BaseModel]:
        return ExtractTablesOutput
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute table extraction."""
        start_time = time.time()
        
        # Validate input
        validated_input = self.validate_input(input_data)
        file_path = self.validate_file_path(validated_input.file_path)
        
        try:
            # Extract tables based on method
            if validated_input.method == "camelot":
                tables_data, stats = await self._extract_with_camelot(file_path, validated_input)
            elif validated_input.method == "tabula":
                tables_data, stats = await self._extract_with_tabula(file_path, validated_input)
            elif validated_input.method == "pdfplumber":
                tables_data, stats = await self._extract_with_pdfplumber(file_path, validated_input)
            else:
                raise ValueError(f"Unsupported method: {validated_input.method}")
            
            # Format output based on requested format
            formatted_tables = self._format_tables(tables_data, validated_input.output_format)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "message": "Table extraction completed successfully",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "tables": formatted_tables,
                "table_count": len(formatted_tables),
                "pages_processed": stats.get("pages_processed", []),
                "method_used": validated_input.method,
                "extraction_stats": stats
            }
            
            return self.validate_output(result).model_dump()
            
        except Exception as e:
            self.logger.error(f"Table extraction failed: {str(e)}")
            processing_time = time.time() - start_time
            
            result = {
                "success": False,
                "message": f"Table extraction failed: {str(e)}",
                "file_path": str(file_path),
                "processing_time": processing_time,
                "tables": [],
                "table_count": 0,
                "pages_processed": [],
                "method_used": validated_input.method,
                "extraction_stats": {}
            }
            
            return self.validate_output(result).model_dump()
    
    async def _extract_with_camelot(self, file_path: Path, config: ExtractTablesInput) -> tuple[List[Dict], Dict]:
        """Extract tables using Camelot."""
        try:
            import camelot
        except ImportError:
            raise ImportError("Camelot is not installed. Install with: pip install camelot-py[cv]")
        
        # Prepare parameters
        kwargs = {
            'flavor': config.flavor,
            'edge_tol': config.edge_tol,
            'row_tol': config.row_tol,
            'column_tol': config.column_tol
        }
        
        if config.table_areas:
            kwargs['table_areas'] = config.table_areas
        
        # Extract tables
        pages_str = self._format_pages_for_camelot(config.pages)
        tables = camelot.read_pdf(str(file_path), pages=pages_str, **kwargs)
        
        # Convert to our format
        tables_data = []
        pages_processed = set()
        
        for i, table in enumerate(tables):
            table_dict = {
                "table_id": i,
                "page": table.page,
                "data": table.df.to_dict('records'),
                "shape": table.shape,
                "accuracy": getattr(table, 'accuracy', None),
                "whitespace": getattr(table, 'whitespace', None)
            }
            tables_data.append(table_dict)
            pages_processed.add(table.page)
        
        stats = {
            "pages_processed": sorted(list(pages_processed)),
            "total_tables": len(tables_data),
            "parsing_report": getattr(tables, 'parsing_report', None)
        }
        
        return tables_data, stats
    
    async def _extract_with_tabula(self, file_path: Path, config: ExtractTablesInput) -> tuple[List[Dict], Dict]:
        """Extract tables using Tabula."""
        try:
            import tabula
        except ImportError:
            raise ImportError("Tabula is not installed. Install with: pip install tabula-py")
        
        # Prepare parameters
        kwargs = {}
        if config.pages:
            kwargs['pages'] = self._format_pages_for_tabula(config.pages)
        if config.table_areas:
            kwargs['area'] = config.table_areas
        
        # Extract tables
        dfs = tabula.read_pdf(str(file_path), **kwargs)
        
        # Convert to our format
        tables_data = []
        for i, df in enumerate(dfs):
            table_dict = {
                "table_id": i,
                "page": i + 1,  # Tabula doesn't provide page info directly
                "data": df.to_dict('records'),
                "shape": df.shape,
                "columns": list(df.columns)
            }
            tables_data.append(table_dict)
        
        stats = {
            "pages_processed": list(range(1, len(dfs) + 1)),
            "total_tables": len(tables_data)
        }
        
        return tables_data, stats
    
    async def _extract_with_pdfplumber(self, file_path: Path, config: ExtractTablesInput) -> tuple[List[Dict], Dict]:
        """Extract tables using pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is not installed. Install with: pip install pdfplumber")
        
        tables_data = []
        pages_processed = []
        
        with pdfplumber.open(str(file_path)) as pdf:
            page_numbers = self._parse_page_range(config.pages, len(pdf.pages))
            
            table_id = 0
            for page_num in page_numbers:
                page = pdf.pages[page_num - 1]
                
                # Extract tables from page
                page_tables = page.extract_tables()
                
                for table in page_tables:
                    if table:  # Skip empty tables
                        # Convert to DataFrame-like structure
                        headers = table[0] if table else []
                        rows = table[1:] if len(table) > 1 else []
                        
                        # Create records
                        records = []
                        for row in rows:
                            record = {}
                            for i, cell in enumerate(row):
                                col_name = headers[i] if i < len(headers) else f"Column_{i}"
                                record[col_name] = cell
                            records.append(record)
                        
                        table_dict = {
                            "table_id": table_id,
                            "page": page_num,
                            "data": records,
                            "shape": (len(rows), len(headers)),
                            "headers": headers
                        }
                        tables_data.append(table_dict)
                        table_id += 1
                
                if page_tables:
                    pages_processed.append(page_num)
        
        stats = {
            "pages_processed": pages_processed,
            "total_tables": len(tables_data)
        }
        
        return tables_data, stats
    
    def _format_tables(self, tables_data: List[Dict], output_format: str) -> List[Dict]:
        """Format tables according to requested output format."""
        if output_format == "json":
            return tables_data
        
        formatted_tables = []
        for table in tables_data:
            if output_format == "csv":
                # Convert to CSV string
                import io
                import csv
                
                output = io.StringIO()
                if table["data"]:
                    fieldnames = table["data"][0].keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(table["data"])
                
                formatted_table = {
                    **table,
                    "csv_data": output.getvalue()
                }
                formatted_tables.append(formatted_table)
            
            elif output_format == "markdown":
                # Convert to Markdown table
                md_table = self._to_markdown_table(table["data"])
                formatted_table = {
                    **table,
                    "markdown_data": md_table
                }
                formatted_tables.append(formatted_table)
            
            elif output_format == "html":
                # Convert to HTML table
                html_table = self._to_html_table(table["data"])
                formatted_table = {
                    **table,
                    "html_data": html_table
                }
                formatted_tables.append(formatted_table)
            
            else:
                formatted_tables.append(table)
        
        return formatted_tables
    
    def _to_markdown_table(self, data: List[Dict]) -> str:
        """Convert table data to Markdown format."""
        if not data:
            return ""
        
        headers = list(data[0].keys())
        
        # Create header row
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        # Create data rows
        data_rows = []
        for row in data:
            row_values = [str(row.get(header, "")) for header in headers]
            data_row = "| " + " | ".join(row_values) + " |"
            data_rows.append(data_row)
        
        return "\n".join([header_row, separator_row] + data_rows)
    
    def _to_html_table(self, data: List[Dict]) -> str:
        """Convert table data to HTML format."""
        if not data:
            return "<table></table>"
        
        headers = list(data[0].keys())
        
        # Create HTML table
        html_parts = ["<table>"]
        
        # Header
        html_parts.append("  <thead>")
        html_parts.append("    <tr>")
        for header in headers:
            html_parts.append(f"      <th>{header}</th>")
        html_parts.append("    </tr>")
        html_parts.append("  </thead>")
        
        # Body
        html_parts.append("  <tbody>")
        for row in data:
            html_parts.append("    <tr>")
            for header in headers:
                value = str(row.get(header, ""))
                html_parts.append(f"      <td>{value}</td>")
            html_parts.append("    </tr>")
        html_parts.append("  </tbody>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    def _format_pages_for_camelot(self, pages: Optional[str]) -> str:
        """Format page range for Camelot."""
        if not pages or pages.lower() == "all":
            return "all"
        return pages
    
    def _format_pages_for_tabula(self, pages: Optional[str]) -> str:
        """Format page range for Tabula."""
        if not pages or pages.lower() == "all":
            return "all"
        return pages
    
    def _parse_page_range(self, pages: Optional[str], total_pages: int) -> List[int]:
        """Parse page range specification."""
        if not pages or pages.lower() == "all":
            return list(range(1, total_pages + 1))
        
        page_numbers = []
        
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                page_numbers.extend(range(start, end + 1))
            else:
                page_numbers.append(int(part))
        
        # Filter valid page numbers
        return [p for p in page_numbers if 1 <= p <= total_pages]