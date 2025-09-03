#!/usr/bin/env python3
"""
PDF Table Extraction Tools

Implements table extraction functionality for PDF files using multiple libraries
including Camelot, Tabula, and pdfplumber for comprehensive table detection and extraction.

Author: PDF-MCP Team
License: MIT
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import camelot
except ImportError:
    camelot = None

try:
    import tabula
except ImportError:
    tabula = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..mcp.tools import PDFTool
from ..mcp.protocol import MCPToolResult, create_text_content, create_error_content
from ..mcp.exceptions import ToolExecutionException, MCPResourceException


class ExtractTablesTool(PDFTool):
    """Extract tables from PDF files using multiple detection methods."""
    
    def __init__(self):
        super().__init__(
            name="extract_tables",
            description="Extract tables from PDF files using multiple detection methods",
            version="1.0.0"
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "method": {
                    "type": "string",
                    "enum": ["auto", "camelot", "tabula", "pdfplumber"],
                    "default": "auto",
                    "description": "Table extraction method to use"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to extract tables from (1-indexed). If not provided, processes all pages"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv", "markdown", "html"],
                    "default": "json",
                    "description": "Output format for extracted tables"
                },
                "table_areas": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "description": "Specific table areas to extract [x1, y1, x2, y2] in points"
                },
                "flavor": {
                    "type": "string",
                    "enum": ["lattice", "stream"],
                    "default": "lattice",
                    "description": "Table detection flavor for Camelot (lattice for tables with borders, stream for tables without borders)"
                },
                "edge_tol": {
                    "type": "number",
                    "default": 50,
                    "description": "Edge tolerance for table detection"
                },
                "row_tol": {
                    "type": "number",
                    "default": 2,
                    "description": "Row tolerance for table detection"
                },
                "column_tol": {
                    "type": "number",
                    "default": 0,
                    "description": "Column tolerance for table detection"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        method = kwargs.get("method", "auto")
        pages = kwargs.get("pages")
        output_format = kwargs.get("output_format", "json")
        table_areas = kwargs.get("table_areas")
        flavor = kwargs.get("flavor", "lattice")
        edge_tol = kwargs.get("edge_tol", 50)
        row_tol = kwargs.get("row_tol", 2)
        column_tol = kwargs.get("column_tol", 0)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Choose extraction method
            if method == "auto":
                method = self._choose_best_method()
            
            # Extract tables based on method
            if method == "camelot":
                result = await self._extract_with_camelot(
                    pdf_path, pages, table_areas, flavor, edge_tol, row_tol, column_tol
                )
            elif method == "tabula":
                result = await self._extract_with_tabula(
                    pdf_path, pages, table_areas
                )
            elif method == "pdfplumber":
                result = await self._extract_with_pdfplumber(
                    pdf_path, pages
                )
            else:
                raise ToolExecutionException(f"Unknown extraction method: {method}")
            
            # Format output
            content = []
            
            # Add summary
            summary = {
                "file": str(pdf_path),
                "method": method,
                "total_tables": len(result["tables"]),
                "pages_processed": result.get("pages_processed", []),
                "extraction_time": result.get("extraction_time", 0)
            }
            
            content.append(create_text_content(f"Table Extraction Summary:\n{json.dumps(summary, indent=2)}"))
            
            # Add extracted tables
            for i, table_info in enumerate(result["tables"]):
                table_data = table_info["data"]
                page_num = table_info.get("page", "unknown")
                confidence = table_info.get("confidence", "unknown")
                
                # Format table based on output format
                if output_format == "json":
                    formatted_table = json.dumps(table_data, indent=2, ensure_ascii=False)
                    content.append(create_text_content(
                        f"Table {i+1} (Page {page_num}, Confidence: {confidence}):\n```json\n{formatted_table}\n```"
                    ))
                elif output_format == "csv":
                    if table_data:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
                        csv_str = df.to_csv(index=False)
                        content.append(create_text_content(
                            f"Table {i+1} (Page {page_num}, Confidence: {confidence}):\n```csv\n{csv_str}\n```"
                        ))
                elif output_format == "markdown":
                    markdown_table = self._format_as_markdown(table_data)
                    content.append(create_text_content(
                        f"Table {i+1} (Page {page_num}, Confidence: {confidence}):\n{markdown_table}"
                    ))
                elif output_format == "html":
                    if table_data:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
                        html_str = df.to_html(index=False)
                        content.append(create_text_content(
                            f"Table {i+1} (Page {page_num}, Confidence: {confidence}):\n```html\n{html_str}\n```"
                        ))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            content = [create_error_content(f"Table extraction failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _choose_best_method(self) -> str:
        """Choose the best available extraction method."""
        if camelot:
            return "camelot"
        elif tabula:
            return "tabula"
        elif pdfplumber:
            return "pdfplumber"
        else:
            raise ToolExecutionException("No table extraction library available")
    
    async def _extract_with_camelot(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        table_areas: Optional[List[List[float]]],
        flavor: str,
        edge_tol: float,
        row_tol: float,
        column_tol: float
    ) -> Dict[str, Any]:
        """Extract tables using Camelot."""
        if not camelot:
            raise ToolExecutionException("Camelot not available")
        
        start_time = datetime.now()
        
        # Prepare parameters
        kwargs = {
            "flavor": flavor,
            "edge_tol": edge_tol,
            "row_tol": row_tol,
            "column_tol": column_tol
        }
        
        if pages:
            kwargs["pages"] = ",".join(map(str, pages))
        else:
            kwargs["pages"] = "all"
        
        if table_areas:
            kwargs["table_areas"] = table_areas
        
        # Extract tables
        tables = camelot.read_pdf(str(pdf_path), **kwargs)
        
        extracted_tables = []
        pages_processed = []
        
        for table in tables:
            # Convert to list format
            table_data = table.df.values.tolist()
            # Add headers
            headers = table.df.columns.tolist()
            if headers and headers != list(range(len(headers))):
                table_data.insert(0, headers)
            
            table_info = {
                "data": table_data,
                "page": table.page,
                "confidence": round(table.accuracy, 2) if hasattr(table, 'accuracy') else "unknown",
                "shape": table.shape,
                "parsing_report": table.parsing_report if hasattr(table, 'parsing_report') else None
            }
            
            extracted_tables.append(table_info)
            if table.page not in pages_processed:
                pages_processed.append(table.page)
        
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        return {
            "tables": extracted_tables,
            "pages_processed": sorted(pages_processed),
            "extraction_time": extraction_time
        }
    
    async def _extract_with_tabula(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
        table_areas: Optional[List[List[float]]]
    ) -> Dict[str, Any]:
        """Extract tables using Tabula."""
        if not tabula:
            raise ToolExecutionException("Tabula not available")
        
        start_time = datetime.now()
        
        # Prepare parameters
        kwargs = {
            "pages": pages if pages else "all",
            "multiple_tables": True,
            "pandas_options": {"header": None}
        }
        
        if table_areas:
            kwargs["area"] = table_areas
        
        # Extract tables
        try:
            dfs = tabula.read_pdf(str(pdf_path), **kwargs)
        except Exception as e:
            # Fallback without area specification
            if table_areas:
                kwargs.pop("area", None)
                dfs = tabula.read_pdf(str(pdf_path), **kwargs)
            else:
                raise e
        
        extracted_tables = []
        pages_processed = pages if pages else [1]  # Tabula doesn't provide page info easily
        
        for i, df in enumerate(dfs):
            if not df.empty:
                # Convert to list format
                table_data = df.values.tolist()
                
                # Clean up NaN values
                table_data = [[str(cell) if pd.notna(cell) else "" for cell in row] for row in table_data]
                
                table_info = {
                    "data": table_data,
                    "page": pages[i] if pages and i < len(pages) else "unknown",
                    "confidence": "unknown",
                    "shape": df.shape
                }
                
                extracted_tables.append(table_info)
        
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        return {
            "tables": extracted_tables,
            "pages_processed": pages_processed,
            "extraction_time": extraction_time
        }
    
    async def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        pages: Optional[List[int]]
    ) -> Dict[str, Any]:
        """Extract tables using pdfplumber."""
        if not pdfplumber:
            raise ToolExecutionException("pdfplumber not available")
        
        start_time = datetime.now()
        
        extracted_tables = []
        pages_processed = []
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            
            # Determine pages to process
            if pages:
                page_numbers = [p - 1 for p in pages if 1 <= p <= total_pages]  # Convert to 0-indexed
            else:
                page_numbers = list(range(total_pages))
            
            for page_num in page_numbers:
                page = pdf.pages[page_num]
                tables = page.extract_tables()
                
                for table in tables:
                    if table:  # Skip empty tables
                        table_info = {
                            "data": table,
                            "page": page_num + 1,  # Convert back to 1-indexed
                            "confidence": "unknown",
                            "shape": (len(table), len(table[0]) if table else 0)
                        }
                        
                        extracted_tables.append(table_info)
                
                if tables:  # Only add to processed if tables were found
                    pages_processed.append(page_num + 1)
        
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        return {
            "tables": extracted_tables,
            "pages_processed": sorted(pages_processed),
            "extraction_time": extraction_time
        }
    
    def _format_as_markdown(self, table_data: List[List[str]]) -> str:
        """Format table data as markdown."""
        if not table_data:
            return "Empty table"
        
        lines = []
        
        # Header row
        if table_data:
            header = table_data[0]
            lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
            lines.append("|" + "---|" * len(header))
            
            # Data rows
            for row in table_data[1:]:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines)


class ExtractTablesAdvancedTool(PDFTool):
    """Advanced table extraction with custom formatting and filtering."""
    
    def __init__(self):
        super().__init__(
            name="extract_tables_advanced",
            description="Advanced table extraction with custom formatting and filtering options",
            version="1.0.0"
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "method": {
                    "type": "string",
                    "enum": ["auto", "camelot", "tabula", "pdfplumber"],
                    "default": "auto",
                    "description": "Table extraction method to use"
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific pages to extract tables from (1-indexed)"
                },
                "min_rows": {
                    "type": "integer",
                    "default": 2,
                    "description": "Minimum number of rows for a valid table"
                },
                "min_cols": {
                    "type": "integer",
                    "default": 2,
                    "description": "Minimum number of columns for a valid table"
                },
                "clean_data": {
                    "type": "boolean",
                    "default": True,
                    "description": "Clean and normalize table data"
                },
                "merge_cells": {
                    "type": "boolean",
                    "default": True,
                    "description": "Attempt to merge split cells"
                },
                "detect_headers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Automatically detect table headers"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv", "excel", "markdown"],
                    "default": "json",
                    "description": "Output format for extracted tables"
                },
                "save_to_file": {
                    "type": "boolean",
                    "default": False,
                    "description": "Save extracted tables to separate files"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> MCPToolResult:
        file_path = kwargs["file_path"]
        method = kwargs.get("method", "auto")
        pages = kwargs.get("pages")
        min_rows = kwargs.get("min_rows", 2)
        min_cols = kwargs.get("min_cols", 2)
        clean_data = kwargs.get("clean_data", True)
        merge_cells = kwargs.get("merge_cells", True)
        detect_headers = kwargs.get("detect_headers", True)
        output_format = kwargs.get("output_format", "json")
        save_to_file = kwargs.get("save_to_file", False)
        
        try:
            # Validate file
            pdf_path = self.validate_file_path(file_path)
            
            # Use the basic extraction tool first
            basic_tool = ExtractTablesTool()
            basic_result = await basic_tool.execute(
                file_path=file_path,
                method=method,
                pages=pages,
                output_format="json"
            )
            
            # Extract tables from the basic result
            if basic_result.isError:
                return basic_result
            
            # Parse the basic result to get table data
            tables_data = self._parse_basic_result(basic_result)
            
            # Apply advanced processing
            processed_tables = []
            for table_info in tables_data:
                table_data = table_info["data"]
                
                # Filter by size
                if len(table_data) < min_rows or (table_data and len(table_data[0]) < min_cols):
                    continue
                
                # Clean data
                if clean_data:
                    table_data = self._clean_table_data(table_data)
                
                # Merge cells
                if merge_cells:
                    table_data = self._merge_split_cells(table_data)
                
                # Detect headers
                if detect_headers:
                    table_data = self._detect_and_format_headers(table_data)
                
                processed_table = {
                    **table_info,
                    "data": table_data,
                    "processed": True
                }
                
                processed_tables.append(processed_table)
            
            # Save to files if requested
            saved_files = []
            if save_to_file:
                saved_files = await self._save_tables_to_files(
                    processed_tables, pdf_path, output_format
                )
            
            # Format output
            content = []
            
            # Add summary
            summary = {
                "file": str(pdf_path),
                "method": method,
                "original_tables": len(tables_data),
                "processed_tables": len(processed_tables),
                "filters_applied": {
                    "min_rows": min_rows,
                    "min_cols": min_cols,
                    "clean_data": clean_data,
                    "merge_cells": merge_cells,
                    "detect_headers": detect_headers
                },
                "saved_files": saved_files
            }
            
            content.append(create_text_content(f"Advanced Table Extraction Summary:\n{json.dumps(summary, indent=2)}"))
            
            # Add processed tables
            for i, table_info in enumerate(processed_tables):
                table_data = table_info["data"]
                page_num = table_info.get("page", "unknown")
                
                if output_format == "json":
                    formatted_table = json.dumps(table_data, indent=2, ensure_ascii=False)
                    content.append(create_text_content(
                        f"Processed Table {i+1} (Page {page_num}):\n```json\n{formatted_table}\n```"
                    ))
                elif output_format == "markdown":
                    markdown_table = self._format_as_markdown(table_data)
                    content.append(create_text_content(
                        f"Processed Table {i+1} (Page {page_num}):\n{markdown_table}"
                    ))
                else:
                    # For CSV and Excel, show first few rows
                    preview = table_data[:5] if len(table_data) > 5 else table_data
                    content.append(create_text_content(
                        f"Processed Table {i+1} (Page {page_num}) - Preview:\n{json.dumps(preview, indent=2)}"
                    ))
            
            return MCPToolResult(content=content)
        
        except Exception as e:
            self.logger.error(f"Advanced table extraction failed: {e}")
            content = [create_error_content(f"Advanced table extraction failed: {str(e)}")]
            return MCPToolResult(content=content, isError=True)
    
    def _parse_basic_result(self, result: MCPToolResult) -> List[Dict[str, Any]]:
        """Parse the basic extraction result to get table data."""
        tables = []
        
        for content_item in result.content:
            if content_item.get("type") == "text":
                text = content_item.get("text", "")
                
                # Look for JSON table data
                if "```json" in text and "Table" in text:
                    try:
                        # Extract JSON from the text
                        start = text.find("```json") + 7
                        end = text.find("```", start)
                        if start > 6 and end > start:
                            json_str = text[start:end].strip()
                            table_data = json.loads(json_str)
                            
                            # Extract page number from text
                            page_match = text.find("Page ")
                            page_num = "unknown"
                            if page_match >= 0:
                                page_text = text[page_match:page_match + 20]
                                import re
                                page_nums = re.findall(r'Page (\d+)', page_text)
                                if page_nums:
                                    page_num = int(page_nums[0])
                            
                            tables.append({
                                "data": table_data,
                                "page": page_num
                            })
                    except:
                        continue
        
        return tables
    
    def _clean_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        """Clean and normalize table data."""
        cleaned = []
        
        for row in table_data:
            cleaned_row = []
            for cell in row:
                # Convert to string and clean
                cell_str = str(cell).strip()
                
                # Remove extra whitespace
                cell_str = " ".join(cell_str.split())
                
                # Handle empty cells
                if not cell_str or cell_str.lower() in ['nan', 'none', 'null']:
                    cell_str = ""
                
                cleaned_row.append(cell_str)
            
            # Skip completely empty rows
            if any(cell for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned
    
    def _merge_split_cells(self, table_data: List[List[str]]) -> List[List[str]]:
        """Attempt to merge cells that were split across rows."""
        if len(table_data) < 2:
            return table_data
        
        merged = [table_data[0]]  # Keep first row as is
        
        for i in range(1, len(table_data)):
            current_row = table_data[i]
            previous_row = merged[-1]
            
            # Check if this row might be a continuation
            if self._is_continuation_row(current_row, previous_row):
                # Merge with previous row
                for j in range(min(len(current_row), len(previous_row))):
                    if current_row[j] and not previous_row[j]:
                        merged[-1][j] = current_row[j]
                    elif current_row[j] and previous_row[j]:
                        merged[-1][j] += " " + current_row[j]
            else:
                merged.append(current_row)
        
        return merged
    
    def _is_continuation_row(self, current_row: List[str], previous_row: List[str]) -> bool:
        """Check if current row is a continuation of the previous row."""
        if len(current_row) != len(previous_row):
            return False
        
        # Count non-empty cells
        current_non_empty = sum(1 for cell in current_row if cell.strip())
        previous_non_empty = sum(1 for cell in previous_row if cell.strip())
        
        # If current row has very few non-empty cells, it might be a continuation
        if current_non_empty <= len(current_row) // 2:
            return True
        
        return False
    
    def _detect_and_format_headers(self, table_data: List[List[str]]) -> List[List[str]]:
        """Detect and format table headers."""
        if len(table_data) < 2:
            return table_data
        
        # Check if first row looks like headers
        first_row = table_data[0]
        second_row = table_data[1] if len(table_data) > 1 else []
        
        # Headers typically have different characteristics than data
        is_header = self._looks_like_header_row(first_row, second_row)
        
        if is_header:
            # Format headers (capitalize, clean up)
            formatted_headers = []
            for header in first_row:
                header_str = str(header).strip()
                if header_str:
                    # Capitalize first letter of each word
                    header_str = " ".join(word.capitalize() for word in header_str.split())
                formatted_headers.append(header_str)
            
            return [formatted_headers] + table_data[1:]
        
        return table_data
    
    def _looks_like_header_row(self, first_row: List[str], second_row: List[str]) -> bool:
        """Check if the first row looks like a header row."""
        if not first_row:
            return False
        
        # Check for common header characteristics
        first_row_text = [str(cell).strip() for cell in first_row]
        
        # Headers usually don't contain only numbers
        numeric_cells = sum(1 for cell in first_row_text if cell.replace('.', '').replace(',', '').isdigit())
        if numeric_cells == len(first_row_text):
            return False
        
        # Headers are usually shorter than data rows
        if second_row:
            avg_first_length = sum(len(cell) for cell in first_row_text) / len(first_row_text)
            avg_second_length = sum(len(str(cell)) for cell in second_row) / len(second_row)
            
            if avg_first_length > avg_second_length * 2:
                return False
        
        return True
    
    async def _save_tables_to_files(
        self,
        tables: List[Dict[str, Any]],
        pdf_path: Path,
        output_format: str
    ) -> List[str]:
        """Save extracted tables to separate files."""
        saved_files = []
        base_name = pdf_path.stem
        output_dir = pdf_path.parent / f"{base_name}_tables"
        output_dir.mkdir(exist_ok=True)
        
        for i, table_info in enumerate(tables):
            table_data = table_info["data"]
            page_num = table_info.get("page", "unknown")
            
            if output_format == "csv":
                file_path = output_dir / f"table_{i+1}_page_{page_num}.csv"
                df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
                df.to_csv(file_path, index=False)
                saved_files.append(str(file_path))
            
            elif output_format == "excel":
                file_path = output_dir / f"table_{i+1}_page_{page_num}.xlsx"
                df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
                df.to_excel(file_path, index=False)
                saved_files.append(str(file_path))
            
            elif output_format == "json":
                file_path = output_dir / f"table_{i+1}_page_{page_num}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(table_data, f, indent=2, ensure_ascii=False)
                saved_files.append(str(file_path))
            
            elif output_format == "markdown":
                file_path = output_dir / f"table_{i+1}_page_{page_num}.md"
                markdown_content = self._format_as_markdown(table_data)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                saved_files.append(str(file_path))
        
        return saved_files
    
    def _format_as_markdown(self, table_data: List[List[str]]) -> str:
        """Format table data as markdown."""
        if not table_data:
            return "Empty table"
        
        lines = []
        
        # Header row
        if table_data:
            header = table_data[0]
            lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
            lines.append("|" + "---|" * len(header))
            
            # Data rows
            for row in table_data[1:]:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines)