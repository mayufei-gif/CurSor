"""Table extraction processor using Camelot and Tabula.

This module provides table extraction functionality with support for
multiple engines and output formats.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import json

try:
    import camelot
except ImportError:
    camelot = None

try:
    import tabula
except ImportError:
    tabula = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..models import (
    ProcessingRequest,
    TableExtractionResult,
    TableData,
    BoundingBox,
    TableEngine,
    OutputFormat,
)
from ..utils.config import Config
from ..utils.exceptions import PDFProcessingError


class TableExtractor:
    """Extracts tables from PDF documents using multiple engines."""
    
    def __init__(self, config: Config):
        """Initialize the table extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        self.available_engines = []
        if camelot:
            self.available_engines.append("camelot")
        if tabula:
            self.available_engines.append("tabula")
        if pdfplumber:
            self.available_engines.append("pdfplumber")
        
        if not self.available_engines:
            self.logger.warning("No table extraction engines available")
        
        self.logger.info(f"Table extractor initialized with engines: {self.available_engines}")
    
    async def initialize(self):
        """Initialize the extractor."""
        self.logger.info("Table extractor initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Table extractor cleanup complete")
    
    async def health_check(self) -> bool:
        """Check if the extractor is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return len(self.available_engines) > 0
    
    async def extract(self, file_path: Path, request: ProcessingRequest) -> TableExtractionResult:
        """Extract tables from PDF.
        
        Args:
            file_path: Path to PDF file
            request: Processing request with options
            
        Returns:
            Table extraction result
            
        Raises:
            PDFProcessingError: If extraction fails
        """
        self.logger.info(f"Extracting tables from: {file_path}")
        
        try:
            # Determine engine order based on request
            engines_to_try = self._get_engine_order(request.table_engine)
            
            all_tables = []
            engines_used = []
            
            # Try engines in order
            for engine in engines_to_try:
                try:
                    if engine == "camelot" and "camelot" in self.available_engines:
                        tables = await self._extract_with_camelot(file_path, request)
                        if tables:
                            all_tables.extend(tables)
                            engines_used.append("camelot")
                    
                    elif engine == "tabula" and "tabula" in self.available_engines:
                        tables = await self._extract_with_tabula(file_path, request)
                        if tables:
                            all_tables.extend(tables)
                            engines_used.append("tabula")
                    
                    elif engine == "pdfplumber" and "pdfplumber" in self.available_engines:
                        tables = await self._extract_with_pdfplumber(file_path, request)
                        if tables:
                            all_tables.extend(tables)
                            engines_used.append("pdfplumber")
                    
                except Exception as e:
                    self.logger.warning(f"Engine {engine} failed: {e}")
                    continue
            
            # Remove duplicates and sort by page/position
            unique_tables = self._deduplicate_tables(all_tables)
            return TableExtractionResult(
                tables=unique_tables,
                total_tables=len(unique_tables),
                extraction_method=engines_used[0] if engines_used else "unknown",
                fallback_used=len(set(engines_used)) > 1,
                processing_time=0.0,
            )
            
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            raise PDFProcessingError(f"Table extraction failed: {e}")
    
    def _get_engine_order(self, preferred_engine: Optional[TableEngine]) -> List[str]:
        """Get engine order based on preference.
        
        Args:
            preferred_engine: Preferred table engine
            
        Returns:
            List of engines in order of preference
        """
        if preferred_engine == TableEngine.CAMELOT:
            return ["camelot", "tabula", "pdfplumber"]
        elif preferred_engine == TableEngine.TABULA:
            return ["tabula", "camelot", "pdfplumber"]
        elif preferred_engine == TableEngine.PDFPLUMBER:
            return ["pdfplumber", "camelot", "tabula"]
        else:
            # Default order: Camelot first (best for lattice tables), then Tabula, then pdfplumber
            return ["camelot", "tabula", "pdfplumber"]
    
    async def _extract_with_camelot(self, file_path: Path, request: ProcessingRequest) -> List[TableData]:
        """Extract tables using Camelot.
        
        Args:
            file_path: Path to PDF file
            request: Processing request
            
        Returns:
            List of extracted tables
        """
        if not camelot:
            return []
        
        tables = []
        
        try:
            # Determine pages to process
            pages = "all"
            if request.pages:
                pages = ",".join(str(p + 1) for p in request.pages)  # Camelot uses 1-indexed pages
            
            # Try lattice method first (best for tables with clear borders)
            try:
                lattice_tables = camelot.read_pdf(
                    str(file_path),
                    pages=pages,
                    flavor='lattice',
                    table_areas=None,
                    columns=None,
                    split_text=True,
                    flag_size=True,
                    strip_text='\n'
                )
                
                for i, table in enumerate(lattice_tables):
                    if len(table.df) > 0 and not table.df.empty:
                        table_data = await self._convert_camelot_table(table, i, "lattice")
                        if table_data:
                            tables.append(table_data)
                            
            except Exception as e:
                self.logger.warning(f"Camelot lattice method failed: {e}")
            
            # Try stream method if lattice didn't find enough tables
            if len(tables) < 2:  # Arbitrary threshold
                try:
                    stream_tables = camelot.read_pdf(
                        str(file_path),
                        pages=pages,
                        flavor='stream',
                        table_areas=None,
                        columns=None,
                        edge_tol=500,
                        row_tol=2,
                        column_tol=0
                    )
                    
                    for i, table in enumerate(stream_tables):
                        if len(table.df) > 0 and not table.df.empty:
                            table_data = await self._convert_camelot_table(table, i + len(tables), "stream")
                            if table_data:
                                tables.append(table_data)
                                
                except Exception as e:
                    self.logger.warning(f"Camelot stream method failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Camelot extraction failed: {e}")
        
        return tables
    
    async def _convert_camelot_table(self, table, table_id: int, method: str) -> Optional[TableData]:
        """Convert Camelot table to TableData.
        
        Args:
            table: Camelot table object
            table_id: Table identifier
            method: Extraction method used
            
        Returns:
            TableData object or None
        """
        try:
            df = table.df
            
            # Skip empty tables
            if df.empty or len(df) == 0:
                return None
            
            # Convert to 2D list and headers
            headers = [str(c) for c in df.columns.tolist()]
            data = [[str(x) if x is not None else "" for x in row] for row in df.values.tolist()]
            
            # Get bounding box if available
            bbox = None
            if hasattr(table, '_bbox') and table._bbox:
                bbox = BoundingBox(
                    x0=table._bbox[0],
                    y0=table._bbox[1],
                    x1=table._bbox[2],
                    y1=table._bbox[3]
                )
            else:
                # Minimal valid bbox to satisfy validators
                bbox = BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
            
            # Camelot page may come as string; coerce to int if possible
            try:
                page_num = int(str(table.page))
            except Exception:
                page_num = 1

            return TableData(
                page=page_num,
                table_id=table_id,
                bbox=bbox,
                data=data,
                headers=headers,
                confidence=float(getattr(table, 'accuracy', 0.0) or 0.0),
                engine=f"camelot_{method}",
                rows=len(df),
                columns=len(df.columns),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to convert Camelot table: {e}")
            return None
    
    async def _extract_with_tabula(self, file_path: Path, request: ProcessingRequest) -> List[TableData]:
        """Extract tables using Tabula.
        
        Args:
            file_path: Path to PDF file
            request: Processing request
            
        Returns:
            List of extracted tables
        """
        if not tabula or not pd:
            return []
        
        tables = []
        
        try:
            # Determine pages to process
            pages = "all"
            if request.pages:
                pages = [p + 1 for p in request.pages]  # Tabula uses 1-indexed pages
            
            # Extract tables
            dfs = tabula.read_pdf(
                str(file_path),
                pages=pages,
                multiple_tables=True,
                pandas_options={'header': None},
                stream=True,  # Try stream method first
                guess=True,
                area=None,
                columns=None
            )
            
            for i, df in enumerate(dfs):
                if not df.empty and len(df) > 0:
                    table_data = await self._convert_tabula_table(df, i, "stream")
                    if table_data:
                        tables.append(table_data)
            
            # If stream method didn't work well, try lattice method
            if len(tables) == 0:
                try:
                    dfs = tabula.read_pdf(
                        str(file_path),
                        pages=pages,
                        multiple_tables=True,
                        pandas_options={'header': None},
                        lattice=True,
                        guess=True
                    )
                    
                    for i, df in enumerate(dfs):
                        if not df.empty and len(df) > 0:
                            table_data = await self._convert_tabula_table(df, i, "lattice")
                            if table_data:
                                tables.append(table_data)
                                
                except Exception as e:
                    self.logger.warning(f"Tabula lattice method failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Tabula extraction failed: {e}")
        
        return tables
    
    async def _convert_tabula_table(self, df, table_id: int, method: str) -> Optional[TableData]:
        """Convert Tabula DataFrame to TableData.
        
        Args:
            df: Pandas DataFrame
            table_id: Table identifier
            method: Extraction method used
            
        Returns:
            TableData object or None
        """
        try:
            # Skip empty tables
            if df.empty or len(df) == 0:
                return None
            
            # Clean DataFrame
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                return None
            
            headers = [str(c) for c in df.columns.tolist()]
            data = [[str(x) if x is not None else "" for x in row] for row in df.values.tolist()]
            
            return TableData(
                page=1,
                table_id=table_id,
                bbox=BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
                data=data,
                headers=headers,
                confidence=0.8,  # Default confidence for Tabula
                engine=f"tabula_{method}",
                rows=len(df),
                columns=len(df.columns),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to convert Tabula table: {e}")
            return None
    
    async def _extract_with_pdfplumber(self, file_path: Path, request: ProcessingRequest) -> List[TableData]:
        """Extract tables using pdfplumber.
        
        Args:
            file_path: Path to PDF file
            request: Processing request
            
        Returns:
            List of extracted tables
        """
        if not pdfplumber:
            return []
        
        tables = []
        
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                # Determine pages to process
                page_range = request.pages if request.pages else range(len(pdf.pages))
                
                for page_num in page_range:
                    if page_num >= len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_num]
                    page_tables = page.extract_tables()
                    
                    for i, table in enumerate(page_tables):
                        if table and len(table) > 0:
                            table_data = await self._convert_pdfplumber_table(
                                table, f"{page_num}_{i}", page_num + 1
                            )
                            if table_data:
                                tables.append(table_data)
        
        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed: {e}")
        
        return tables
    
    async def _convert_pdfplumber_table(self, table, table_id: str, page_num: int) -> Optional[TableData]:
        """Convert pdfplumber table to TableData.
        
        Args:
            table: pdfplumber table (list of lists)
            table_id: Table identifier
            page_num: Page number
            
        Returns:
            TableData object or None
        """
        try:
            if not table or len(table) == 0:
                return None
            
            # Convert to DataFrame
            if pd:
                df = pd.DataFrame(table)
                df = df.dropna(how='all').dropna(axis=1, how='all')
                if df.empty:
                    return None
                headers = [str(c) for c in df.columns.tolist()]
                data = [[str(x) if x is not None else "" for x in row] for row in df.values.tolist()]
                rows = len(df)
                cols = len(df.columns)
            else:
                # Fallback without pandas
                data = [[str(cell) if cell else "" for cell in row] for row in table]
                rows = len(data)
                cols = len(data[0]) if data else 0
                headers = None

            return TableData(
                page=page_num,
                table_id=int(''.join(filter(str.isdigit, str(table_id))) or 0),
                bbox=BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
                data=data,
                headers=headers,
                confidence=0.7,
                engine="pdfplumber",
                rows=rows,
                columns=cols,
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to convert pdfplumber table: {e}")
            return None
    
    def _deduplicate_tables(self, tables: List[TableData]) -> List[TableData]:
        """Remove duplicate tables.
        
        Args:
            tables: List of tables
            
        Returns:
            List of unique tables
        """
        if not tables:
            return []
        
        unique_tables = []
        seen_hashes = set()
        
        for table in tables:
            # Create a simple hash based on table content
            content = getattr(table, "data", None)
            content_hash = hash(str(content)) if content is not None else hash(str(table))

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_tables.append(table)
        
        # Sort by page number and position
        unique_tables.sort(key=lambda t: (getattr(t, 'page', 1), getattr(t, 'table_id', 0)))
        
        return unique_tables
