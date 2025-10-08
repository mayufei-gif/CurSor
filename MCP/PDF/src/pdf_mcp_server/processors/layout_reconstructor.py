"""Layout reconstruction processor.

Rebuilds a near-original representation of a PDF page layout using
previous extraction outputs (text, tables, formulas) and the source PDF
geometry.
"""

from __future__ import annotations

import html
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    fitz = None

from ..models import (
    BoundingBox,
    LayoutElement,
    LayoutElementType,
    LayoutReconstructionResult,
    ProcessingContent,
    ProcessingRequest,
    ReconstructedPage,
    TableData,
)
from ..utils.config import Config
from ..utils.exceptions import PDFProcessingError


class LayoutReconstructor:
    """Reconstruct PDF layout using extracted content and document geometry."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = False

    async def initialize(self) -> None:
        """Prepare the reconstructor for use."""
        if fitz is None:
            self.logger.warning("PyMuPDF not available; layout reconstruction limited")
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup internal state."""
        self._initialized = False

    async def health_check(self) -> bool:
        """Return True when the reconstructor can operate."""
        return fitz is not None

    async def reconstruct(
        self,
        file_path: Path,
        request: ProcessingRequest,
        content: ProcessingContent,
    ) -> LayoutReconstructionResult:
        """Rebuild layout for the requested pages."""

        if fitz is None:
            raise PDFProcessingError("PyMuPDF is required for layout reconstruction")

        start = time.time()
        notes: List[str] = []

        doc = fitz.open(str(file_path))
        try:
            pages = self._resolve_pages(request.pages, len(doc))
            reconstructed_pages: List[ReconstructedPage] = []

            for page_index in pages:
                page = doc[page_index]
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)

                text_blocks = self._extract_text_blocks(page)
                recognized_blocks = self._recognized_blocks_for_page(content, page_index + 1)

                if recognized_blocks and len(recognized_blocks) < len(text_blocks):
                    notes.append(
                        f"Page {page_index + 1}: only {len(recognized_blocks)} recognized text blocks for "
                        f"{len(text_blocks)} layout regions; falling back to source text for the remainder."
                    )
                elif not recognized_blocks:
                    notes.append(
                        f"Page {page_index + 1}: no recognized text blocks; using source PDF text for layout."
                    )

                elements: List[LayoutElement] = []

                # Build text elements
                for idx, block in enumerate(text_blocks):
                    text_content = self._select_text_content(idx, block["text"], recognized_blocks)
                    if not text_content:
                        continue

                    html_fragment = self._render_text_block(
                        text_content,
                        block["bbox"],
                        page_width,
                        page_height,
                        block.get("fonts", []),
                    )
                    element = LayoutElement(
                        element_id=f"text_{page_index + 1}_{idx}",
                        type=LayoutElementType.TEXT,
                        bbox=block["bbox"],
                        text=text_content,
                        html=html_fragment,
                        metadata={
                            "font_summary": self._summarize_fonts(block.get("fonts", [])),
                        },
                    )
                    elements.append(element)

                # Build table elements
                table_elements = self._render_tables_for_page(content, page_index + 1, page_width, page_height)
                if table_elements:
                    elements.extend(table_elements)

                # Build formula elements
                formula_elements = self._render_formulas_for_page(content, page_index + 1, page_width, page_height)
                if formula_elements:
                    elements.extend(formula_elements)

                # Sort elements by position for consistent rendering order
                elements.sort(key=lambda el: (el.bbox.y0, el.bbox.x0))

                page_html = self._compose_page_html(page_index + 1, page_width, page_height, elements)
                reconstructed_pages.append(
                    ReconstructedPage(
                        page_number=page_index + 1,
                        width=page_width,
                        height=page_height,
                        elements=elements,
                        html=page_html,
                    )
                )

            css = self._base_css()
            processing_time = time.time() - start

            return LayoutReconstructionResult(
                method="pymupdf-layout",
                processing_time=processing_time,
                css=css,
                pages=reconstructed_pages,
                notes=notes,
            )
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_pages(self, pages: Optional[List[int]], total_pages: int) -> List[int]:
        if not pages:
            return list(range(total_pages))
        resolved: List[int] = []
        for page in pages:
            index = max(page - 1, 0)
            if index < total_pages:
                resolved.append(index)
        return resolved

    def _extract_text_blocks(self, page: "fitz.Page") -> List[Dict[str, Any]]:
        text_dict = page.get_text("dict")
        blocks: List[Dict[str, Any]] = []

        for block in text_dict.get("blocks", []):
            block_bbox = block.get("bbox")
            if not block_bbox or not block.get("lines"):
                continue

            text_lines: List[str] = []
            fonts: List[Dict[str, Any]] = []
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    segment = span.get("text", "")
                    if segment:
                        line_text += segment
                        fonts.append({"font": span.get("font"), "size": span.get("size")})
                if line_text.strip():
                    text_lines.append(line_text)

            combined = "\n".join(text_lines).strip()
            if not combined:
                continue

            bbox = BoundingBox(x0=block_bbox[0], y0=block_bbox[1], x1=block_bbox[2], y1=block_bbox[3])
            blocks.append({"bbox": bbox, "text": combined, "fonts": fonts})

        blocks.sort(key=lambda b: (b["bbox"].y0, b["bbox"].x0))
        return blocks

    def _recognized_blocks_for_page(self, content: ProcessingContent, page_number: int) -> List[str]:
        if not content or not content.text:
            return []
        for page_text in content.text.pages:
            if page_text.page == page_number and page_text.text:
                return self._split_text_blocks(page_text.text)
        return []

    def _split_text_blocks(self, text: str) -> List[str]:
        chunks = [chunk.strip() for chunk in text.split("\n\n")]
        return [chunk for chunk in chunks if chunk]

    def _select_text_content(self, index: int, fallback: str, recognized: List[str]) -> str:
        if index < len(recognized) and recognized[index].strip():
            return recognized[index].strip()
        return fallback.strip()

    def _render_text_block(
        self,
        text: str,
        bbox: BoundingBox,
        page_width: float,
        page_height: float,
        fonts: Iterable[Dict[str, Any]],
    ) -> str:
        escaped = html.escape(text).replace("\n", "<br />")
        style = self._bbox_to_style(bbox, page_width, page_height)
        font_style = self._font_style(fonts)
        return f'<div class="pdf-block pdf-block-text" style="{style}{font_style}">{escaped}</div>'

    def _render_tables_for_page(
        self,
        content: ProcessingContent,
        page_number: int,
        page_width: float,
        page_height: float,
    ) -> List[LayoutElement]:
        if not content.tables:
            return []

        table_elements: List[LayoutElement] = []
        for table in content.tables.tables:
            if table.page != page_number:
                continue
            html_table = self._table_to_html(table)
            wrapped = self._wrap_absolute(
                table.bbox,
                page_width,
                page_height,
                html_table,
                "pdf-block-table",
            )
            table_elements.append(
                LayoutElement(
                    element_id=f"table_{page_number}_{table.table_id}",
                    type=LayoutElementType.TABLE,
                    bbox=table.bbox,
                    text=None,
                    html=wrapped,
                    metadata={
                        "engine": table.engine,
                        "confidence": table.confidence,
                        "rows": table.rows,
                        "columns": table.columns,
                    },
                )
            )
        return table_elements

    def _render_formulas_for_page(
        self,
        content: ProcessingContent,
        page_number: int,
        page_width: float,
        page_height: float,
    ) -> List[LayoutElement]:
        if not content.formulas:
            return []

        elements: List[LayoutElement] = []
        for formula in content.formulas.formulas:
            if formula.page != page_number:
                continue
            latex = html.escape(formula.latex)
            span = f'<span class="pdf-formula" data-latex="{latex}">\\({latex}\\)</span>'
            wrapped = self._wrap_absolute(
                formula.bbox,
                page_width,
                page_height,
                span,
                "pdf-block-formula",
            )
            elements.append(
                LayoutElement(
                    element_id=f"formula_{page_number}_{formula.formula_id}",
                    type=LayoutElementType.FORMULA,
                    bbox=formula.bbox,
                    text=formula.latex,
                    html=wrapped,
                    metadata={
                        "confidence": formula.confidence,
                        "model": formula.model,
                    },
                )
            )
        return elements

    def _table_to_html(self, table: TableData) -> str:
        rows_html: List[str] = []
        for row in table.data:
            cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
            rows_html.append(f"<tr>{cells}</tr>")
        header_html = ""
        if table.headers:
            header_cells = "".join(f"<th>{html.escape(cell)}</th>" for cell in table.headers)
            header_html = f"<thead><tr>{header_cells}</tr></thead>"
        body_html = "<tbody>" + "".join(rows_html) + "</tbody>"
        return f"<table class=\"pdf-table\">{header_html}{body_html}</table>"

    def _wrap_absolute(
        self,
        bbox: BoundingBox,
        page_width: float,
        page_height: float,
        inner_html: str,
        extra_class: str,
    ) -> str:
        style = self._bbox_to_style(bbox, page_width, page_height)
        return f'<div class="pdf-block {extra_class}" style="{style}">{inner_html}</div>'

    def _compose_page_html(
        self,
        page_number: int,
        page_width: float,
        page_height: float,
        elements: List[LayoutElement],
    ) -> str:
        joined = "\n".join(element.html for element in elements)
        return (
            f'<section class="pdf-page" data-page="{page_number}" '
            f'style="width:{page_width}px;height:{page_height}px;">\n{joined}\n</section>'
        )

    def _bbox_to_style(self, bbox: BoundingBox, page_width: float, page_height: float) -> str:
        if page_width == 0 or page_height == 0:
            return ""
        left = (bbox.x0 / page_width) * 100
        top = (bbox.y0 / page_height) * 100
        width = max((bbox.x1 - bbox.x0) / page_width * 100, 0.1)
        height = max((bbox.y1 - bbox.y0) / page_height * 100, 0.1)
        return (
            f"left:{left:.3f}%;top:{top:.3f}%;width:{width:.3f}%;height:{height:.3f}%;"
        )

    def _font_style(self, fonts: Iterable[Dict[str, Any]]) -> str:
        if not fonts:
            return ""
        sizes = [font.get("size") for font in fonts if isinstance(font.get("size"), (int, float))]
        families = [font.get("font") for font in fonts if font.get("font")]
        pieces: List[str] = []
        if sizes:
            average = sum(sizes) / len(sizes)
            pieces.append(f"font-size:{average:.1f}px;")
        if families:
            most_common = Counter(families).most_common(1)[0][0]
            pieces.append(f"font-family:'{most_common}', sans-serif;")
        return "".join(pieces)

    def _summarize_fonts(self, fonts: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        fonts = [font for font in fonts if font.get("font")]
        if not fonts:
            return None
        families = Counter(font.get("font") for font in fonts if font.get("font"))
        sizes = [font.get("size") for font in fonts if isinstance(font.get("size"), (int, float))]
        summary: Dict[str, Any] = {
            "primary_font": families.most_common(1)[0][0] if families else None,
            "font_count": len(families),
        }
        if sizes:
            summary.update(
                {
                    "average_size": sum(sizes) / len(sizes),
                    "min_size": min(sizes),
                    "max_size": max(sizes),
                }
            )
        return summary

    def _base_css(self) -> str:
        return """
.pdf-page {
  position: relative;
  margin: 1rem auto;
  box-shadow: 0 0 8px rgba(0, 0, 0, 0.15);
  background: #fff;
}

.pdf-block {
  position: absolute;
  overflow: hidden;
  white-space: normal;
}

.pdf-block-text {
  line-height: 1.35;
  color: #222;
}

.pdf-block-table table {
  border-collapse: collapse;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.95);
}

.pdf-block-table th,
.pdf-block-table td {
  border: 1px solid rgba(0, 0, 0, 0.15);
  padding: 0.2rem 0.3rem;
  font-size: 0.9em;
}

.pdf-block-formula {
  display: flex;
  align-items: center;
  justify-content: center;
}

.pdf-formula {
  font-family: "Times New Roman", serif;
  font-size: 1em;
}
"""

