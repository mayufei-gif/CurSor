"""Layout reconstruction processor.

Rebuilds a near-original representation of a PDF page layout using
previous extraction outputs (text, tables, formulas) and the source PDF
geometry.
"""

from __future__ import annotations

import html
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    TextBlock,
    TextLine,
    TextSpan,
)
from ..utils.config import Config
from ..utils.exceptions import PDFProcessingError
from ..utils.geometry import (
    bbox_distance,
    bbox_iou,
    normalize_bbox,
)


@dataclass
class LayoutBlock:
    bbox: BoundingBox
    text: str
    fonts: List[Dict[str, Any]]
    lines: List[BoundingBox]
    spans: List[TextSpan]
    alignment: Optional[str]
    line_texts: List[str]


@dataclass
class RecognizedBlock:
    block: TextBlock
    bbox: BoundingBox


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
                recognized_blocks = self._recognized_blocks_for_page(page, content, page_index + 1)
                match_map = self._align_text_blocks(text_blocks, recognized_blocks, page_width, page_height)

                if recognized_blocks and len(match_map) < len(text_blocks):
                    notes.append(
                        f"Page {page_index + 1}: matched {len(match_map)} of {len(text_blocks)} layout blocks to recognised text; remaining blocks use PDF text."
                    )
                elif not recognized_blocks:
                    notes.append(
                        f"Page {page_index + 1}: recognised text blocks unavailable; using PDF text only."
                    )

                elements: List[LayoutElement] = []

                ordered_indices = self._compute_reading_order(text_blocks, page_width, page_height)

                for position, idx in enumerate(ordered_indices):
                    block = text_blocks[idx]
                    matched = match_map.get(idx)
                    block_text = matched.block.text if matched else block.text
                    if not block_text:
                        continue

                    spans = matched.block.spans if matched and matched.block.spans else block.spans
                    lines = matched.block.lines if matched and matched.block.lines else self._build_faux_lines(block)
                    alignment = matched.block.alignment if matched and matched.block.alignment else block.alignment

                    html_fragment = self._render_text_block(
                        block_text,
                        block.bbox,
                        page_width,
                        page_height,
                        spans,
                        lines,
                        alignment,
                        z_index=self._element_z_index(LayoutElementType.TEXT),
                    )
                    element = LayoutElement(
                        element_id=f"text_{page_index + 1}_{idx}",
                        type=LayoutElementType.TEXT,
                        bbox=block.bbox,
                        text=block_text,
                        html=html_fragment,
                        metadata={
                            "font_summary": self._summarize_fonts(block.fonts),
                            "source": "recognized" if matched else "pdf_layout",
                            "reading_order": position,
                        },
                    )
                    elements.append(element)

                # Build table elements
                table_elements = self._render_tables_for_page(
                    content,
                    page_index + 1,
                    page_width,
                    page_height,
                    recognized_blocks,
                    page,
                )
                if table_elements:
                    elements.extend(table_elements)

                # Build formula elements
                formula_elements = self._render_formulas_for_page(
                    content,
                    page_index + 1,
                    page_width,
                    page_height,
                    recognized_blocks,
                    page,
                )
                if formula_elements:
                    elements.extend(formula_elements)

                elements = self._sort_elements_with_layers(elements)

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

            dependency_notes = [
                "Layout HTML expects MathJax or KaTeX CSS/JS to render formulas (KaTeX stylesheet is referenced via @import).",
                "For best visual fidelity load document fonts via @font-face or ensure serif fallback fonts are available.",
            ]
            for note in dependency_notes:
                if note not in notes:
                    notes.append(note)

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
            index = max(page, 0)
            if index < total_pages:
                resolved.append(index)
        return resolved

    def _linear_sum_assignment(
        self, cost_matrix: List[List[float]], filler: float
    ) -> List[Tuple[int, int]]:
        """Solve the rectangular assignment problem using the Hungarian algorithm."""

        n_rows = len(cost_matrix)
        n_cols = max((len(row) for row in cost_matrix), default=0)
        size = max(n_rows, n_cols)
        if size == 0:
            return []

        matrix = [row + [filler] * (size - len(row)) for row in cost_matrix]
        for _ in range(size - n_rows):
            matrix.append([filler] * size)

        starred = [[False] * size for _ in range(size)]
        primed = [[False] * size for _ in range(size)]
        row_covered = [False] * size
        col_covered = [False] * size

        # Step 1: subtract row minima
        for i in range(size):
            row_min = min(matrix[i])
            matrix[i] = [val - row_min for val in matrix[i]]

        # Step 2: subtract column minima
        for j in range(size):
            col_min = min(matrix[i][j] for i in range(size))
            for i in range(size):
                matrix[i][j] -= col_min

        # Step 3: star zeros greedily
        for i in range(size):
            for j in range(size):
                if matrix[i][j] == 0 and not row_covered[i] and not col_covered[j]:
                    starred[i][j] = True
                    row_covered[i] = True
                    col_covered[j] = True

        row_covered = [False] * size
        col_covered = [False] * size

        def cover_columns_with_starred_zero():
            for i in range(size):
                for j in range(size):
                    if starred[i][j]:
                        col_covered[j] = True

        cover_columns_with_starred_zero()

        def find_a_zero():
            for i in range(size):
                if row_covered[i]:
                    continue
                for j in range(size):
                    if not col_covered[j] and matrix[i][j] == 0:
                        return i, j
            return None

        def find_star_in_row(row: int) -> Optional[int]:
            for j in range(size):
                if starred[row][j]:
                    return j
            return None

        def find_star_in_col(col: int) -> Optional[int]:
            for i in range(size):
                if starred[i][col]:
                    return i
            return None

        def find_prime_in_row(row: int) -> Optional[int]:
            for j in range(size):
                if primed[row][j]:
                    return j
            return None

        while sum(col_covered) < size:
            zero = find_a_zero()
            while zero is None:
                # Step 5: adjust matrix
                uncovered_values = [
                    matrix[i][j]
                    for i in range(size)
                    for j in range(size)
                    if not row_covered[i] and not col_covered[j]
                ]
                min_uncovered = min(uncovered_values)
                for i in range(size):
                    for j in range(size):
                        if row_covered[i]:
                            matrix[i][j] += min_uncovered
                        if not col_covered[j]:
                            matrix[i][j] -= min_uncovered
                zero = find_a_zero()

            i, j = zero
            primed[i][j] = True
            star_col = find_star_in_row(i)
            if star_col is not None:
                row_covered[i] = True
                col_covered[star_col] = False
            else:
                # Augmenting path
                path = [(i, j)]
                while True:
                    star_row = find_star_in_col(path[-1][1])
                    if star_row is None:
                        break
                    path.append((star_row, path[-1][1]))
                    prime_col = find_prime_in_row(star_row)
                    if prime_col is None:
                        break
                    path.append((star_row, prime_col))

                for r, c in path:
                    starred[r][c] = not starred[r][c]
                    primed[r][c] = False

                primed = [[False] * size for _ in range(size)]
                row_covered = [False] * size
                col_covered = [False] * size
                cover_columns_with_starred_zero()

        assignments: List[Tuple[int, int]] = []
        for i in range(n_rows):
            for j in range(n_cols):
                if starred[i][j]:
                    assignments.append((i, j))
        return assignments

    def _extract_text_blocks(self, page: "fitz.Page") -> List[LayoutBlock]:
        text_dict = page.get_text("dict")
        blocks: List[LayoutBlock] = []
        page_width = float(page.rect.width or 1.0)

        for block in text_dict.get("blocks", []):
            raw_bbox = block.get("bbox")
            if not raw_bbox or not block.get("lines"):
                continue

            normalized_bbox = normalize_bbox(page, raw_bbox)

            lines: List[BoundingBox] = []
            spans: List[TextSpan] = []
            fonts: List[Dict[str, Any]] = []
            line_texts: List[str] = []

            for line in block.get("lines", []):
                line_bbox = line.get("bbox")
                if line_bbox:
                    lines.append(normalize_bbox(page, line_bbox))
                line_text = []
                for span in line.get("spans", []):
                    text_segment = span.get("text", "")
                    if not text_segment:
                        continue
                    fonts.append({"font": span.get("font"), "size": span.get("size")})
                    spans.append(
                        TextSpan(
                            text=text_segment,
                            font=span.get("font"),
                            size=span.get("size"),
                            bold=bool(span.get("flags", 0) & 2),
                            italic=bool(span.get("flags", 0) & 1),
                        )
                    )
                    line_text.append(text_segment)
                joined = "".join(line_text).strip()
                if joined:
                    line_texts.append(joined)

            combined_text = "\n".join(line_texts).strip()
            if not combined_text:
                continue

            alignment = self._infer_alignment(normalized_bbox, page_width)
            blocks.append(
                LayoutBlock(
                    bbox=normalized_bbox,
                    text=combined_text,
                    fonts=fonts,
                    lines=lines,
                    spans=spans,
                    alignment=alignment,
                    line_texts=line_texts,
                )
            )

        blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
        return blocks

    def _align_text_blocks(
        self,
        layout_blocks: List[LayoutBlock],
        recognized_blocks: List[RecognizedBlock],
        page_width: float,
        page_height: float,
    ) -> Dict[int, RecognizedBlock]:
        if not layout_blocks or not recognized_blocks:
            return {}

        cost_matrix: List[List[float]] = []
        max_cost = 0.0
        for layout in layout_blocks:
            diag = math.hypot(layout.bbox.x1 - layout.bbox.x0, layout.bbox.y1 - layout.bbox.y0) or 1.0
            row: List[float] = []
            for recognized in recognized_blocks:
                iou = bbox_iou(layout.bbox, recognized.bbox)
                distance = bbox_distance(layout.bbox, recognized.bbox)
                normalized_distance = distance / diag
                cost = (1.0 - iou) + 0.25 * normalized_distance
                row.append(cost)
                if cost > max_cost:
                    max_cost = cost
            cost_matrix.append(row)

        assignments = self._linear_sum_assignment(cost_matrix, max_cost + 10.0)
        matches: Dict[int, RecognizedBlock] = {}
        for row, col in assignments:
            if row >= len(layout_blocks) or col >= len(recognized_blocks):
                continue
            layout = layout_blocks[row]
            recognized = recognized_blocks[col]
            if not self._is_viable_match(layout, recognized):
                continue
            matches[row] = recognized

        if matches:
            self._harmonize_alignment(layout_blocks, matches, page_width, page_height)

        return matches

    def _is_viable_match(self, layout: LayoutBlock, recognized: RecognizedBlock) -> bool:
        iou = bbox_iou(layout.bbox, recognized.bbox)
        if iou >= 0.1:
            return True
        distance = bbox_distance(layout.bbox, recognized.bbox)
        diag = math.hypot(layout.bbox.x1 - layout.bbox.x0, layout.bbox.y1 - layout.bbox.y0) or 1.0
        return distance < diag * 0.7

    def _harmonize_alignment(
        self,
        layout_blocks: List[LayoutBlock],
        matches: Dict[int, RecognizedBlock],
        page_width: float,
        page_height: float,
    ) -> None:
        if not matches:
            return

        shifts_x: List[float] = []
        shifts_y: List[float] = []
        for idx, recognized in matches.items():
            layout_bbox = layout_blocks[idx].bbox
            shifts_x.append(layout_bbox.x0 - recognized.bbox.x0)
            shifts_y.append(layout_bbox.y0 - recognized.bbox.y0)

        mean_dx = sum(shifts_x) / len(shifts_x)
        mean_dy = sum(shifts_y) / len(shifts_y)

        if abs(mean_dx) < 0.5 and abs(mean_dy) < 0.5:
            return

        for match in matches.values():
            shifted_bbox = self._shift_bbox(match.bbox, mean_dx, mean_dy, page_width, page_height)
            updated_lines: List[TextLine] = []
            for line in match.block.lines:
                if line.bbox:
                    shifted_line_bbox = self._shift_bbox(line.bbox, mean_dx, mean_dy, page_width, page_height)
                    updated_lines.append(line.model_copy(update={"bbox": shifted_line_bbox}))
                else:
                    updated_lines.append(line)
            match.block = match.block.model_copy(update={"bbox": shifted_bbox, "lines": updated_lines})
            match.bbox = shifted_bbox

    def _shift_bbox(
        self,
        bbox: BoundingBox,
        dx: float,
        dy: float,
        page_width: float,
        page_height: float,
    ) -> BoundingBox:
        x0 = min(max(bbox.x0 + dx, 0.0), page_width)
        y0 = min(max(bbox.y0 + dy, 0.0), page_height)
        x1 = min(max(bbox.x1 + dx, 0.0), page_width)
        y1 = min(max(bbox.y1 + dy, 0.0), page_height)
        if x1 <= x0:
            x1 = min(page_width, x0 + 0.5)
        if y1 <= y0:
            y1 = min(page_height, y0 + 0.5)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _compute_reading_order(
        self,
        blocks: List[LayoutBlock],
        page_width: float,
        page_height: float,
    ) -> List[int]:
        if not blocks:
            return []

        columns: List[List[int]] = []
        footnotes: List[int] = []
        column_threshold = max(page_width * 0.12, 36.0)

        for idx, block in enumerate(blocks):
            if self._is_footnote(block, page_height):
                footnotes.append(idx)
                continue

            placed = False
            for column in columns:
                ref = blocks[column[0]]
                if abs(block.bbox.x0 - ref.bbox.x0) <= column_threshold:
                    column.append(idx)
                    placed = True
                    break
            if not placed:
                columns.append([idx])

        columns.sort(key=lambda col: min(blocks[i].bbox.x0 for i in col))
        ordered: List[int] = []
        for column in columns:
            column.sort(key=lambda i: blocks[i].bbox.y0)
            ordered.extend(column)

        footnotes.sort(key=lambda i: blocks[i].bbox.y0)
        ordered.extend(footnotes)

        if not ordered:
            ordered = list(range(len(blocks)))
        return ordered

    def _is_footnote(self, block: LayoutBlock, page_height: float) -> bool:
        height = block.bbox.y1 - block.bbox.y0
        return block.bbox.y0 > page_height * 0.82 and height < page_height * 0.18 and len(block.line_texts) <= 3

    def _build_faux_lines(self, block: LayoutBlock) -> List[TextLine]:
        if block.lines and block.line_texts:
            return [
                TextLine(text=text, bbox=bbox, spans=[])
                for text, bbox in zip(block.line_texts, block.lines)
            ]
        return [TextLine(text=block.text, bbox=block.bbox, spans=[])]

    def _element_z_index(self, element_type: LayoutElementType) -> int:
        mapping = {
            LayoutElementType.TEXT: 10,
            LayoutElementType.TABLE: 30,
            LayoutElementType.FORMULA: 40,
            LayoutElementType.IMAGE: 20,
            LayoutElementType.OTHER: 5,
        }
        return mapping.get(element_type, 1)

    def _sort_elements_with_layers(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        return sorted(
            elements,
            key=lambda el: (
                self._element_z_index(el.type),
                el.metadata.get("reading_order", 0) if isinstance(el.metadata, dict) else 0,
                el.bbox.y0,
                el.bbox.x0,
            ),
        )

    def _recognized_blocks_for_page(
        self,
        page: "fitz.Page",
        content: ProcessingContent,
        page_number: int,
    ) -> List[RecognizedBlock]:
        if not content or not content.text:
            return []

        for page_text in content.text.pages:
            if page_text.page != page_number:
                continue

            recognized: List[RecognizedBlock] = []
            for block in page_text.blocks:
                if not block.bbox:
                    continue
                normalized_bbox = normalize_bbox(page, block.bbox)
                normalized_lines: List[TextLine] = []
                for line in block.lines:
                    if line.bbox:
                        normalized_line = line.model_copy(
                            update={
                                "bbox": normalize_bbox(page, line.bbox),
                            }
                        )
                    else:
                        normalized_line = line
                    normalized_lines.append(normalized_line)

                block_copy = block.model_copy(
                    update={
                        "bbox": normalized_bbox,
                        "lines": normalized_lines,
                    }
                )
                recognized.append(RecognizedBlock(block=block_copy, bbox=normalized_bbox))
            return recognized

        return []

    def _render_text_block(
        self,
        text: str,
        bbox: BoundingBox,
        page_width: float,
        page_height: float,
        spans: Iterable[TextSpan],
        lines: Iterable[TextLine],
        alignment: Optional[str],
        *,
        z_index: Optional[int] = None,
    ) -> str:
        style = self._bbox_to_style(bbox, page_width, page_height, z_index=z_index)
        alignment_style = f"text-align:{alignment};" if alignment else ""
        content = self._render_lines(list(lines), list(spans), text)
        return f'<div class="pdf-block pdf-block-text" style="{style}{alignment_style}">{content}</div>'

    def _render_lines(
        self,
        lines: List[TextLine],
        spans: List[TextSpan],
        fallback_text: str,
    ) -> str:
        if lines:
            fragments: List[str] = []
            for line in lines:
                fragments.append(
                    '<div class="pdf-line">'
                    + self._render_span_sequence(line.spans, line.text)
                    + "</div>"
                )
            return "".join(fragments)
        return self._render_span_sequence(spans, fallback_text)

    def _render_span_sequence(self, spans: List[TextSpan], fallback_text: str) -> str:
        if not spans:
            return html.escape(fallback_text).replace("\n", "<br />")

        pieces: List[str] = []
        for span in spans:
            if not span.text:
                continue
            style_bits: List[str] = []
            if span.size:
                style_bits.append(f"font-size:{span.size:.1f}px;")
            if span.font:
                style_bits.append(f"font-family:'{span.font}', sans-serif;")
            if span.bold:
                style_bits.append("font-weight:bold;")
            if span.italic:
                style_bits.append("font-style:italic;")
            style_attr = f" style=\"{''.join(style_bits)}\"" if style_bits else ""
            pieces.append(f"<span{style_attr}>{html.escape(span.text)}</span>")
        return "".join(pieces) if pieces else html.escape(fallback_text)

    def _normalize_or_fallback_bbox(
        self,
        page: "fitz.Page",
        bbox: BoundingBox,
        page_width: float,
        page_height: float,
        recognized_blocks: List[RecognizedBlock],
        tokens: Iterable[Optional[str]],
    ) -> BoundingBox:
        candidate = bbox
        if candidate:
            try:
                candidate = normalize_bbox(page, candidate)
            except Exception:  # pragma: no cover - defensive
                candidate = bbox

        if not candidate or self._is_degenerate(candidate):
            fallback = self._fallback_bbox_from_text(recognized_blocks, tokens)
            if fallback:
                candidate = fallback

        if not candidate:
            candidate = BoundingBox(x0=0.0, y0=0.0, x1=page_width, y1=page_height / 6)

        return BoundingBox(
            x0=min(max(candidate.x0, 0.0), page_width),
            y0=min(max(candidate.y0, 0.0), page_height),
            x1=min(max(candidate.x1, 0.0), page_width),
            y1=min(max(candidate.y1, 0.0), page_height),
        )

    def _is_degenerate(self, bbox: BoundingBox) -> bool:
        return (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0) < 1.0

    def _fallback_bbox_from_text(
        self,
        recognized_blocks: List[RecognizedBlock],
        tokens: Iterable[Optional[str]],
    ) -> Optional[BoundingBox]:
        cleaned_tokens = [token.strip() for token in tokens if token]
        if not cleaned_tokens:
            return None

        matched_boxes: List[BoundingBox] = []
        for candidate in recognized_blocks:
            text = candidate.block.text.replace("\n", " ") if candidate.block.text else ""
            for token in cleaned_tokens:
                if token and token in text:
                    matched_boxes.append(candidate.bbox)
                    break

        if not matched_boxes:
            return None

        x0 = min(box.x0 for box in matched_boxes)
        y0 = min(box.y0 for box in matched_boxes)
        x1 = max(box.x1 for box in matched_boxes)
        y1 = max(box.y1 for box in matched_boxes)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _render_tables_for_page(
        self,
        content: ProcessingContent,
        page_number: int,
        page_width: float,
        page_height: float,
        recognized_blocks: List[RecognizedBlock],
        page: "fitz.Page",
    ) -> List[LayoutElement]:
        if not content.tables:
            return []

        table_elements: List[LayoutElement] = []
        for table in content.tables.tables:
            if table.page != page_number:
                continue
            html_table = self._table_to_html(table)
            bbox = self._normalize_or_fallback_bbox(
                page,
                table.bbox,
                page_width,
                page_height,
                recognized_blocks,
                [table.headers[0]] if table.headers else table.data[0] if table.data else [],
            )
            wrapped = self._wrap_absolute(
                bbox,
                page_width,
                page_height,
                html_table,
                "pdf-block-table",
                z_index=self._element_z_index(LayoutElementType.TABLE),
            )
            table_elements.append(
                LayoutElement(
                    element_id=f"table_{page_number}_{table.table_id}",
                    type=LayoutElementType.TABLE,
                    bbox=bbox,
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
        recognized_blocks: List[RecognizedBlock],
        page: "fitz.Page",
    ) -> List[LayoutElement]:
        if not content.formulas:
            return []

        elements: List[LayoutElement] = []
        for formula in content.formulas.formulas:
            if formula.page != page_number:
                continue
            latex = html.escape(formula.latex)
            span = f'<span class="pdf-formula" data-latex="{latex}">\\({latex}\\)</span>'
            bbox = self._normalize_or_fallback_bbox(
                page,
                formula.bbox,
                page_width,
                page_height,
                recognized_blocks,
                [formula.raw_text] if getattr(formula, "raw_text", None) else [formula.latex],
            )
            wrapped = self._wrap_absolute(
                bbox,
                page_width,
                page_height,
                span,
                "pdf-block-formula",
                z_index=self._element_z_index(LayoutElementType.FORMULA),
            )
            elements.append(
                LayoutElement(
                    element_id=f"formula_{page_number}_{formula.formula_id}",
                    type=LayoutElementType.FORMULA,
                    bbox=bbox,
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
        *,
        z_index: Optional[int] = None,
    ) -> str:
        style = self._bbox_to_style(bbox, page_width, page_height, z_index=z_index)
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

    def _bbox_to_style(
        self,
        bbox: BoundingBox,
        page_width: float,
        page_height: float,
        *,
        z_index: Optional[int] = None,
    ) -> str:
        if page_width == 0 or page_height == 0:
            return ""
        left = (bbox.x0 / page_width) * 100
        top = (bbox.y0 / page_height) * 100
        width = max((bbox.x1 - bbox.x0) / page_width * 100, 0.1)
        height = max((bbox.y1 - bbox.y0) / page_height * 100, 0.1)
        pieces = [
            f"left:{left:.3f}%;",
            f"top:{top:.3f}%;",
            f"width:{width:.3f}%;",
            f"height:{height:.3f}%;",
        ]
        if z_index is not None:
            pieces.append(f"z-index:{z_index};")
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
@import url('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css');

.pdf-page {
  position: relative;
  margin: 1rem auto;
  box-shadow: 0 0 8px rgba(0, 0, 0, 0.15);
  background: #fff;
  font-family: 'Noto Serif', 'Times New Roman', serif;
}

.pdf-block {
  position: absolute;
  overflow: hidden;
  white-space: normal;
}

.pdf-block-text {
  line-height: 1.35;
  color: #222;
  word-break: break-word;
}

.pdf-line {
  width: 100%;
}

.pdf-line span {
  display: inline;
}

.pdf-block-table {
  background: rgba(255, 255, 255, 0.95);
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

