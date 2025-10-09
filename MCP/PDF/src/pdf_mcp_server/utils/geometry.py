"""Geometry helpers for PDF coordinate normalization.

The PDF toolchain mixes coordinate systems depending on the backend. This
module centralises conversions so that layout reconstruction, table placement
and other downstream consumers rely on a single canonical format:

* origin at the top-left corner of the page
* coordinates expressed in PDF points (float)
* bounding boxes guaranteed to be ordered (x0 <= x1, y0 <= y1)

In addition to the normalisation routine the module exposes small helpers for
area, IoU and distance computations that are frequently required when aligning
recognised elements to geometric primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

try:  # Optional dependency at runtime; callers must guard usage accordingly.
    import fitz  # type: ignore
except Exception:  # pragma: no cover - handled by graceful degradation
    fitz = None  # type: ignore

from ..models import BoundingBox


BBoxLike = Union[BoundingBox, Sequence[float], Tuple[float, float, float, float]]


@dataclass(frozen=True)
class NormalizationOptions:
    """Options controlling the behaviour of :func:`normalize_bbox`."""

    clamp: bool = True
    origin_top_left: bool = True


def _ensure_rect(raw: BBoxLike) -> "fitz.Rect":
    if fitz is None:  # pragma: no cover - defensive, should be guarded by callers
        raise RuntimeError("PyMuPDF is required for geometry normalisation")

    if isinstance(raw, BoundingBox):
        return fitz.Rect(raw.x0, raw.y0, raw.x1, raw.y1)

    if isinstance(raw, Iterable):
        values = list(raw)
        if len(values) != 4:
            raise ValueError("Bounding box iterable must contain exactly 4 numbers")
        return fitz.Rect(*values)

    raise TypeError(f"Unsupported bounding box type: {type(raw)!r}")


def normalize_bbox(
    page: "fitz.Page",
    raw_bbox: BBoxLike,
    *,
    options: Optional[NormalizationOptions] = None,
) -> BoundingBox:
    """Convert a bounding box into the canonical coordinate system.

    Args:
        page: PyMuPDF page instance owning the coordinates
        raw_bbox: Bounding box in any page-specific coordinate system
        options: Optional normalisation flags

    Returns:
        A :class:`BoundingBox` anchored at the top-left corner of the page.
    """

    if fitz is None:  # pragma: no cover - defensive, runtime guard
        raise RuntimeError("PyMuPDF is not available; cannot normalise bounding boxes")

    opts = options or NormalizationOptions()
    rect = _ensure_rect(raw_bbox)

    # Apply page rotation and crop offsets so that coordinates are expressed
    # relative to the final rendered page rectangle.
    matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
    if getattr(page, "rotation", 0):
        matrix = page.rotation_matrix * matrix
    rect = rect.transform(matrix)

    if hasattr(page, "cropbox_position"):
        crop_x, crop_y = page.cropbox_position
        rect = fitz.Rect(rect.x0 - crop_x, rect.y0 - crop_y, rect.x1 - crop_x, rect.y1 - crop_y)

    page_rect = page.rect
    width = float(page_rect.width)
    height = float(page_rect.height)

    x0 = min(rect.x0, rect.x1)
    y0 = min(rect.y0, rect.y1)
    x1 = max(rect.x0, rect.x1)
    y1 = max(rect.y0, rect.y1)

    if opts.origin_top_left:
        # PDF coordinates are bottom-left by default; invert to top-left origin.
        y0, y1 = height - y1, height - y0

    if opts.clamp:
        x0 = max(0.0, min(x0, width))
        x1 = max(0.0, min(x1, width))
        y0 = max(0.0, min(y0, height))
        y1 = max(0.0, min(y1, height))

    # Ensure non-degenerate boxes even when upstream data is imperfect.
    if x1 - x0 < 1e-3:
        x1 = min(width, x0 + 1e-3)
    if y1 - y0 < 1e-3:
        y1 = min(height, y0 + 1e-3)

    return BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))


def bbox_area(bbox: BoundingBox) -> float:
    return max(0.0, (bbox.x1 - bbox.x0)) * max(0.0, (bbox.y1 - bbox.y0))


def bbox_intersection(a: BoundingBox, b: BoundingBox) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    inter = bbox_intersection(a, b)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_center(bbox: BoundingBox) -> Tuple[float, float]:
    return (bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0


def bbox_distance(a: BoundingBox, b: BoundingBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


__all__ = [
    "NormalizationOptions",
    "normalize_bbox",
    "bbox_area",
    "bbox_intersection",
    "bbox_iou",
    "bbox_center",
    "bbox_distance",
]

