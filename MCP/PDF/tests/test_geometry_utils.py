"""Unit tests for geometry helper functions."""

import sys
from pathlib import Path

import fitz
import pytest


# Ensure the src package is importable when tests execute from the repository root.
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pdf_mcp_server.models import BoundingBox
from pdf_mcp_server.utils.geometry import (
    bbox_distance,
    bbox_iou,
    normalize_bbox,
)


def _make_page(width: int = 200, height: int = 100, rotation: int = 0):
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    if rotation:
        page.set_rotation(rotation)
    return doc, page


def test_normalize_bbox_top_left_origin():
    doc, page = _make_page()
    try:
        bbox = normalize_bbox(page, (0, 0, 20, 10))
        assert bbox.x0 == pytest.approx(0)
        assert bbox.y0 == pytest.approx(90)
        assert bbox.x1 == pytest.approx(20)
        assert bbox.y1 == pytest.approx(100)
    finally:
        doc.close()


def test_normalize_bbox_rotated_page():
    doc, page = _make_page(rotation=90)
    try:
        bbox = normalize_bbox(page, (0, 0, 20, 10))
        assert bbox.x0 == pytest.approx(90)
        assert bbox.y0 == pytest.approx(180)
        assert bbox.x1 == pytest.approx(100)
        assert bbox.y1 == pytest.approx(200)
    finally:
        doc.close()


def test_bbox_distance_and_iou():
    a = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    b = BoundingBox(x0=5, y0=5, x1=15, y1=15)

    assert bbox_iou(a, b) == pytest.approx(25 / 175)
    assert bbox_distance(a, b) == pytest.approx(((5) ** 2 + (5) ** 2) ** 0.5)
