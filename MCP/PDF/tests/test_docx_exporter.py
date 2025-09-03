import os
from pathlib import Path

from pdf_mcp_server.utils.docx_exporter import build_docx_from_pipeline


def test_build_docx_from_pipeline(tmp_path: Path):
    # Prepare minimal pipeline-like dict
    sample = {
        "content": {
            "text": {
                "full_text": "Hello world\n\nThis is a test document.",
                "pages": [
                    {"page": 1, "text": "Hello world"},
                ],
                "word_count": 5,
                "character_count": 27,
                "extraction_method": "unit-test",
                "processing_time": 0.01,
            },
            "tables": {
                "tables": [
                    {
                        "page": 1,
                        "table_id": 0,
                        "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                        "data": [["H1", "H2"], ["1", "2"]],
                        "headers": ["H1", "H2"],
                        "confidence": 0.9,
                        "engine": "unit",
                        "rows": 2,
                        "columns": 2,
                    }
                ],
                "total_tables": 1,
                "extraction_method": "unit",
                "fallback_used": False,
                "processing_time": 0.01,
            },
            "formulas": {
                "formulas": [
                    {
                        "page": 1,
                        "formula_id": 0,
                        "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                        "latex": r"E = mc^2",
                        "confidence": 0.9,
                        "model": "unit",
                        "image_path": None,
                        "raw_text": "E = mc^2",
                    }
                ],
                "total_formulas": 1,
                "model_used": "unit",
                "processing_time": 0.01,
            },
        }
    }

    out = tmp_path / "sample.docx"
    p = build_docx_from_pipeline(sample, out)
    assert Path(p).exists()

