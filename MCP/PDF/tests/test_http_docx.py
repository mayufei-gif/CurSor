from pathlib import Path

from fastapi.testclient import TestClient

from pdf_mcp_server.http_app import app


def _make_pdf(tmp_path: Path) -> Path:
    # Create a minimal PDF using PyMuPDF
    import fitz  # PyMuPDF

    out = tmp_path / "mini.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF-MCP!", fontsize=12)
    doc.save(out)
    doc.close()
    return out


def test_extract_docx_sync(tmp_path: Path):
    client = TestClient(app)
    pdf = _make_pdf(tmp_path)
    with open(pdf, "rb") as f:
        files = {"file": (pdf.name, f, "application/pdf")}
        r = client.post("/extract?mode=read_text&output_format=docx", files=files)
        assert r.status_code == 200
        # Save returned content to file
        out = tmp_path / "api.docx"
        out.write_bytes(r.content)
        assert out.exists()

