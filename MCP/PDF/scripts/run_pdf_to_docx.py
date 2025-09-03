#!/usr/bin/env python3
"""
Helper script: process a PDF via pdf_mcp_server and export a .docx using the
built-in docx exporter.

Usage:
  python run_pdf_to_docx.py --in <pdf_path> --outdir <output_dir>
  python run_pdf_to_docx.py --in <pdf_path> --out <output_docx_path>
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from pdf_mcp_server.models import ProcessingRequest, ProcessingMode
from pdf_mcp_server.processors.pdf_processor import PDFProcessor
from pdf_mcp_server.utils.config import Config
from pdf_mcp_server.utils.docx_exporter import build_docx_from_pipeline


async def process_to_docx(pdf_path: Path, out_path: Path,
                          mode: ProcessingMode = ProcessingMode.FULL,
                          include_ocr: bool = True,
                          include_formulas: bool = False,
                          include_grobid: bool = False) -> Path:
    config = Config()
    processor = PDFProcessor(config)
    await processor.initialize()
    try:
        request = ProcessingRequest(
            file_path=str(pdf_path),
            mode=mode,
            include_ocr=include_ocr,
            include_formulas=include_formulas,
            include_grobid=include_grobid,
        )
        result = await processor.process(request)
    finally:
        await processor.cleanup()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_docx_from_pipeline(result, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Process a PDF and export DOCX")
    ap.add_argument("--in", dest="infile", required=True, help="Input PDF path")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--outdir", help="Output directory for DOCX")
    group.add_argument("--out", help="Output DOCX path")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR even if needed")
    ap.add_argument("--formulas", action="store_true", help="Enable formula extraction")
    args = ap.parse_args()

    pdf_path = Path(args.infile)
    if not pdf_path.exists():
        raise SystemExit(f"Input PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit("Input file must be a .pdf")

    if args.out:
        out_path = Path(args.out)
    else:
        outdir = Path(args.outdir) if args.outdir else pdf_path.parent
        out_path = outdir / (pdf_path.stem + ".docx")

    include_ocr = not args.no_ocr
    include_formulas = bool(args.formulas)

    out = asyncio.run(
        process_to_docx(
            pdf_path=pdf_path,
            out_path=out_path,
            mode=ProcessingMode.FULL,
            include_ocr=include_ocr,
            include_formulas=include_formulas,
            include_grobid=False,
        )
    )
    print(str(out))


if __name__ == "__main__":
    main()

