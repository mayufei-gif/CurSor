"""
FastAPI HTTP application for the PDF-MCP server.

This module exposes REST endpoints that wrap the existing PDFProcessor
pipeline (text, tables, OCR, formulas) so you can run the service via
uvicorn and call it with curl/Postman.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import uuid

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder

from .processors.pdf_processor import PDFProcessor
from .models import ProcessingRequest, ProcessingMode, FormulaModel, TableEngine
from .utils.config import Config
from .utils.docx_exporter import build_docx_from_pipeline
try:
    from .worker import get_queue
except Exception:
    get_queue = lambda: None  # type: ignore


app = FastAPI(title="PDF-MCP HTTP Server", version="0.1.0")


# Global singletons (initialized on startup)
_config: Optional[Config] = None
_processor: Optional[PDFProcessor] = None


# Simple in-memory task registry for optional async execution
_tasks: Dict[str, Dict[str, Any]] = {}
_rq_queue = None


@app.on_event("startup")
async def on_startup():
    global _config, _processor
    _config = Config.load()
    _processor = PDFProcessor(_config)
    await _processor.initialize()
    # Optional RQ queue
    global _rq_queue
    try:
        _rq_queue = get_queue()  # type: ignore
    except Exception:
        _rq_queue = None


@app.on_event("shutdown")
async def on_shutdown():
    global _processor
    if _processor:
        await _processor.cleanup()


@app.get("/health")
async def health() -> Dict[str, Any]:
    if not _processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    deps = await _processor.health_check()

    return {
        "status": "healthy" if all(deps.values()) else "degraded",
        "dependencies": deps,
    }


def _parse_pages(pages: Optional[str]) -> Optional[List[int]]:
    if not pages:
        return None
    pages = pages.strip()
    if "-" in pages and "," not in pages:
        start, end = map(int, pages.split("-"))
        if start <= 0 or end <= 0 or end < start:
            raise ValueError("Invalid page range")
        return list(range(start, end + 1))
    if "," in pages:
        return [int(p.strip()) for p in pages.split(",") if p.strip()]
    # single page
    num = int(pages)
    if num <= 0:
        raise ValueError("Page must be >= 1")
    return [num]


@app.post("/extract")
async def extract(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    file_path: Optional[str] = Query(default=None, description="Local PDF path if not uploading"),
    mode: ProcessingMode = Query(default=ProcessingMode.FULL, description="Processing mode"),
    pages: Optional[str] = Query(default=None, description="Pages e.g. '1-3' or '1,3,5'"),
    include_ocr: bool = True,
    include_formulas: bool = True,
    include_grobid: bool = False,
    table_engine: TableEngine = Query(default=TableEngine.CAMELOT),
    formula_model: FormulaModel = Query(default=FormulaModel.LATEX_OCR),
    output_format: str = Query(default="json"),
    async_mode: bool = Query(default=False, description="Queue job asynchronously and return task id"),
):
    if not _processor or not _config:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    # Determine local file path
    tmp_path: Optional[Path] = None
    try:
        local_path: Optional[Path] = None
        if file is not None:
            # Save upload to a temporary file
            suffix = Path(file.filename or "uploaded.pdf").suffix or ".pdf"
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            Path(tmp).write_bytes(await file.read())
            local_path = Path(tmp)
            tmp_path = local_path
        elif file_path:
            local_path = Path(file_path)
            if not local_path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
        else:
            raise HTTPException(status_code=400, detail="Provide either file upload or file_path")

        # Build request
        req = ProcessingRequest(
            file_path=str(local_path),
            mode=mode,
            pages=_parse_pages(pages),
            output_format=output_format,
            include_ocr=include_ocr,
            include_formulas=include_formulas,
            include_grobid=include_grobid,
            table_engine=table_engine,
            formula_model=formula_model,
        )

        if async_mode:
            # Queue task and return id immediately
            if _rq_queue is not None and output_format in ("json", "docx"):
                # Enqueue on RQ
                job = _rq_queue.enqueue(
                    "pdf_mcp_server.worker:process_job",
                    str(local_path),
                    {
                        "mode": str(mode.value) if hasattr(mode, "value") else str(mode),
                        "pages": _parse_pages(pages),
                        "include_ocr": include_ocr,
                        "include_formulas": include_formulas,
                        "include_grobid": include_grobid,
                        "table_engine": table_engine.value if hasattr(table_engine, "value") else str(table_engine),
                        "formula_model": formula_model.value if hasattr(formula_model, "value") else str(formula_model),
                        "output_format": output_format,
                    },
                )
                return JSONResponse(jsonable_encoder({"task_id": job.get_id(), "status": "queued"}))
            else:
                task_id = f"task_{uuid.uuid4()}"
                _tasks[task_id] = {"status": "queued", "output_format": output_format}

                async def _run():
                    try:
                        _tasks[task_id] = {"status": "running"}
                        result = await _processor.process(req)
                        task_payload: Dict[str, Any] = {"status": "finished", "result": result.dict()}
                        if output_format == "docx":
                            out_dir = Path(os.getenv("PDF_MCP_DOCX_DIR", tempfile.gettempdir())) / "pdf_mcp_docx"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / f"export_{local_path.stem}.docx"
                            from .utils.docx_exporter import build_docx_from_pipeline
                            build_docx_from_pipeline(result.dict(), out_path)
                            task_payload["docx_path"] = str(out_path)
                        _tasks[task_id] = task_payload
                    except Exception as e:  # pylint: disable=broad-except
                        _tasks[task_id] = {"status": "error", "error": str(e)}
                    finally:
                        if tmp_path and tmp_path.exists():
                            try:
                                tmp_path.unlink(missing_ok=True)
                            except Exception:
                                pass

                background_tasks.add_task(_run)
                return JSONResponse(jsonable_encoder({"task_id": task_id, "status": "queued"}))

        # Synchronous processing
        result = await _processor.process(req)
        if output_format == "docx":
            # Export and return file
            out_dir = Path(os.getenv("PDF_MCP_DOCX_DIR", tempfile.gettempdir())) / "pdf_mcp_docx"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"export_{local_path.stem}.docx"
            build_docx_from_pipeline(result.dict(), out_path)
            return FileResponse(
                path=str(out_path),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename=out_path.name,
            )
        else:
            return JSONResponse(jsonable_encoder(result))

    finally:
        # Clean temporary upload if we processed synchronously
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/task/{task_id}")
async def task_status(task_id: str) -> JSONResponse:
    # Prefer RQ job if queue available
    if _rq_queue is not None:
        try:
            from rq.job import Job
            from redis import Redis
            conn = _rq_queue.connection  # type: ignore
            job = Job.fetch(task_id, connection=conn)
            status = job.get_status()
            payload: Dict[str, Any] = {"task_id": task_id, "status": status}
            if status == "finished":
                result = job.return_value or {}
                payload["result"] = result.get("result")
                if result.get("docx_path"):
                    payload["download_url"] = f"/download/{task_id}"
            elif status == "failed":
                payload["error"] = str(job.exc_info or "failed")
            return JSONResponse(jsonable_encoder(payload))
        except Exception:
            pass

    data = _tasks.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    # Attach download_url if docx is available
    if data.get("docx_path"):
        data = {**data, "download_url": f"/download/{task_id}"}
    return JSONResponse(jsonable_encoder(data))


@app.get("/download/{task_id}")
async def download_result(task_id: str):
    # RQ job case
    if _rq_queue is not None:
        try:
            from rq.job import Job
            job = Job.fetch(task_id, connection=_rq_queue.connection)  # type: ignore
            res = job.return_value or {}
            path = res.get("docx_path")
            if path and Path(path).exists():
                return FileResponse(
                    path=path,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    filename=Path(path).name,
                )
        except Exception:
            pass

    # In-memory case
    data = _tasks.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    path = data.get("docx_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(
        path=path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=Path(path).name,
    )


@app.get("/models")
async def models_info() -> Dict[str, Any]:
    # Minimal information; extend as models load in FormulaExtractor/TableExtractor
    return {
        "torch_cuda": bool(PDFProcessor.__module__),  # placeholder
    }
