#!/usr/bin/env python3  # 指定解释器（类 Unix 环境），Windows 无影响
# -*- coding: utf-8 -*-  # 源文件编码声明，保障中文注释在部分环境下正常识别
"""RQ worker and task helpers for PDF-MCP server.

该模块提供异步任务相关的辅助：
- `process_job`: 将 PDF 处理作业执行（并可导出 DOCX）
- `get_queue`/`get_redis`: 获取 Redis 与 RQ 队列
- `main`: 简易 Worker 启动入口
"""

from __future__ import annotations  # 推迟类型注解求值，避免循环依赖/提升导入性能

# 标准库导入
import os  # 访问环境变量（例如 REDIS_URL）
import json  # 序列化处理结果
import tempfile  # 系统临时目录（导出 DOCX 的默认位置）
from pathlib import Path  # 跨平台路径处理
from typing import Any, Dict, Optional  # 类型注解：任意类型/字典/可选

# 第三方依赖
from redis import Redis  # Redis 客户端
from rq import Queue, Worker  # RQ 队列与工作进程

# 项目内模块
from .utils.config import Config  # 配置加载工具
from .processors.pdf_processor import PDFProcessor  # PDF 主处理器
from .utils.docx_exporter import build_docx_from_pipeline  # DOCX 导出工具


def get_redis() -> Optional[Redis]:  # 获取 Redis 连接；失败则返回 None
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")  # 从环境变量取 URL，默认本地实例
    try:
        conn = Redis.from_url(url)  # 构造连接
        conn.ping()  # 健康检查
        return conn  # 返回可用连接
    except Exception:  # 容错：连接失败时返回 None（上层可采用降级逻辑）
        return None


def get_queue(name: str = "pdf_tasks") -> Optional[Queue]:  # 获取指定名称的 RQ 队列
    conn = get_redis()  # 先获取 Redis 连接
    if not conn:  # 无连接则无法创建队列
        return None
    return Queue(name, connection=conn)  # 返回队列对象


def process_job(
    file_path: str,  # 待处理 PDF 文件路径
    options: Dict[str, Any],  # 处理选项；兼容 ProcessingRequest 字段，支持 output_format
) -> Dict[str, Any]:
    """运行完整处理流水线，并可选导出 DOCX。

    Args:
        file_path: PDF 文件路径
        options: 处理选项（与 ProcessingRequest 对齐）+ {"output_format": "json|docx"}

    Returns:
        包含处理状态、JSON 结果，及可选 DOCX 路径的字典
    """
    config = Config.load()  # 加载配置（默认路径/环境）
    processor = PDFProcessor(config)  # 主处理器实例

    # 构造 ProcessingRequest 的参数字典（仅保留非 None）
    req_kwargs = {
        "file_path": file_path,
        "mode": options.get("mode"),
        "pages": options.get("pages"),
        "include_ocr": options.get("include_ocr", True),
        "include_formulas": options.get("include_formulas", True),
        "include_grobid": options.get("include_grobid", False),
        "table_engine": options.get("table_engine"),
        "formula_model": options.get("formula_model"),
    }

    import asyncio  # 局部导入，避免同步环境提前加载

    async def _run():  # 内部协程：初始化 -> 处理 -> 清理
        await processor.initialize()
        from .models import ProcessingRequest  # 延迟导入，避免循环依赖
        request = ProcessingRequest(**{k: v for k, v in req_kwargs.items() if v is not None})
        result = await processor.process(request)
        await processor.cleanup()
        return result

    result = asyncio.run(_run())  # 在同步上下文运行协程
    res_dict = result.dict()  # Pydantic 模型转字典

    out: Dict[str, Any] = {"status": "done", "result": res_dict}  # 基础返回结构

    # 可选：将流水线结果导出为 DOCX
    if options.get("output_format") == "docx":
        temp_dir = Path(tempfile.gettempdir()) / "pdf_mcp_docx"  # 系统临时根目录下的子目录
        temp_dir.mkdir(parents=True, exist_ok=True)  # 保证目录存在
        out_path = temp_dir / f"export_{Path(file_path).stem}.docx"  # 输出 DOCX 路径
        try:
            build_docx_from_pipeline(res_dict, out_path)  # 依据流水线结果生成 DOCX
            out["docx_path"] = str(out_path)
        except Exception as e:  # 导出失败不影响 JSON 结果
            out.setdefault("warnings", []).append(f"DOCX export failed: {e}")

    return out  # 返回结果


def main():  # Worker 启动入口：连接队列并开始工作
    q = get_queue()
    if not q:  # Redis 未连接或不可用
        raise SystemExit("Redis connection not available; set REDIS_URL")
    Worker([q]).work()  # 监听并处理任务


if __name__ == "__main__":  # 仅在直接运行时启用 Worker
    main()

