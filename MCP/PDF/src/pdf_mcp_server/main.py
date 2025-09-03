#!/usr/bin/env python3
"""
PDF-MCP Server Main Entry Point

This module provides the main entry point for the PDF-MCP server,
handling server initialization, tool registration, and request routing.

Author: PDF-MCP Team
License: MIT
"""

# 导入标准库模块
import asyncio  # 异步IO支持
import json  # JSON数据处理
import logging  # 日志记录
import os  # 操作系统接口
import time  # 时间处理
from contextlib import asynccontextmanager  # 异步上下文管理器
from typing import Dict, Any, Optional  # 类型提示
from pathlib import Path  # 路径处理

# 导入第三方库模块
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends  # FastAPI框架
from fastapi.middleware.cors import CORSMiddleware  # CORS中间件
from fastapi.responses import JSONResponse  # JSON响应
from mcp.server import Server  # MCP服务器
from mcp.types import Tool, TextContent  # MCP类型定义
import uvicorn  # ASGI服务器

# 导入项目内部模块
from .models import (
    ProcessingRequest,  # 处理请求模型
    ProcessingResponse,  # 处理响应模型
    ErrorResponse,  # 错误响应模型
    HealthResponse,  # 健康检查响应模型
    ProcessingMode,  # 处理模式枚举
)
from .processors import PDFProcessor  # PDF处理器
from .utils.config import Config  # 配置管理
from .utils.logger import setup_logging  # 日志设置
from .utils.exceptions import PDFProcessingError, ValidationError  # 自定义异常

# 全局变量定义
# 存储配置对象，初始为None
config: Optional[Config] = None
# 存储PDF处理器实例，初始为None
pdf_processor: Optional[PDFProcessor] = None
# 记录服务器启动时间
start_time = time.time()


# 定义应用生命周期管理器，用于处理应用启动和关闭时的操作
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    管理应用程序的生命周期，包括启动时的初始化和关闭时的清理工作。
    """
    global config, pdf_processor  # 声明使用全局变量
    
    # 启动阶段
    logger = logging.getLogger(__name__)  # 获取日志记录器，用于记录运行时信息
    logger.info("Starting PDF-MCP Server...")  # 记录服务器启动的日志信息
    
    try:
        # 加载配置文件，初始化配置对象
        config = Config()  # 创建Config实例，加载配置
        
        # 设置日志记录级别和日志文件路径
        setup_logging(config.log_level, config.log_file)  # 配置日志系统
        
        # 初始化PDF处理器
        pdf_processor = PDFProcessor(config)  # 创建PDFProcessor实例
        await pdf_processor.initialize()  # 异步初始化PDF处理器
        
        logger.info("PDF-MCP Server started successfully")  # 记录服务器成功启动的日志信息
        
    except Exception as e:
        # 捕获启动过程中的异常并记录错误日志
        logger.error(f"Failed to start server: {e}")  # 记录启动失败的日志信息
        raise  # 重新抛出捕获的异常，终止程序启动
    
    yield  # 应用运行期间暂停此处，等待应用关闭
    
    # 关闭阶段
    logger.info("Shutting down PDF-MCP Server...")  # 记录服务器关闭开始的日志信息
    if pdf_processor:
        await pdf_processor.cleanup()  # 清理PDF处理器资源，释放相关资源
    logger.info("PDF-MCP Server shutdown complete")  # 记录服务器关闭完成的日志信息


# 创建FastAPI应用实例
app = FastAPI(
    title="PDF-MCP Server",  # API标题
    description="A comprehensive PDF processing MCP server",  # API描述
    version="0.1.0",  # API版本
    lifespan=lifespan,  # 关联生命周期管理器
)

# 添加CORS中间件，处理跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境中应适当配置）
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 创建MCP服务器实例
mcp_server = Server("pdf-mcp-server")


# 定义依赖项函数，用于获取PDF处理器实例
def get_pdf_processor() -> PDFProcessor:
    """Get the PDF processor instance.
    
    获取PDF处理器实例的依赖注入函数。
    如果处理器未初始化，则抛出HTTP 503错误。
    """
    if pdf_processor is None:  # 检查PDF处理器是否已初始化
        # 如果PDF处理器未初始化，抛出服务不可用异常
        raise HTTPException(status_code=503, detail="PDF processor not initialized")
    return pdf_processor  # 返回初始化后的PDF处理器实例


# 定义异常处理器，处理PDF处理错误
@app.exception_handler(PDFProcessingError)
async def pdf_processing_exception_handler(request, exc: PDFProcessingError):
    """Handle PDF processing errors.
    
    处理PDF处理过程中发生的错误，返回统一的错误响应格式。
    """
    return JSONResponse(
        status_code=422,  # 请求格式正确但语义错误
        content=ErrorResponse(
            error_code="PDF_PROCESSING_ERROR",  # 错误代码
            message=str(exc),  # 错误消息
            details={"type": type(exc).__name__}  # 错误详情
        ).dict()
    )


# 定义异常处理器，处理验证错误
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation errors.
    
    处理输入验证过程中发生的错误，返回统一的错误响应格式。
    """
    return JSONResponse(
        status_code=400,  # 客户端错误
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",  # 验证错误代码
            message=str(exc),  # 错误消息
            details=exc.details if hasattr(exc, 'details') else {}  # 错误详情
        ).dict()
    )


# 定义健康检查端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    健康检查端点，用于监控服务器运行状态。
    返回服务器运行时间、状态和时间戳等信息。
    """
    return HealthResponse(
        status="healthy",  # 服务状态为健康
        uptime=time.time() - start_time,  # 计算服务器运行时间（当前时间减去启动时间）
        timestamp=datetime.utcnow().isoformat()  # 获取当前UTC时间并格式化为ISO格式
    )


# 定义PDF处理端点
@app.post("/api/v1/process", response_model=ProcessingResponse)
async def process_pdf(
    request: ProcessingRequest,  # 处理请求参数
    processor: PDFProcessor = Depends(get_pdf_processor)  # 依赖注入PDF处理器
):
    """Process PDF with specified mode and options.
    
    根据指定的模式和选项处理PDF文件。
    这是主要的PDF处理端点，支持多种处理模式。
    """
    try:
        # 调用处理器执行PDF处理
        result = await processor.process(request)
        return result  # 返回处理结果
    except Exception as e:
        logger = logging.getLogger(__name__)  # 获取日志记录器
        logger.error(f"Processing failed: {e}", exc_info=True)  # 记录错误日志
        raise HTTPException(status_code=500, detail=str(e))  # 抛出服务器内部错误


# 定义文件上传处理端点
@app.post("/api/v1/upload", response_model=ProcessingResponse)
async def upload_and_process(
    file: UploadFile = File(...),  # 上传的文件
    mode: ProcessingMode = Form(ProcessingMode.FULL),  # 处理模式，默认为完整处理
    include_ocr: bool = Form(True),  # 是否包含OCR处理，默认为True
    include_formulas: bool = Form(False),  # 是否包含公式识别，默认为False
    include_grobid: bool = Form(False),  # 是否包含GROBID处理，默认为False
    processor: PDFProcessor = Depends(get_pdf_processor)  # 依赖注入PDF处理器
):
    """Upload and process PDF file.
    
    上传并处理PDF文件的端点。
    先将上传的文件保存到临时位置，然后进行处理，最后清理临时文件。
    """
    # 检查文件类型，只允许PDF文件
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # 创建临时目录用于存储上传文件
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(exist_ok=True)  # 如果目录不存在则创建
    
    # 生成临时文件路径，包含时间戳和原文件名
    temp_file = temp_dir / f"upload_{int(time.time())}_{file.filename}"
    
    try:
        # 保存上传的文件内容到临时文件
        with open(temp_file, "wb") as f:
            content = await file.read()  # 异步读取文件内容
            f.write(content)  # 写入临时文件
        
        # 创建处理请求对象
        request = ProcessingRequest(
            file_path=str(temp_file),  # 临时文件路径
            mode=mode,  # 处理模式
            include_ocr=include_ocr,  # 是否包含OCR
            include_formulas=include_formulas,  # 是否包含公式识别
            include_grobid=include_grobid  # 是否包含GROBID处理
        )
        
        # 调用处理器执行PDF处理
        result = await processor.process(request)
        return result  # 返回处理结果
        
    finally:
        # 清理临时文件，无论处理成功与否都会执行
        if temp_file.exists():
            temp_file.unlink()  # 删除临时文件


# MCP工具定义部分
# 注册read_text工具到MCP服务器
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools.
    
    列出所有可用的MCP工具。
    这些工具可以通过MCP协议调用，提供各种PDF处理功能。
    """
    return [
        Tool(
            name="read_text",  # 工具名称
            description="Extract text content from PDF",  # 工具描述
            inputSchema={
                "type": "object",  # 输入参数模式类型
                "properties": {  # 参数属性定义
                    "file_path": {"type": "string", "description": "Path to PDF file"},  # 文件路径参数
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 页面参数
                    "include_bbox": {"type": "boolean", "description": "Include bounding box information"}  # 是否包含边界框信息
                },
                "required": ["file_path"]  # 必需参数
            }
        ),
        Tool(
            name="extract_tables",  # 工具名称
            description="Extract tables from PDF",  # 工具描述
            inputSchema={
                "type": "object",  # 输入参数模式类型
                "properties": {  # 参数属性定义
                    "file_path": {"type": "string", "description": "Path to PDF file"},  # 文件路径参数
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 页面参数
                    "engine": {"type": "string", "enum": ["camelot", "tabula", "pdfplumber"], "description": "Table extraction engine"},  # 表格提取引擎
                    "output_format": {"type": "string", "enum": ["json", "csv", "dataframe"], "description": "Output format"}  # 输出格式
                },
                "required": ["file_path"]  # 必需参数
            }
        ),
        Tool(
            name="extract_formulas",  # 工具名称
            description="Extract mathematical formulas from PDF",  # 工具描述
            inputSchema={
                "type": "object",  # 输入参数模式类型
                "properties": {  # 参数属性定义
                    "file_path": {"type": "string", "description": "Path to PDF file"},  # 文件路径参数
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 页面参数
                    "engine": {"type": "string", "enum": ["pix2tex", "latex-ocr"], "description": "Formula extraction engine"},  # 公式提取引擎
                    "output_format": {"type": "string", "enum": ["latex", "mathml"], "description": "Output format"}  # 输出格式
                },
                "required": ["file_path"]  # 必需参数
            }
        ),
        Tool(
            name="process_ocr",  # 工具名称
            description="Perform OCR on PDF",  # 工具描述
            inputSchema={
                "type": "object",  # 输入参数模式类型
                "properties": {  # 参数属性定义
                    "file_path": {"type": "string", "description": "Path to PDF file"},  # 文件路径参数
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 页面参数
                    "language": {"type": "string", "description": "OCR language"},  # OCR语言
                    "output_format": {"type": "string", "enum": ["text", "hocr", "pdf"], "description": "Output format"}  # 输出格式
                },
                "required": ["file_path"]  # 必需参数
            }
        ),
        Tool(
            name="analyze_pdf",  # 工具名称
            description="Analyze PDF structure and content",  # 工具描述
            inputSchema={
                "type": "object",  # 输入参数模式类型
                "properties": {  # 参数属性定义
                    "file_path": {"type": "string", "description": "Path to PDF file"},  # 文件路径参数
                    "analysis_type": {"type": "string", "enum": ["basic", "detailed"], "description": "Analysis depth"}  # 分析深度
                },
                "required": ["file_path"]  # 必需参数
            }
        ),
        Tool(
            name="full_pipeline",  # 工具名称
            description="Run complete PDF processing pipeline",  # 工具描述
            inputSchema={
                "type": "object",  # 输入参数模式类型
                "properties": {  # 参数属性定义
                    "file_path": {"type": "string", "description": "Path to PDF file"},  # 文件路径参数
                    "include_ocr": {"type": "boolean", "description": "Include OCR processing"},  # 是否包含OCR
                    "include_formulas": {"type": "boolean", "description": "Include formula detection"},  # 是否包含公式识别
                    "include_grobid": {"type": "boolean", "description": "Include GROBID processing"}  # 是否包含GROBID处理
                },
                "required": ["file_path"]  # 必需参数
            }
        )
    ]


# 主函数，用于直接运行服务器
def main():
    """Main entry point for running the server.
    
    主函数，解析命令行参数并启动服务器。
    支持指定主机、端口、重载和工作进程数等参数。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="PDF-MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")  # 主机地址参数
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")  # 端口参数
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")  # 重载参数
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")  # 工作进程数参数
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用uvicorn运行FastAPI应用
    uvicorn.run(
        "pdf_mcp_server.main:app",  # 应用路径
        host=args.host,  # 主机地址
        port=args.port,  # 端口
        reload=args.reload,  # 重载选项
        workers=args.workers  # 工作进程数
    )


# 如果作为主模块运行，则执行主函数
if __name__ == "__main__":
    main()