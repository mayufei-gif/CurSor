"""主要的PDF处理器，协调所有处理任务。

本模块提供了中央PDFProcessor类，用于编排文本提取、表格提取、OCR和公式识别等功能。
该处理器作为整个PDF处理流程的核心控制器，负责：
- 初始化和管理各个子处理器
- 验证处理请求的有效性
- 制定处理计划和策略
- 协调执行各种处理任务
- 收集和整合处理结果
- 提供健康检查和统计信息

主要特性：
- 异步处理支持
- 模块化设计，易于扩展
- 完整的错误处理和日志记录
- 性能统计和监控
- 资源管理和清理
"""

# 标准库导入
import asyncio  # 异步编程支持
import logging  # 日志记录
import time     # 时间处理
from pathlib import Path  # 路径操作
from typing import Dict, Any, Optional  # 类型注解
import uuid     # 唯一标识符生成

# 数据模型导入
from ..models import (
    ProcessingRequest,    # 处理请求模型
    ProcessingResponse,   # 处理响应模型
    ProcessingContent,    # 处理内容模型
    ProcessingMetadata,   # 处理元数据模型
    PDFInfo,             # PDF信息模型
    ProcessingMode,      # 处理模式枚举
)
# 工具类导入
from ..utils.config import Config  # 配置管理
from ..utils.exceptions import PDFProcessingError, ValidationError  # 自定义异常
# 处理器导入
from .document_analyzer import DocumentAnalyzer    # 文档分析器
from .text_extractor import TextExtractor          # 文本提取器
from .table_extractor import TableExtractor        # 表格提取器
from .ocr_processor import OCRProcessor            # OCR处理器
from .formula_extractor import FormulaExtractor    # 公式提取器


class PDFProcessor:
    """主要的PDF处理器，协调所有处理任务。
    
    该类作为PDF处理系统的核心控制器，负责管理和协调各个子处理器的工作。
    它提供了完整的PDF处理流程，包括文档分析、文本提取、表格提取、
    OCR处理和公式识别等功能。
    
    主要职责：
    - 初始化和管理各个子处理器实例
    - 验证处理请求的合法性和可行性
    - 根据文档特性制定最优处理策略
    - 协调执行各种处理任务
    - 整合和返回处理结果
    - 提供系统健康检查和性能统计
    
    属性：
        config: 系统配置对象
        logger: 日志记录器
        document_analyzer: 文档分析器实例
        text_extractor: 文本提取器实例
        table_extractor: 表格提取器实例
        ocr_processor: OCR处理器实例
        formula_extractor: 公式提取器实例
        total_processed: 已处理文档总数
        total_errors: 处理错误总数
        start_time: 处理器启动时间
    """
    
    def __init__(self, config: Config):
        """初始化PDF处理器。
        
        设置处理器的基本配置和初始状态，创建日志记录器，
        初始化各个子处理器的占位符，并设置统计计数器。
        
        Args:
            config: 配置对象，包含所有处理器的配置参数
        """
        self.config = config  # 保存配置对象
        self.logger = logging.getLogger(__name__)  # 创建日志记录器
        
        # 初始化各个子处理器（延迟初始化）
        self.document_analyzer: Optional[DocumentAnalyzer] = None  # 文档分析器
        self.text_extractor: Optional[TextExtractor] = None       # 文本提取器
        self.table_extractor: Optional[TableExtractor] = None     # 表格提取器
        self.ocr_processor: Optional[OCRProcessor] = None         # OCR处理器
        self.formula_extractor: Optional[FormulaExtractor] = None # 公式提取器
        
        # 处理统计信息
        self.total_processed = 0  # 已处理文档总数
        self.total_errors = 0     # 处理错误总数
        self.start_time = time.time()  # 记录启动时间
    
    async def initialize(self):
        """初始化所有子处理器。
        
        按顺序初始化文档分析器、文本提取器、表格提取器、
        OCR处理器和公式提取器。如果任何一个处理器初始化失败，
        将抛出PDFProcessingError异常。
        
        Raises:
            PDFProcessingError: 当任何处理器初始化失败时抛出
        """
        self.logger.info("正在初始化PDF处理器...")
        
        try:
            # 初始化文档分析器
            self.document_analyzer = DocumentAnalyzer(self.config)
            await self.document_analyzer.initialize()
            
            # 初始化文本提取器
            self.text_extractor = TextExtractor(self.config)
            await self.text_extractor.initialize()
            
            # 初始化表格提取器
            self.table_extractor = TableExtractor(self.config)
            await self.table_extractor.initialize()
            
            # 初始化OCR处理器
            self.ocr_processor = OCRProcessor(self.config)
            await self.ocr_processor.initialize()
            
            # 初始化公式提取器
            self.formula_extractor = FormulaExtractor(self.config)
            await self.formula_extractor.initialize()
            
            self.logger.info("PDF处理器初始化成功")
            
        except Exception as e:
            self.logger.error(f"PDF处理器初始化失败: {e}")
            raise PDFProcessingError(f"初始化失败: {e}")
    
    async def cleanup(self):
        """清理资源。
        
        依次清理所有子处理器的资源，包括关闭文件句柄、
        释放内存、断开网络连接等。即使某个处理器清理失败，
        也会继续清理其他处理器，确保资源得到最大程度的释放。
        """
        self.logger.info("正在清理PDF处理器资源...")
        
        # 收集所有需要清理的处理器
        processors = [
            self.document_analyzer,  # 文档分析器
            self.text_extractor,     # 文本提取器
            self.table_extractor,    # 表格提取器
            self.ocr_processor,      # OCR处理器
            self.formula_extractor   # 公式提取器
        ]
        
        # 逐个清理处理器资源
        for processor in processors:
            if processor:  # 检查处理器是否已初始化
                try:
                    await processor.cleanup()  # 调用处理器的清理方法
                except Exception as e:
                    # 记录清理错误但不中断其他处理器的清理
                    self.logger.warning(f"处理器清理时发生错误: {e}")
        
        self.logger.info("PDF处理器资源清理完成")
    
    async def health_check(self) -> Dict[str, bool]:
        """检查所有处理器的健康状态。
        
        遍历所有子处理器，调用它们的健康检查方法，
        收集每个处理器的状态信息。如果处理器未初始化
        或健康检查失败，则标记为不健康。
        
        Returns:
            包含每个处理器健康状态的字典，键为处理器名称，值为布尔状态
        """
        health = {}  # 存储健康状态结果
        
        # 定义所有需要检查的处理器
        processors = {
            "document_analyzer": self.document_analyzer,    # 文档分析器
            "text_extractor": self.text_extractor,          # 文本提取器
            "table_extractor": self.table_extractor,        # 表格提取器
            "ocr_processor": self.ocr_processor,            # OCR处理器
            "formula_extractor": self.formula_extractor     # 公式提取器
        }
        
        # 逐个检查处理器健康状态
        for name, processor in processors.items():
            if processor:  # 检查处理器是否已初始化
                try:
                    # 调用处理器的健康检查方法
                    health[name] = await processor.health_check()
                except Exception:
                    # 健康检查失败，标记为不健康
                    health[name] = False
            else:
                # 处理器未初始化，标记为不健康
                health[name] = False
        
        return health
    
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """根据请求处理PDF文档。
        
        这是主要的处理入口点，负责协调整个PDF处理流程：
        1. 验证请求的有效性
        2. 获取文件路径
        3. 分析文档特性
        4. 制定处理计划
        5. 执行处理任务
        6. 收集结果和元数据
        7. 返回处理响应
        
        Args:
            request: 包含处理选项的处理请求对象
            
        Returns:
            包含处理结果的响应对象
            
        Raises:
            PDFProcessingError: 当处理过程中发生错误时抛出
            ValidationError: 当请求验证失败时抛出
        """
        start_time = time.time()  # 记录处理开始时间
        request_id = str(uuid.uuid4())  # 生成唯一的请求ID
        
        self.logger.info(f"开始处理请求 {request_id}")
        
        try:
            # 第一步：验证请求的有效性
            await self._validate_request(request)
            
            # 第二步：获取文件路径
            file_path = await self._get_file_path(request)
            
            # 第三步：分析文档特性
            pdf_info = await self.document_analyzer.analyze(file_path)
            
            # 第四步：制定处理策略
            processing_plan = await self._create_processing_plan(request, pdf_info)
            
            # 第五步：执行处理任务
            content = await self._execute_processing(file_path, processing_plan, request)
            
            # 第六步：创建处理元数据
            processing_time = time.time() - start_time  # 计算处理耗时
            metadata = ProcessingMetadata(
                processing_time=processing_time,
                engines_used=processing_plan["engines_used"],
                file_info=pdf_info
            )
            
            # 更新统计信息
            self.total_processed += 1
            
            self.logger.info(f"处理请求 {request_id} 完成，耗时 {processing_time:.2f}秒")
            
            # 返回成功响应
            return ProcessingResponse(
                status="success",
                content=content,
                metadata=metadata,
                request_id=request_id
            )
            
        except Exception as e:
            # 处理失败，更新错误统计
            self.total_errors += 1
            self.logger.error(f"处理请求 {request_id} 失败: {e}", exc_info=True)
            
            # 重新抛出已知异常类型
            if isinstance(e, (PDFProcessingError, ValidationError)):
                raise
            else:
                # 包装未知异常
                raise PDFProcessingError(f"处理失败: {e}")
    
    async def _validate_request(self, request: ProcessingRequest):
        """验证处理请求的有效性。
        
        检查请求中的各项参数是否合法，包括：
        - 文件源（路径或URL）是否提供
        - 文件是否存在且可访问
        - 文件大小是否超出限制
        - 页面范围是否有效
        
        Args:
            request: 要验证的处理请求对象
            
        Raises:
            ValidationError: 当请求参数无效时抛出
        """
        # 检查文件源是否提供
        if not request.file_path and not request.file_url:
            raise ValidationError("必须提供file_path或file_url中的一个")
        
        # 检查文件大小限制
        if request.file_path:
            file_path = Path(request.file_path)
            # 检查文件是否存在
            if not file_path.exists():
                raise ValidationError(f"文件不存在: {request.file_path}")
            
            # 检查文件大小
            file_size = file_path.stat().st_size
            if file_size > request.max_file_size:
                raise ValidationError(f"文件过大: {file_size} 字节 (最大: {request.max_file_size})")
        
        # 验证页面范围
        if request.pages:
            for page in request.pages:
                if page < 1:
                    raise ValidationError(f"无效的页面号: {page}")
    
    async def _get_file_path(self, request: ProcessingRequest) -> Path:
        """从请求中获取本地文件路径。
        
        如果请求包含本地文件路径，直接返回；
        如果请求包含URL，则下载文件到本地（待实现）。
        
        Args:
            request: 处理请求对象
            
        Returns:
            本地文件的路径对象
            
        Raises:
            ValidationError: 当无法访问文件时抛出
            NotImplementedError: 当使用URL下载功能时抛出（待实现）
        """
        if request.file_path:
            # 直接返回本地文件路径
            return Path(request.file_path)
        
        elif request.file_url:
            # 从URL下载文件（待实现）
            # TODO: 实现文件下载功能
            raise NotImplementedError("URL下载功能尚未实现")
        
        else:
            # 没有提供文件源
            raise ValidationError("未提供文件源")
    
    async def _create_processing_plan(self, request: ProcessingRequest, pdf_info: PDFInfo) -> Dict[str, Any]:
        """根据请求和文档分析结果创建处理计划。
        
        分析文档特性和用户需求，制定最优的处理策略，
        包括确定需要使用的引擎、处理步骤和执行顺序。
        
        Args:
            request: 处理请求对象
            pdf_info: PDF文档信息对象
            
        Returns:
            包含处理计划的字典，包含处理模式、引擎列表、步骤等信息
        """
        # 初始化处理计划
        plan = {
            "mode": request.mode,           # 处理模式
            "engines_used": [],            # 使用的引擎列表
            "steps": [],                   # 处理步骤列表
            "requires_ocr": False,         # 是否需要OCR
            "extract_text": False,         # 是否提取文本
            "extract_tables": False,       # 是否提取表格
            "extract_formulas": False,     # 是否提取公式
            "use_grobid": False           # 是否使用GROBID
        }
        
        # 判断是否需要OCR处理
        if request.include_ocr and (pdf_info.is_scanned or not pdf_info.has_text_layer):
            plan["requires_ocr"] = True
            plan["steps"].append("ocr")
            plan["engines_used"].append("ocrmypdf")
        
        # 根据处理模式确定处理步骤
        if request.mode in [ProcessingMode.TEXT, ProcessingMode.FULL]:
            plan["extract_text"] = True
            plan["steps"].append("text_extraction")
            plan["engines_used"].extend(["pymupdf", "pdfplumber"])
        
        if request.mode in [ProcessingMode.TABLES, ProcessingMode.FULL]:
            plan["extract_tables"] = True
            plan["steps"].append("table_extraction")
            plan["engines_used"].extend(["camelot", "tabula"])
        
        if request.mode in [ProcessingMode.FORMULAS, ProcessingMode.FULL] and request.include_formulas:
            plan["extract_formulas"] = True
            plan["steps"].append("formula_extraction")
            plan["engines_used"].append(str(request.formula_model))
        
        if request.include_grobid:
            plan["use_grobid"] = True
            plan["steps"].append("grobid_parsing")
            plan["engines_used"].append("grobid")
        
        return plan
    
    async def _execute_processing(self, file_path: Path, plan: Dict[str, Any], request: ProcessingRequest) -> ProcessingContent:
        """执行处理计划。
        
        按照制定的处理计划，依次执行各个处理步骤，
        包括OCR、文本提取、表格提取、公式提取和GROBID处理。
        
        Args:
            file_path: PDF文件路径
            plan: 处理计划字典
            request: 原始请求对象
            
        Returns:
            包含所有处理结果的内容对象
        """
        content = ProcessingContent()  # 创建内容容器
        
        # 如果需要，应用OCR处理
        if plan["requires_ocr"]:
            self.logger.info("正在对文档应用OCR处理")
            file_path = await self.ocr_processor.process(file_path, request)
        
        # 提取文本内容
        if plan["extract_text"]:
            self.logger.info("正在提取文本")
            content.text = await self.text_extractor.extract(file_path, request)
        
        # 提取表格数据
        if plan["extract_tables"]:
            self.logger.info("正在提取表格")
            content.tables = await self.table_extractor.extract(file_path, request)
        
        # 提取公式信息
        if plan["extract_formulas"]:
            self.logger.info("正在提取公式")
            content.formulas = await self.formula_extractor.extract(file_path, request)
        
        # GROBID学术文档处理
        if plan["use_grobid"]:
            self.logger.info("正在使用GROBID处理")
            # TODO: 实现GROBID处理功能
            pass
        
        return content
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息。
        
        返回处理器的运行统计数据，包括处理总数、错误数、
        成功率、运行时间和平均处理时间等指标。
        
        Returns:
            包含统计信息的字典
        """
        uptime = time.time() - self.start_time  # 计算运行时间
        
        return {
            "total_processed": self.total_processed,  # 已处理文档总数
            "total_errors": self.total_errors,        # 处理错误总数
            "success_rate": (self.total_processed - self.total_errors) / max(self.total_processed, 1),  # 成功率
            "uptime_seconds": uptime,                 # 运行时间（秒）
            "average_processing_time": uptime / max(self.total_processed, 1)  # 平均处理时间
        }