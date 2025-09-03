# -*- coding: utf-8 -*-
"""Data models for PDF-MCP Server.
PDF-MCP服务器数据模型定义

This module defines all Pydantic models used for API requests and responses.

本模块定义了PDF-MCP服务器中使用的所有Pydantic数据模型，包括：
- 枚举类型：处理模式、输出格式、公式模型、表格引擎等
- 基础数据结构：边界框、PDF信息、页面文本等
- 提取结果模型：文本、表格、公式提取结果
- 请求和响应模型：处理请求、处理响应、错误响应等
- 验证器：确保数据的有效性和一致性

所有模型都基于Pydantic，提供自动验证、序列化和文档生成功能。
"""

# 标准库导入
from typing import List, Optional, Dict, Any, Union  # 类型提示支持
from enum import Enum  # 枚举类型支持
from datetime import datetime  # 日期时间处理

# 第三方库导入
from pydantic import BaseModel, Field, field_validator  # 数据验证和序列化框架


class ProcessingMode(str, Enum):
    """可用的PDF处理模式枚举
    
    定义了PDF处理器支持的不同处理模式，每种模式对应不同的功能组合。
    """
    TEXT = "read_text"  # 纯文本提取模式
    TABLES = "extract_tables"  # 表格提取模式
    FORMULAS = "extract_formulas"  # 公式提取模式
    FULL = "full_pipeline"  # 完整流水线模式（包含所有功能）


class OutputFormat(str, Enum):
    """表格数据的可用输出格式枚举
    
    定义了表格提取结果的不同输出格式选项。
    """
    JSON = "json"  # JSON格式输出
    CSV = "csv"  # CSV格式输出
    DATAFRAME = "dataframe"  # Pandas DataFrame格式输出


class FormulaModel(str, Enum):
    """可用的数学公式识别模型枚举
    
    定义了支持的不同公式识别模型，每个模型有不同的准确率和性能特点。
    """
    PIX2TEX = "pix2tex"  # Pix2Tex模型（基于图像到LaTeX转换）
    LATEX_OCR = "latex-ocr"  # LaTeX OCR模型
    TEXIFY = "texify"  # Texify模型（Microsoft开发）


class TableEngine(str, Enum):
    """可用的表格提取引擎枚举
    
    定义了支持的不同表格提取引擎，每个引擎适用于不同类型的PDF表格。
    """
    CAMELOT = "camelot"  # Camelot引擎（适用于格线表格）
    TABULA = "tabula"  # Tabula引擎（适用于流式表格）
    PDFPLUMBER = "pdfplumber"  # PDFPlumber引擎（通用表格提取）


class BoundingBox(BaseModel):
    """边界框坐标模型
    
    定义PDF中元素（文本、表格、公式等）的矩形边界框坐标。
    使用PDF坐标系统，其中(0,0)位于页面左下角。
    
    Attributes:
        x0: 左边界坐标
        y0: 下边界坐标  
        x1: 右边界坐标
        y1: 上边界坐标
        
    Note:
        坐标值必须满足 x1 > x0 且 y1 > y0 的约束条件
    """
    x0: float = Field(..., description="左边界坐标")
    y0: float = Field(..., description="下边界坐标")
    x1: float = Field(..., description="右边界坐标")
    y1: float = Field(..., description="上边界坐标")

    @field_validator('x1')
    @classmethod
    def x1_greater_than_x0(cls, v, info):
        """验证右边界坐标必须大于左边界坐标"""
        if 'x0' in info.data and v <= info.data['x0']:
            raise ValueError('x1 must be greater than x0')
        return v

    @field_validator('y1')
    @classmethod
    def y1_greater_than_y0(cls, v, info):
        """验证上边界坐标必须大于下边界坐标"""
        if 'y0' in info.data and v <= info.data['y0']:
            raise ValueError('y1 must be greater than y0')
        return v


class PDFInfo(BaseModel):
    """PDF文件信息模型
    
    包含PDF文件的基本信息和元数据，用于描述文档的属性和特征。
    
    Attributes:
        filename: PDF文件名
        pages: 总页数
        file_size: 文件大小（字节）
        is_scanned: 是否为扫描版PDF
        has_text_layer: 是否包含文本层
        creation_date: 创建日期
        modification_date: 修改日期
        title: 文档标题
        author: 作者
        subject: 主题
        keywords: 关键词
    """
    filename: str = Field(..., description="PDF文件名")
    pages: int = Field(..., description="总页数")
    file_size: int = Field(..., description="文件大小（字节）")
    is_scanned: bool = Field(..., description="是否为扫描版PDF")
    has_text_layer: bool = Field(..., description="是否包含文本层")
    creation_date: Optional[datetime] = Field(None, description="PDF创建日期")
    modification_date: Optional[datetime] = Field(None, description="PDF修改日期")
    title: Optional[str] = Field(None, description="PDF标题")
    author: Optional[str] = Field(None, description="PDF作者")
    subject: Optional[str] = Field(None, description="PDF主题")
    keywords: Optional[str] = Field(None, description="PDF关键词")


class PageText(BaseModel):
    """单页文本内容模型
    
    表示PDF单个页面的文本提取结果，包含文本内容和相关元数据。
    
    Attributes:
        page: 页码（从1开始）
        text: 提取的文本内容
        bbox: 文本边界框列表（可选）
        confidence: OCR置信度分数（可选）
        language: 检测到的语言（可选）
    """
    page: int = Field(..., description="页码（从1开始）")
    text: str = Field(..., description="提取的文本内容")
    bbox: Optional[List[BoundingBox]] = Field(None, description="文本边界框列表")
    confidence: Optional[float] = Field(None, description="OCR置信度分数")
    language: Optional[str] = Field(None, description="检测到的语言")


class TextExtractionResult(BaseModel):
    """文本提取结果模型
    
    包含完整的文本提取结果，包括全文、分页内容和统计信息。
    
    Attributes:
        full_text: 完整提取的文本
        pages: 分页文本内容列表
        word_count: 总词数
        character_count: 总字符数
        extraction_method: 使用的提取方法
        processing_time: 处理时间（秒）
    """
    full_text: str = Field(..., description="完整提取的文本")
    pages: List[PageText] = Field(..., description="分页文本内容列表")
    word_count: int = Field(..., description="总词数")
    character_count: int = Field(..., description="总字符数")
    extraction_method: str = Field(..., description="使用的提取方法")
    processing_time: float = Field(..., description="处理时间（秒）")


class TableData(BaseModel):
    """表格数据结构模型
    
    表示从PDF中提取的单个表格的完整信息，包括数据内容和元数据。
    
    Attributes:
        page: 包含表格的页码
        table_id: 页面内表格的标识符
        bbox: 表格边界框
        data: 表格数据（二维数组）
        headers: 表格标题行（可选）
        confidence: 提取置信度分数
        engine: 使用的提取引擎
        rows: 行数
        columns: 列数
    """
    page: int = Field(..., description="包含表格的页码")
    table_id: int = Field(..., description="页面内表格的标识符")
    bbox: BoundingBox = Field(..., description="表格边界框")
    data: List[List[str]] = Field(..., description="表格数据（二维数组）")
    headers: Optional[List[str]] = Field(None, description="表格标题行")
    confidence: float = Field(..., description="提取置信度分数")
    engine: str = Field(..., description="使用的提取引擎")
    rows: int = Field(..., description="行数")
    columns: int = Field(..., description="列数")


class TableExtractionResult(BaseModel):
    """表格提取结果模型
    
    包含完整的表格提取结果，包括所有提取的表格和统计信息。
    
    Attributes:
        tables: 提取的表格列表
        total_tables: 发现的表格总数
        extraction_method: 主要使用的方法
        fallback_used: 是否使用了备用方法
        processing_time: 处理时间（秒）
    """
    tables: List[TableData] = Field(..., description="提取的表格列表")
    total_tables: int = Field(..., description="发现的表格总数")
    extraction_method: str = Field(..., description="主要使用的方法")
    fallback_used: bool = Field(False, description="是否使用了备用方法")
    processing_time: float = Field(..., description="处理时间（秒）")


class FormulaData(BaseModel):
    """公式数据结构模型
    
    表示从PDF中提取的单个数学公式的完整信息，包括多种格式表示和元数据。
    
    Attributes:
        page: 包含公式的页码
        formula_id: 页面内公式的标识符
        bbox: 公式边界框
        latex: LaTeX格式表示
        confidence: 识别置信度分数
        model: 使用的识别模型
        image_path: 提取的公式图像路径（可选）
        raw_text: 原始OCR文本（可选）
    """
    page: int = Field(..., description="包含公式的页码")
    formula_id: int = Field(..., description="页面内公式的标识符")
    bbox: BoundingBox = Field(..., description="公式边界框")
    latex: str = Field(..., description="LaTeX格式表示")
    confidence: float = Field(..., description="识别置信度分数")
    model: str = Field(..., description="使用的识别模型")
    image_path: Optional[str] = Field(None, description="提取的公式图像路径")
    raw_text: Optional[str] = Field(None, description="原始OCR文本（如果可用）")


class FormulaExtractionResult(BaseModel):
    """公式提取结果模型
    
    包含完整的公式提取结果，包括所有提取的公式和统计信息。
    
    Attributes:
        formulas: 提取的公式列表
        total_formulas: 发现的公式总数
        model_used: 主要使用的模型
        processing_time: 处理时间（秒）
    """
    formulas: List[FormulaData] = Field(..., description="提取的公式列表")
    total_formulas: int = Field(..., description="发现的公式总数")
    model_used: str = Field(..., description="主要使用的模型")
    processing_time: float = Field(..., description="处理时间（秒）")


class GrobidResult(BaseModel):
    """GROBID解析结果模型
    
    包含GROBID学术文档解析的完整结果，提取文档的结构化信息。
    
    Attributes:
        title: 文档标题
        authors: 作者列表
        abstract: 文档摘要
        sections: 文档章节列表
        references: 参考文献列表
        keywords: 关键词列表
        doi: DOI标识符（如果可用）
        tei_xml: 完整的TEI XML输出
    """
    title: Optional[str] = Field(None, description="文档标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    abstract: Optional[str] = Field(None, description="文档摘要")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="文档章节列表")
    references: List[Dict[str, Any]] = Field(default_factory=list, description="参考文献列表")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    doi: Optional[str] = Field(None, description="DOI标识符（如果可用）")
    tei_xml: Optional[str] = Field(None, description="完整的TEI XML输出")


class ProcessingMetadata(BaseModel):
    """处理元数据和统计信息模型
    
    包含PDF处理过程的元数据信息，用于跟踪处理状态、性能和统计数据。
    
    Attributes:
        processing_time: 总处理时间（秒）
        engines_used: 使用的引擎/模型列表
        file_info: PDF文件信息
        timestamp: 处理时间戳
        version: 服务器版本
        errors: 遇到的非致命错误列表
        warnings: 生成的警告列表
    """
    processing_time: float = Field(..., description="总处理时间（秒）")
    engines_used: List[str] = Field(..., description="使用的引擎/模型列表")
    file_info: PDFInfo = Field(..., description="PDF文件信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="处理时间戳")
    version: str = Field("0.1.0", description="服务器版本")
    errors: List[str] = Field(default_factory=list, description="遇到的非致命错误列表")
    warnings: List[str] = Field(default_factory=list, description="生成的警告列表")


class ProcessingContent(BaseModel):
    """处理结果的主要内容容器模型
    
    包含PDF处理的所有内容结果，根据处理模式可能包含不同类型的提取结果。
    
    Attributes:
        text: 文本提取结果（可选）
        tables: 表格提取结果（可选）
        formulas: 公式提取结果（可选）
        grobid: GROBID解析结果（可选）
    """
    text: Optional[TextExtractionResult] = Field(None, description="文本提取结果")
    tables: Optional[TableExtractionResult] = Field(None, description="表格提取结果")
    formulas: Optional[FormulaExtractionResult] = Field(None, description="公式提取结果")
    grobid: Optional[GrobidResult] = Field(None, description="GROBID解析结果")


class ProcessingResponse(BaseModel):
    """完整处理响应模型
    
    包含PDF处理的完整响应，包括状态、内容、元数据和请求标识符。
    
    Attributes:
        status: 处理状态
        content: 提取的内容
        metadata: 处理元数据
        request_id: 请求标识符（可选）
    """
    status: str = Field(..., description="处理状态")
    content: ProcessingContent = Field(..., description="提取的内容")
    metadata: ProcessingMetadata = Field(..., description="处理元数据")
    request_id: Optional[str] = Field(None, description="请求标识符")


class ProcessingRequest(BaseModel):
    """PDF处理请求模型
    
    定义PDF处理请求的所有参数和选项，包括文件来源、处理模式和各种提取选项。
    支持本地文件和远程URL两种输入方式，提供灵活的处理配置。
    
    Attributes:
        file_path: 本地文件路径
        file_url: 远程文件URL
        mode: 处理模式
        pages: 要处理的特定页面列表
        include_bbox: 是否包含边界框信息
        table_engine: 主要表格提取引擎
        table_output_format: 表格输出格式
        include_ocr: 是否在需要时应用OCR
        ocr_language: OCR语言代码
        ocr_dpi: OCR DPI设置
        include_formulas: 是否提取数学公式
        formula_model: 公式识别模型
        formula_confidence_threshold: 公式的最小置信度阈值
        include_grobid: 是否使用GROBID进行学术文档解析
        max_file_size: 最大文件大小限制
        timeout: 处理超时时间
    """
    file_path: Optional[str] = Field(None, description="本地文件路径")
    file_url: Optional[str] = Field(None, description="远程文件URL")
    mode: ProcessingMode = Field(ProcessingMode.FULL, description="处理模式")
    pages: Optional[List[int]] = Field(None, description="要处理的特定页面列表")
    
    # 文本提取选项
    include_bbox: bool = Field(False, description="是否包含边界框信息")
    
    # 表格提取选项
    table_engine: TableEngine = Field(TableEngine.CAMELOT, description="主要表格提取引擎")
    table_output_format: OutputFormat = Field(OutputFormat.JSON, description="表格输出格式")
    
    # OCR选项
    include_ocr: bool = Field(True, description="是否在需要时应用OCR")
    ocr_language: str = Field("eng", description="OCR语言代码")
    ocr_dpi: int = Field(300, description="OCR DPI设置")
    
    # 公式识别选项
    include_formulas: bool = Field(False, description="是否提取数学公式")
    formula_model: FormulaModel = Field(FormulaModel.PIX2TEX, description="公式识别模型")
    formula_confidence_threshold: float = Field(0.7, description="公式的最小置信度阈值")
    
    # GROBID选项
    include_grobid: bool = Field(False, description="是否使用GROBID进行学术文档解析")
    
    # 通用选项
    max_file_size: int = Field(100 * 1024 * 1024, description="最大文件大小（字节）")
    timeout: int = Field(300, description="处理超时时间（秒）")
    
    @field_validator('file_path', 'file_url')
    @classmethod
    def at_least_one_source(cls, v, info):
        """验证至少提供一个文件来源
        
        确保用户至少提供了本地文件路径或远程文件URL中的一个。
        
        Args:
            v: 当前字段值
            info: 验证上下文信息
            
        Returns:
            验证后的字段值
            
        Raises:
            ValueError: 当两个文件来源都未提供时
        """
        if not v and not info.data.get('file_path') and not info.data.get('file_url'):
            raise ValueError('必须提供file_path或file_url中的至少一个')
        return v

    @field_validator('pages')
    @classmethod
    def pages_positive(cls, v):
        """验证页码必须为正数
        
        确保所有指定的页码都是正整数（从1开始）。
        
        Args:
            v: 页码列表
            
        Returns:
            验证后的页码列表
            
        Raises:
            ValueError: 当页码不是正数时
        """
        if v is not None:
            for page in v:
                if page < 1:
                    raise ValueError('页码必须为正数')
        return v

    @field_validator('formula_confidence_threshold')
    @classmethod
    def confidence_range(cls, v):
        """验证置信度阈值范围
        
        确保置信度阈值在有效范围[0.0, 1.0]内。
        
        Args:
            v: 置信度阈值
            
        Returns:
            验证后的置信度阈值
            
        Raises:
            ValueError: 当置信度阈值超出有效范围时
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError('置信度阈值必须在0.0到1.0之间')
        return v


class ErrorResponse(BaseModel):
    """错误响应模型
    
    定义API错误响应的标准格式，包含错误状态、代码、消息和详细信息。
    用于统一处理和返回各种类型的错误信息。
    
    Attributes:
        status: 响应状态（固定为"error"）
        error_code: 错误代码
        message: 错误消息
        details: 额外的错误详细信息
        timestamp: 错误发生时间戳
        request_id: 请求标识符
    """
    status: str = Field("error", description="响应状态")
    error_code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="额外的错误详细信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间戳")
    request_id: Optional[str] = Field(None, description="请求标识符")


class HealthResponse(BaseModel):
    """健康检查响应模型
    
    定义服务健康检查的响应格式，包含服务状态、版本信息和依赖状态。
    用于监控服务的运行状态和可用性。
    
    Attributes:
        status: 服务状态（通常为"healthy"）
        version: 服务版本号
        timestamp: 检查时间戳
        dependencies: 依赖服务状态字典
        uptime: 服务运行时间（秒）
    """
    status: str = Field("healthy", description="服务状态")
    version: str = Field("0.1.0", description="服务版本")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间戳")
    dependencies: Dict[str, bool] = Field(..., description="依赖状态")
    uptime: float = Field(..., description="服务运行时间（秒）")