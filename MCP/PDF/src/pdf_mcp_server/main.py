#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF-MCP Server Main Entry Point
PDF-MCP鏈嶅姟鍣ㄤ富鍏ュ彛鐐?
This module provides the main entry point for the PDF-MCP server,
handling server initialization, tool registration, and request routing.
鏈ā鍧楁彁渚汸DF-MCP鏈嶅姟鍣ㄧ殑涓诲叆鍙ｇ偣锛屽鐞嗘湇鍔″櫒鍒濆鍖栥€佸伐鍏锋敞鍐屽拰璇锋眰璺敱銆?
Author: PDF-MCP Team
License: MIT
"""

# 瀵煎叆鏍囧噯搴撴ā鍧?- 鐢ㄤ簬鍩虹鍔熻兘鏀寔
import asyncio  # 寮傛IO鏀寔锛岀敤浜庡鐞嗗紓姝ユ搷浣滃拰骞跺彂浠诲姟
import json  # JSON鏁版嵁澶勭悊锛岀敤浜庡簭鍒楀寲鍜屽弽搴忓垪鍖朖SON鏍煎紡鏁版嵁
import logging  # 鏃ュ織璁板綍锛岀敤浜庤褰曞簲鐢ㄧ▼搴忚繍琛屾椂鐨勪俊鎭拰閿欒
import os  # 鎿嶄綔绯荤粺鎺ュ彛锛岀敤浜庝笌鎿嶄綔绯荤粺杩涜浜や簰
import time  # 鏃堕棿澶勭悊锛岀敤浜庢椂闂寸浉鍏崇殑鎿嶄綔濡傛椂闂存埑璁＄畻
import argparse  # 命令行参数解析，用于解析启动时的命令行参数
from datetime import datetime  # 日期时间处理，用于获取和格式化时间
from contextlib import asynccontextmanager  # 异步上下文管理器，用于管理异步资源的生命周期  # 寮傛涓婁笅鏂囩鐞嗗櫒锛岀敤浜庣鐞嗗紓姝ヨ祫婧愮殑鐢熷懡鍛ㄦ湡
from typing import Dict, Any, Optional  # 绫诲瀷鎻愮ず锛岀敤浜庢彁渚涙洿濂界殑浠ｇ爜鍙鎬у拰IDE鏀寔
from pathlib import Path  # 璺緞澶勭悊锛岀敤浜庤法骞冲彴鐨勬枃浠惰矾寰勬搷浣?
# 瀵煎叆绗笁鏂瑰簱妯″潡 - 鐢ㄤ簬Web妗嗘灦鍜孧CP鍗忚鏀寔
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends  # FastAPI妗嗘灦锛岀敤浜庢瀯寤洪珮鎬ц兘鐨刉eb API
from fastapi.middleware.cors import CORSMiddleware  # CORS涓棿浠讹紝鐢ㄤ簬澶勭悊璺ㄥ煙璇锋眰
from fastapi.responses import JSONResponse  # JSON鍝嶅簲锛岀敤浜庤繑鍥濲SON鏍煎紡鐨凥TTP鍝嶅簲
from mcp.server import Server  # MCP服务器，用于实现MCP协议的服务器
from mcp.types import Tool, TextContent  # MCP类型定义，用于定义MCP协议中的工具和内容类型
import uvicorn  # ASGI服务器，用于运行FastAPI应用程序  # ASGI鏈嶅姟鍣紝鐢ㄤ簬杩愯FastAPI搴旂敤绋嬪簭

# 瀵煎叆椤圭洰鍐呴儴妯″潡 - 鐢ㄤ簬PDF澶勭悊鍜屼笟鍔￠€昏緫
from .models import (
    ProcessingRequest,  # 澶勭悊璇锋眰妯″瀷锛屽畾涔塒DF澶勭悊璇锋眰鐨勬暟鎹粨鏋?    ProcessingResponse,  # 澶勭悊鍝嶅簲妯″瀷锛屽畾涔塒DF澶勭悊鍝嶅簲鐨勬暟鎹粨鏋?    ErrorResponse,  # 閿欒鍝嶅簲妯″瀷锛屽畾涔夐敊璇俊鎭殑缁熶竴鍝嶅簲鏍煎紡
    HealthResponse,  # 鍋ュ悍妫€鏌ュ搷搴旀ā鍨嬶紝瀹氫箟鏈嶅姟鍣ㄥ仴搴风姸鎬佺殑鍝嶅簲鏍煎紡
    ProcessingMode,  # 澶勭悊妯″紡鏋氫妇锛屽畾涔変笉鍚岀殑PDF澶勭悊妯″紡锛堝鏂囨湰鎻愬彇銆丱CR绛夛級
)
from .processors import PDFProcessor  # PDF处理器，核心的PDF处理业务逻辑
from .utils.config import Config  # 配置管理，用于加载和管理应用程序配置  # 閰嶇疆绠＄悊锛岀敤浜庡姞杞藉拰绠＄悊搴旂敤绋嬪簭閰嶇疆
from .utils.logging_config import setup_logging  # 日志设置，用于配置日志记录系统
from .utils.exceptions import PDFProcessingError, ValidationError  # 自定义异常，定义PDF处理和验证相关的异常类型  # 鑷畾涔夊紓甯革紝瀹氫箟PDF澶勭悊鍜岄獙璇佺浉鍏崇殑寮傚父绫诲瀷

# 鍏ㄥ眬鍙橀噺瀹氫箟 - 鐢ㄤ簬瀛樺偍搴旂敤绋嬪簭鐨勬牳蹇冪粍浠跺拰鐘舵€佷俊鎭?# 瀛樺偍閰嶇疆瀵硅薄锛屽垵濮嬩负None锛屽湪搴旂敤鍚姩鏃跺垵濮嬪寲锛屽寘鍚墍鏈夊簲鐢ㄧ▼搴忛厤缃弬鏁?config: Optional[Config] = None
# 瀛樺偍PDF澶勭悊鍣ㄥ疄渚嬶紝鍒濆涓篘one锛屽湪搴旂敤鍚姩鏃跺垵濮嬪寲锛岃礋璐ｆ墍鏈塒DF澶勭悊鎿嶄綔
pdf_processor: Optional[PDFProcessor] = None
# 璁板綍鏈嶅姟鍣ㄥ惎鍔ㄦ椂闂达紝鐢ㄤ簬璁＄畻鏈嶅姟鍣ㄨ繍琛屾椂闀匡紝鍦ㄥ仴搴锋鏌ヤ腑浣跨敤
start_time = time.time()


# 瀹氫箟搴旂敤鐢熷懡鍛ㄦ湡绠＄悊鍣紝鐢ㄤ簬澶勭悊搴旂敤鍚姩鍜屽叧闂椂鐨勬搷浣?# 浣跨敤寮傛涓婁笅鏂囩鐞嗗櫒瑁呴グ鍣紝纭繚璧勬簮鐨勬纭垵濮嬪寲鍜屾竻鐞?@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    搴旂敤绋嬪簭鐢熷懡鍛ㄦ湡绠＄悊鍣?    
    绠＄悊搴旂敤绋嬪簭鐨勭敓鍛藉懆鏈燂紝鍖呮嫭鍚姩鏃剁殑鍒濆鍖栧拰鍏抽棴鏃剁殑娓呯悊宸ヤ綔銆?    鍦ㄥ簲鐢ㄥ惎鍔ㄦ椂鍒濆鍖栭厤缃€佹棩蹇楀拰PDF澶勭悊鍣紝鍦ㄥ叧闂椂娓呯悊璧勬簮銆?    
    Args:
        app (FastAPI): FastAPI搴旂敤瀹炰緥
        
    Yields:
        None: 鍦ㄥ簲鐢ㄨ繍琛屾湡闂存殏鍋滄墽琛?    """
    global config, pdf_processor  # 澹版槑浣跨敤鍏ㄥ眬鍙橀噺锛屽厑璁稿湪鍑芥暟鍐呬慨鏀瑰叏灞€鍙橀噺鐨勫€?    
    # 鍚姩闃舵 - 鍒濆鍖栧簲鐢ㄧ▼搴忔墍闇€鐨勬墍鏈夌粍浠?    logger = logging.getLogger(__name__)  # 鑾峰彇褰撳墠妯″潡鐨勬棩蹇楄褰曞櫒锛岀敤浜庤褰曡繍琛屾椂淇℃伅
    logger.info("Starting PDF-MCP Server...")  # 璁板綍鏈嶅姟鍣ㄥ惎鍔ㄥ紑濮嬬殑鏃ュ織淇℃伅
    
    try:
        # 绗竴姝ワ細鍔犺浇閰嶇疆鏂囦欢锛屽垵濮嬪寲閰嶇疆瀵硅薄
        config = Config()  # 鍒涘缓Config瀹炰緥锛屼粠閰嶇疆鏂囦欢鎴栫幆澧冨彉閲忎腑鍔犺浇鎵€鏈夐厤缃弬鏁?        
        # 绗簩姝ワ細璁剧疆鏃ュ織璁板綍绾у埆鍜屾棩蹇楁枃浠惰矾寰?        setup_logging(config.log_level, config.log_file)  # 鏍规嵁閰嶇疆璁剧疆鏃ュ織绯荤粺锛屽寘鎷棩蹇楃骇鍒拰杈撳嚭鏂囦欢
        
        # 绗笁姝ワ細鍒濆鍖朠DF澶勭悊鍣?        pdf_processor = PDFProcessor(config)  # 鍒涘缓PDFProcessor瀹炰緥锛屼紶鍏ラ厤缃璞?        await pdf_processor.initialize()  # 寮傛鍒濆鍖朠DF澶勭悊鍣紝鍖呮嫭鍔犺浇妯″瀷鍜岃缃鐞嗗紩鎿?        
        logger.info("PDF-MCP Server started successfully")  # 璁板綍鏈嶅姟鍣ㄦ垚鍔熷惎鍔ㄧ殑鏃ュ織淇℃伅
        
    except Exception as e:
        # 鎹曡幏鍚姩杩囩▼涓殑浠讳綍寮傚父骞惰褰曢敊璇棩蹇?        logger.error(f"Failed to start server: {e}")  # 璁板綍鍚姩澶辫触鐨勮缁嗛敊璇俊鎭?        raise  # 閲嶆柊鎶涘嚭鎹曡幏鐨勫紓甯革紝缁堟绋嬪簭鍚姩锛岄槻姝㈠湪閿欒鐘舵€佷笅缁х画杩愯
    
    yield  # 搴旂敤杩愯鏈熼棿鏆傚仠姝ゅ锛岀瓑寰呭簲鐢ㄥ叧闂俊鍙凤紙濡侰trl+C鎴栫郴缁熷叧闂級
    
    # 鍏抽棴闃舵 - 娓呯悊搴旂敤绋嬪簭璧勬簮锛岀‘淇濅紭闆呭叧闂?    logger.info("Shutting down PDF-MCP Server...")  # 璁板綍鏈嶅姟鍣ㄥ叧闂紑濮嬬殑鏃ュ織淇℃伅
    if pdf_processor:
        await pdf_processor.cleanup()  # 娓呯悊PDF澶勭悊鍣ㄨ祫婧愶紝鍖呮嫭鍏抽棴鏂囦欢鍙ユ焺銆侀噴鏀惧唴瀛樼瓑
    logger.info("PDF-MCP Server shutdown complete")  # 璁板綍鏈嶅姟鍣ㄥ叧闂畬鎴愮殑鏃ュ織淇℃伅


# 鍒涘缓FastAPI搴旂敤瀹炰緥 - 閰嶇疆Web API鏈嶅姟鍣?app = FastAPI(
    title="PDF-MCP Server",  # API鏍囬锛屾樉绀哄湪OpenAPI鏂囨。涓?    description="A comprehensive PDF processing MCP server",  # API鎻忚堪锛岃缁嗚鏄庢湇鍔″櫒鍔熻兘
    version="0.1.0",  # API鐗堟湰鍙凤紝鐢ㄤ簬鐗堟湰绠＄悊鍜屽吋瀹规€ф帶鍒?    lifespan=lifespan,  # 鍏宠仈鐢熷懡鍛ㄦ湡绠＄悊鍣紝纭繚搴旂敤鍚姩鍜屽叧闂椂鐨勬纭鐞?)

# 娣诲姞CORS涓棿浠讹紝澶勭悊璺ㄥ煙璇锋眰 - 鍏佽鍓嶇搴旂敤璁块棶API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 鍏佽鎵€鏈夋潵婧愯闂紙鐢熶骇鐜涓簲閰嶇疆鍏蜂綋鐨勫煙鍚嶅垪琛ㄤ互鎻愰珮瀹夊叏鎬э級
    allow_credentials=True,  # 鍏佽鎼哄甫鍑瘉锛堝cookies銆佽璇佸ご绛夛級
    allow_methods=["*"],  # 鍏佽鎵€鏈塇TTP鏂规硶锛圙ET銆丳OST銆丳UT銆丏ELETE绛夛級
    allow_headers=["*"],  # 鍏佽鎵€鏈夎姹傚ご锛堝寘鎷嚜瀹氫箟澶达級
)

# 鍒涘缓MCP鏈嶅姟鍣ㄥ疄渚?- 鐢ㄤ簬澶勭悊MCP鍗忚閫氫俊
mcp_server = Server("pdf-mcp-server")  # 鍒涘缓MCP鏈嶅姟鍣紝鎸囧畾鏈嶅姟鍣ㄥ悕绉扮敤浜庢爣璇?

# 瀹氫箟渚濊禆椤瑰嚱鏁帮紝鐢ㄤ簬鑾峰彇PDF澶勭悊鍣ㄥ疄渚?- FastAPI渚濊禆娉ㄥ叆绯荤粺
def get_pdf_processor() -> PDFProcessor:
    """Get the PDF processor instance.
    鑾峰彇PDF澶勭悊鍣ㄥ疄渚?    
    鑾峰彇PDF澶勭悊鍣ㄥ疄渚嬬殑渚濊禆娉ㄥ叆鍑芥暟銆?    濡傛灉澶勭悊鍣ㄦ湭鍒濆鍖栵紝鍒欐姏鍑篐TTP 503閿欒銆?    杩欎釜鍑芥暟鐢ㄤ簬FastAPI鐨勪緷璧栨敞鍏ョ郴缁燂紝纭繚姣忎釜闇€瑕丳DF澶勭悊鍣ㄧ殑绔偣閮借兘鑾峰緱鏈夋晥鐨勫疄渚嬨€?    
    Returns:
        PDFProcessor: 鍒濆鍖栧悗鐨凱DF澶勭悊鍣ㄥ疄渚?        
    Raises:
        HTTPException: 褰揚DF澶勭悊鍣ㄦ湭鍒濆鍖栨椂鎶涘嚭503閿欒
    """
    if pdf_processor is None:  # 妫€鏌DF澶勭悊鍣ㄦ槸鍚﹀凡鍒濆鍖栵紝闃叉鍦ㄦ湭鍒濆鍖栫姸鎬佷笅浣跨敤
        # 濡傛灉PDF澶勭悊鍣ㄦ湭鍒濆鍖栵紝鎶涘嚭鏈嶅姟涓嶅彲鐢ㄥ紓甯革紝鐘舵€佺爜503琛ㄧず鏈嶅姟鏆傛椂涓嶅彲鐢?        raise HTTPException(status_code=503, detail="PDF processor not initialized")
    return pdf_processor  # 杩斿洖鍒濆鍖栧悗鐨凱DF澶勭悊鍣ㄥ疄渚嬶紝渚涚鐐瑰嚱鏁颁娇鐢?

# 瀹氫箟寮傚父澶勭悊鍣紝澶勭悊PDF澶勭悊閿欒 - 鍏ㄥ眬閿欒澶勭悊鏈哄埗
@app.exception_handler(PDFProcessingError)
async def pdf_processing_exception_handler(request, exc: PDFProcessingError):
    """Handle PDF processing errors.
    澶勭悊PDF澶勭悊閿欒
    
    澶勭悊PDF澶勭悊杩囩▼涓彂鐢熺殑閿欒锛岃繑鍥炵粺涓€鐨勯敊璇搷搴旀牸寮忋€?    褰揚DF澶勭悊杩囩▼涓嚭鐜伴敊璇椂锛岃繖涓鐞嗗櫒浼氭崟鑾峰紓甯稿苟杩斿洖鏍囧噯鍖栫殑閿欒鍝嶅簲銆?    
    Args:
        request: HTTP璇锋眰瀵硅薄
        exc (PDFProcessingError): PDF澶勭悊寮傚父瀹炰緥
        
    Returns:
        JSONResponse: 鍖呭惈閿欒淇℃伅鐨凧SON鍝嶅簲
    """
    return JSONResponse(
        status_code=422,  # HTTP鐘舵€佺爜422琛ㄧず璇锋眰鏍煎紡姝ｇ‘浣嗚涔夐敊璇紙鏃犳硶澶勭悊鐨勫疄浣擄級
        content=ErrorResponse(
            error_code="PDF_PROCESSING_ERROR",  # 鏍囧噯鍖栫殑閿欒浠ｇ爜锛屼究浜庡鎴风璇嗗埆閿欒绫诲瀷
            message=str(exc),  # 閿欒娑堟伅锛屽寘鍚叿浣撶殑閿欒鎻忚堪
            details={"type": type(exc).__name__}  # 閿欒璇︽儏锛屽寘鍚紓甯哥被鍨嬪悕绉?        ).dict()  # 灏咵rrorResponse瀵硅薄杞崲涓哄瓧鍏告牸寮?    )


# 瀹氫箟寮傚父澶勭悊鍣紝澶勭悊楠岃瘉閿欒 - 杈撳叆鍙傛暟楠岃瘉閿欒澶勭悊
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation errors.
    澶勭悊楠岃瘉閿欒
    
    澶勭悊杈撳叆楠岃瘉杩囩▼涓彂鐢熺殑閿欒锛岃繑鍥炵粺涓€鐨勯敊璇搷搴旀牸寮忋€?    褰撹姹傚弬鏁颁笉绗﹀悎瑕佹眰鏃讹紝杩欎釜澶勭悊鍣ㄤ細鎹曡幏楠岃瘉寮傚父骞惰繑鍥炶缁嗙殑閿欒淇℃伅銆?    
    Args:
        request: HTTP璇锋眰瀵硅薄
        exc (ValidationError): 楠岃瘉寮傚父瀹炰緥
        
    Returns:
        JSONResponse: 鍖呭惈楠岃瘉閿欒淇℃伅鐨凧SON鍝嶅簲
    """
    return JSONResponse(
        status_code=400,  # HTTP鐘舵€佺爜400琛ㄧず瀹㈡埛绔敊璇紙璇锋眰鍙傛暟鏈夎锛?        content=ErrorResponse(
            error_code="VALIDATION_ERROR",  # 鏍囧噯鍖栫殑楠岃瘉閿欒浠ｇ爜
            message=str(exc),  # 閿欒娑堟伅锛屽寘鍚叿浣撶殑楠岃瘉澶辫触鍘熷洜
            details=exc.details if hasattr(exc, 'details') else {}  # 閿欒璇︽儏锛屽鏋滃紓甯稿寘鍚缁嗕俊鎭垯杩斿洖锛屽惁鍒欒繑鍥炵┖瀛楀吀
        ).dict()  # 灏咵rrorResponse瀵硅薄杞崲涓哄瓧鍏告牸寮?    )


# 瀹氫箟鍋ュ悍妫€鏌ョ鐐?- 鐢ㄤ簬鐩戞帶鏈嶅姟鍣ㄧ姸鎬?@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    鍋ュ悍妫€鏌ョ鐐?    
    鍋ュ悍妫€鏌ョ鐐癸紝鐢ㄤ簬鐩戞帶鏈嶅姟鍣ㄨ繍琛岀姸鎬併€?    杩斿洖鏈嶅姟鍣ㄨ繍琛屾椂闂淬€佺姸鎬佸拰鏃堕棿鎴崇瓑淇℃伅銆?    杩欎釜绔偣閫氬父琚礋杞藉潎琛″櫒銆佺洃鎺х郴缁熸垨瀹瑰櫒缂栨帓宸ュ叿璋冪敤锛岀敤浜庣‘璁ゆ湇鍔℃槸鍚︽甯歌繍琛屻€?    
    Returns:
        HealthResponse: 鍖呭惈鏈嶅姟鍣ㄥ仴搴风姸鎬佷俊鎭殑鍝嶅簲瀵硅薄
    """
    return HealthResponse(
        status="healthy",  # 鏈嶅姟鐘舵€佹爣璇嗭紝琛ㄧず鏈嶅姟鍣ㄦ甯歌繍琛?        uptime=time.time() - start_time,  # 璁＄畻鏈嶅姟鍣ㄨ繍琛屾椂闂达紙褰撳墠鏃堕棿鎴冲噺鍘诲惎鍔ㄦ椂闂存埑锛?        timestamp=datetime.utcnow().isoformat()  # 鑾峰彇褰撳墠UTC鏃堕棿骞舵牸寮忓寲涓篒SO 8601鏍囧噯鏍煎紡
    )


# 瀹氫箟PDF澶勭悊绔偣 - 涓昏鐨凱DF澶勭悊API鎺ュ彛
@app.post("/api/v1/process", response_model=ProcessingResponse)
async def process_pdf(
    request: ProcessingRequest,  # 澶勭悊璇锋眰鍙傛暟锛屽寘鍚枃浠惰矾寰勫拰澶勭悊閫夐」
    processor: PDFProcessor = Depends(get_pdf_processor)  # 渚濊禆娉ㄥ叆PDF澶勭悊鍣ㄥ疄渚?):
    """Process PDF with specified mode and options.
    澶勭悊PDF鏂囦欢
    
    鏍规嵁鎸囧畾鐨勬ā寮忓拰閫夐」澶勭悊PDF鏂囦欢銆?    杩欐槸涓昏鐨凱DF澶勭悊绔偣锛屾敮鎸佸绉嶅鐞嗘ā寮忥紙鏂囨湰鎻愬彇銆丱CR銆佸叕寮忚瘑鍒瓑锛夈€?    瀹㈡埛绔渶瑕佹彁渚汸DF鏂囦欢璺緞鍜屽鐞嗗弬鏁般€?    
    Args:
        request (ProcessingRequest): 鍖呭惈鏂囦欢璺緞鍜屽鐞嗛€夐」鐨勮姹傚璞?        processor (PDFProcessor): 閫氳繃渚濊禆娉ㄥ叆鑾峰彇鐨凱DF澶勭悊鍣ㄥ疄渚?        
    Returns:
        ProcessingResponse: 鍖呭惈澶勭悊缁撴灉鐨勫搷搴斿璞?        
    Raises:
        HTTPException: 褰撳鐞嗗け璐ユ椂鎶涘嚭500閿欒
    """
    try:
        # 璋冪敤澶勭悊鍣ㄦ墽琛孭DF澶勭悊锛屼紶鍏ヨ姹傚弬鏁?        result = await processor.process(request)
        return result  # 杩斿洖澶勭悊缁撴灉锛屽寘鍚彁鍙栫殑鏂囨湰銆佽〃鏍笺€佸叕寮忕瓑淇℃伅
    except Exception as e:
        logger = logging.getLogger(__name__)  # 鑾峰彇褰撳墠妯″潡鐨勬棩蹇楄褰曞櫒
        logger.error(f"Processing failed: {e}", exc_info=True)  # 璁板綍璇︾粏鐨勯敊璇棩蹇楋紝鍖呭惈寮傚父鍫嗘爤淇℃伅
        raise HTTPException(status_code=500, detail=str(e))  # 鎶涘嚭HTTP 500鏈嶅姟鍣ㄥ唴閮ㄩ敊璇?

# 瀹氫箟鏂囦欢涓婁紶澶勭悊绔偣 - 鏀寔鏂囦欢涓婁紶鍜屽鐞嗙殑API鎺ュ彛
@app.post("/api/v1/upload", response_model=ProcessingResponse)
async def upload_and_process(
    file: UploadFile = File(...),  # 涓婁紶鐨凱DF鏂囦欢锛屼娇鐢‵astAPI鐨凢ile绫诲瀷
    mode: ProcessingMode = Form(ProcessingMode.FULL),  # 澶勭悊妯″紡锛岄粯璁や负瀹屾暣澶勭悊妯″紡
    include_ocr: bool = Form(True),  # 鏄惁鍖呭惈OCR鏂囧瓧璇嗗埆澶勭悊锛岄粯璁ゅ惎鐢?    include_formulas: bool = Form(False),  # 鏄惁鍖呭惈鏁板鍏紡璇嗗埆锛岄粯璁ゅ叧闂?    include_grobid: bool = Form(False),  # 鏄惁鍖呭惈GROBID瀛︽湳鏂囨。澶勭悊锛岄粯璁ゅ叧闂?    processor: PDFProcessor = Depends(get_pdf_processor)  # 渚濊禆娉ㄥ叆PDF澶勭悊鍣ㄥ疄渚?):
    """Upload and process PDF file.
    涓婁紶骞跺鐞哖DF鏂囦欢
    
    涓婁紶骞跺鐞哖DF鏂囦欢鐨勭鐐广€?    鍏堝皢涓婁紶鐨勬枃浠朵繚瀛樺埌涓存椂浣嶇疆锛岀劧鍚庤繘琛屽鐞嗭紝鏈€鍚庢竻鐞嗕复鏃舵枃浠躲€?    杩欎釜绔偣閫傜敤浜庡鎴风鐩存帴涓婁紶PDF鏂囦欢杩涜澶勭悊鐨勫満鏅€?    
    Args:
        file (UploadFile): 涓婁紶鐨凱DF鏂囦欢
        mode (ProcessingMode): 澶勭悊妯″紡锛堝畬鏁淬€佹枃鏈€佽〃鏍肩瓑锛?        include_ocr (bool): 鏄惁鍚敤OCR鏂囧瓧璇嗗埆
        include_formulas (bool): 鏄惁鍚敤鍏紡璇嗗埆
        include_grobid (bool): 鏄惁鍚敤GROBID澶勭悊
        processor (PDFProcessor): PDF澶勭悊鍣ㄥ疄渚?        
    Returns:
        ProcessingResponse: 鍖呭惈澶勭悊缁撴灉鐨勫搷搴斿璞?        
    Raises:
        HTTPException: 褰撴枃浠剁被鍨嬩笉鏀寔鎴栧鐞嗗け璐ユ椂鎶涘嚭閿欒
    """
    # 妫€鏌ユ枃浠剁被鍨嬶紝鍙厑璁窹DF鏂囦欢 - 瀹夊叏鎬ч獙璇?    if not file.filename.lower().endswith('.pdf'):
        # 濡傛灉鏂囦欢鎵╁睍鍚嶄笉鏄?pdf锛屾姏鍑?00閿欒锛堝鎴风閿欒锛?        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # 鍒涘缓涓存椂鐩綍鐢ㄤ簬瀛樺偍涓婁紶鏂囦欢 - 鏂囦欢绠＄悊
    temp_dir = Path(config.temp_dir)  # 浠庨厤缃腑鑾峰彇涓存椂鐩綍璺緞
    temp_dir.mkdir(exist_ok=True)  # 濡傛灉鐩綍涓嶅瓨鍦ㄥ垯鍒涘缓锛宔xist_ok=True閬垮厤閲嶅鍒涘缓閿欒
    
    # 鐢熸垚涓存椂鏂囦欢璺緞锛屽寘鍚椂闂存埑鍜屽師鏂囦欢鍚?- 閬垮厤鏂囦欢鍚嶅啿绐?    temp_file = temp_dir / f"upload_{int(time.time())}_{file.filename}"  # 浣跨敤鏃堕棿鎴崇‘淇濇枃浠跺悕鍞竴鎬?    
    try:
        # 淇濆瓨涓婁紶鐨勬枃浠跺唴瀹瑰埌涓存椂鏂囦欢 - 鏂囦欢I/O鎿嶄綔
        with open(temp_file, "wb") as f:  # 浠ヤ簩杩涘埗鍐欏叆妯″紡鎵撳紑涓存椂鏂囦欢
            content = await file.read()  # 寮傛璇诲彇涓婁紶鏂囦欢鐨勫叏閮ㄥ唴瀹瑰埌鍐呭瓨
            f.write(content)  # 灏嗗唴瀹瑰啓鍏ヤ复鏃舵枃浠讹紝瀹屾垚鏂囦欢淇濆瓨
        
        # 鍒涘缓澶勭悊璇锋眰瀵硅薄 - 鏋勫缓澶勭悊鍙傛暟
        request = ProcessingRequest(
            file_path=str(temp_file),  # 涓存椂鏂囦欢鐨勭粷瀵硅矾寰勫瓧绗︿覆
            mode=mode,  # 澶勭悊妯″紡锛堝畬鏁淬€佹枃鏈€佽〃鏍肩瓑锛?            include_ocr=include_ocr,  # 鏄惁鍚敤OCR鏂囧瓧璇嗗埆鍔熻兘
            include_formulas=include_formulas,  # 鏄惁鍚敤鏁板鍏紡璇嗗埆鍔熻兘
            include_grobid=include_grobid  # 鏄惁鍚敤GROBID瀛︽湳鏂囨。澶勭悊鍔熻兘
        )
        
        # 璋冪敤澶勭悊鍣ㄦ墽琛孭DF澶勭悊 - 鏍稿績澶勭悊閫昏緫
        result = await processor.process(request)  # 寮傛璋冪敤PDF澶勭悊鍣ㄨ繘琛屾枃妗ｅ鐞?        return result  # 杩斿洖澶勭悊缁撴灉锛屽寘鍚彁鍙栫殑鏂囨湰銆佽〃鏍笺€佸叕寮忕瓑淇℃伅
        
    finally:
        # 娓呯悊涓存椂鏂囦欢锛屾棤璁哄鐞嗘垚鍔熶笌鍚﹂兘浼氭墽琛?- 璧勬簮娓呯悊
        if temp_file.exists():  # 妫€鏌ヤ复鏃舵枃浠舵槸鍚﹀瓨鍦紝閬垮厤鍒犻櫎涓嶅瓨鍦ㄧ殑鏂囦欢
            temp_file.unlink()  # 鍒犻櫎涓存椂鏂囦欢锛岄噴鏀剧鐩樼┖闂?

# MCP宸ュ叿瀹氫箟閮ㄥ垎 - Model Context Protocol宸ュ叿娉ㄥ唽
# 娉ㄥ唽鎵€鏈夊彲鐢ㄧ殑MCP宸ュ叿鍒版湇鍔″櫒锛屼緵瀹㈡埛绔皟鐢?@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools.
    鍒楀嚭鎵€鏈夊彲鐢ㄧ殑MCP宸ュ叿
    
    鍒楀嚭鎵€鏈夊彲鐢ㄧ殑MCP宸ュ叿銆?    杩欎簺宸ュ叿鍙互閫氳繃MCP鍗忚璋冪敤锛屾彁渚涘悇绉峆DF澶勭悊鍔熻兘銆?    姣忎釜宸ュ叿閮芥湁鏄庣‘鐨勮緭鍏ュ弬鏁版ā寮忓拰鍔熻兘鎻忚堪銆?    
    Returns:
        list[Tool]: 鍖呭惈鎵€鏈夊彲鐢ㄥ伐鍏风殑鍒楄〃锛屾瘡涓伐鍏峰寘鍚悕绉般€佹弿杩板拰杈撳叆妯″紡
    """
    return [
        # 鏂囨湰鎻愬彇宸ュ叿 - 浠嶱DF涓彁鍙栫函鏂囨湰鍐呭
        Tool(
            name="read_text",  # 宸ュ叿鍚嶇О锛岀敤浜嶮CP瀹㈡埛绔皟鐢?            description="Extract text content from PDF",  # 宸ュ叿鍔熻兘鎻忚堪
            inputSchema={  # 杈撳叆鍙傛暟鐨凧SON Schema瀹氫箟
                "type": "object",  # 鍙傛暟绫诲瀷涓哄璞?                "properties": {  # 瀹氫箟鍚勪釜鍙傛暟鐨勫睘鎬?                    "file_path": {"type": "string", "description": "Path to PDF file"},  # PDF鏂囦欢璺緞锛屽瓧绗︿覆绫诲瀷
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 鎸囧畾澶勭悊鐨勯〉闈㈠垪琛紝鏁存暟鏁扮粍
                    "include_bbox": {"type": "boolean", "description": "Include bounding box information"}  # 鏄惁鍖呭惈鏂囨湰杈圭晫妗嗗潗鏍囦俊鎭?                },
                "required": ["file_path"]  # 蹇呴渶鍙傛暟鍒楄〃锛宖ile_path涓哄繀濉」
            }
        ),
        # 琛ㄦ牸鎻愬彇宸ュ叿 - 浠嶱DF涓瘑鍒拰鎻愬彇琛ㄦ牸鏁版嵁
        Tool(
            name="extract_tables",  # 宸ュ叿鍚嶇О锛岀敤浜庤〃鏍兼彁鍙栧姛鑳?            description="Extract tables from PDF",  # 宸ュ叿鍔熻兘鎻忚堪
            inputSchema={  # 杈撳叆鍙傛暟鐨凧SON Schema瀹氫箟
                "type": "object",  # 鍙傛暟绫诲瀷涓哄璞?                "properties": {  # 瀹氫箟鍚勪釜鍙傛暟鐨勫睘鎬?                    "file_path": {"type": "string", "description": "Path to PDF file"},  # PDF鏂囦欢璺緞锛屽瓧绗︿覆绫诲瀷
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 鎸囧畾澶勭悊鐨勯〉闈㈠垪琛紝鏁存暟鏁扮粍
                    "engine": {"type": "string", "enum": ["camelot", "tabula", "pdfplumber"], "description": "Table extraction engine"},  # 琛ㄦ牸鎻愬彇寮曟搸
                    "output_format": {"type": "string", "enum": ["json", "csv", "dataframe"], "description": "Output format"}  # 杈撳嚭鏍煎紡閫夐」锛欽SON銆丆SV鎴朌ataFrame
                },
                "required": ["file_path"]  # 蹇呴渶鍙傛暟鍒楄〃锛宖ile_path涓哄繀濉」
            }
        ),
        # 鍏紡鎻愬彇宸ュ叿 - 浠嶱DF涓瘑鍒拰鎻愬彇鏁板鍏紡
        Tool(
            name="extract_formulas",  # 宸ュ叿鍚嶇О锛岀敤浜庢暟瀛﹀叕寮忔彁鍙栧姛鑳?            description="Extract mathematical formulas from PDF",  # 宸ュ叿鍔熻兘鎻忚堪
            inputSchema={  # 杈撳叆鍙傛暟鐨凧SON Schema瀹氫箟
                "type": "object",  # 鍙傛暟绫诲瀷涓哄璞?                "properties": {  # 瀹氫箟鍚勪釜鍙傛暟鐨勫睘鎬?                    "file_path": {"type": "string", "description": "Path to PDF file"},  # PDF鏂囦欢璺緞锛屽瓧绗︿覆绫诲瀷
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 鎸囧畾澶勭悊鐨勯〉闈㈠垪琛紝鏁存暟鏁扮粍
                    "engine": {"type": "string", "enum": ["pix2tex", "latex-ocr"], "description": "Formula extraction engine"},  # 鍏紡鎻愬彇寮曟搸閫夋嫨锛歱ix2tex鎴杔atex-ocr
                    "output_format": {"type": "string", "enum": ["latex", "mathml"], "description": "Output format"}  # 杈撳嚭鏍煎紡閫夐」锛歀aTeX鎴朚athML
                },
                "required": ["file_path"]  # 蹇呴渶鍙傛暟鍒楄〃锛宖ile_path涓哄繀濉」
            }
        ),
        # OCR澶勭悊宸ュ叿 - 瀵筆DF杩涜鍏夊瀛楃璇嗗埆
        Tool(
            name="process_ocr",  # 宸ュ叿鍚嶇О锛岀敤浜嶰CR鏂囧瓧璇嗗埆鍔熻兘
            description="Perform OCR on PDF",  # 宸ュ叿鍔熻兘鎻忚堪
            inputSchema={  # 杈撳叆鍙傛暟鐨凧SON Schema瀹氫箟
                "type": "object",  # 鍙傛暟绫诲瀷涓哄璞?                "properties": {  # 瀹氫箟鍚勪釜鍙傛暟鐨勫睘鎬?                    "file_path": {"type": "string", "description": "Path to PDF file"},  # PDF鏂囦欢璺緞锛屽瓧绗︿覆绫诲瀷
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to process"},  # 鎸囧畾澶勭悊鐨勯〉闈㈠垪琛紝鏁存暟鏁扮粍
                    "language": {"type": "string", "description": "OCR language"},  # OCR璇嗗埆璇█璁剧疆锛屽'eng'銆?chi_sim'绛?                    "output_format": {"type": "string", "enum": ["text", "hocr", "pdf"], "description": "Output format"}  # 杈撳嚭鏍煎紡閫夐」锛氱函鏂囨湰銆乭OCR鎴栧彲鎼滅储PDF
                },
                "required": ["file_path"]  # 蹇呴渶鍙傛暟鍒楄〃锛宖ile_path涓哄繀濉」
            }
        ),
        # PDF鍒嗘瀽宸ュ叿 - 鍒嗘瀽PDF鏂囨。缁撴瀯鍜屽唴瀹?        Tool(
            name="analyze_pdf",  # 宸ュ叿鍚嶇О锛岀敤浜嶱DF鏂囨。鍒嗘瀽鍔熻兘
            description="Analyze PDF structure and content",  # 宸ュ叿鍔熻兘鎻忚堪
            inputSchema={  # 杈撳叆鍙傛暟鐨凧SON Schema瀹氫箟
                "type": "object",  # 鍙傛暟绫诲瀷涓哄璞?                "properties": {  # 瀹氫箟鍚勪釜鍙傛暟鐨勫睘鎬?                    "file_path": {"type": "string", "description": "Path to PDF file"},  # PDF鏂囦欢璺緞锛屽瓧绗︿覆绫诲瀷
                    "analysis_type": {"type": "string", "enum": ["basic", "detailed"], "description": "Analysis depth"}  # 鍒嗘瀽娣卞害閫夋嫨锛氬熀纭€鍒嗘瀽鎴栬缁嗗垎鏋?                },
                "required": ["file_path"]  # 蹇呴渶鍙傛暟鍒楄〃锛宖ile_path涓哄繀濉」
            }
        ),
        # 瀹屾暣澶勭悊绠￠亾宸ュ叿 - 杩愯瀹屾暣鐨凱DF澶勭悊娴佺▼
        Tool(
            name="full_pipeline",  # 宸ュ叿鍚嶇О锛岀敤浜庡畬鏁寸殑PDF澶勭悊绠￠亾
            description="Run complete PDF processing pipeline",  # 宸ュ叿鍔熻兘鎻忚堪
            inputSchema={  # 杈撳叆鍙傛暟鐨凧SON Schema瀹氫箟
                "type": "object",  # 鍙傛暟绫诲瀷涓哄璞?                "properties": {  # 瀹氫箟鍚勪釜鍙傛暟鐨勫睘鎬?                    "file_path": {"type": "string", "description": "Path to PDF file"},  # PDF鏂囦欢璺緞锛屽瓧绗︿覆绫诲瀷
                    "include_ocr": {"type": "boolean", "description": "Include OCR processing"},  # 鏄惁鍖呭惈OCR鏂囧瓧璇嗗埆澶勭悊
                    "include_formulas": {"type": "boolean", "description": "Include formula detection"},  # 鏄惁鍖呭惈鏁板鍏紡璇嗗埆澶勭悊
                    "include_grobid": {"type": "boolean", "description": "Include GROBID processing"}  # 鏄惁鍖呭惈GROBID瀛︽湳鏂囨。澶勭悊
                },
                "required": ["file_path"]  # 蹇呴渶鍙傛暟鍒楄〃锛宖ile_path涓哄繀濉」
            }
        )
    ]


# 涓诲嚱鏁帮紝鐢ㄤ簬鐩存帴杩愯鏈嶅姟鍣?- 绋嬪簭鍏ュ彛鐐?def main():
    """Main entry point for running the server.
    涓诲嚱鏁?- 鏈嶅姟鍣ㄥ惎鍔ㄥ叆鍙?    
    涓诲嚱鏁帮紝瑙ｆ瀽鍛戒护琛屽弬鏁板苟鍚姩鏈嶅姟鍣ㄣ€?    鏀寔鎸囧畾涓绘満銆佺鍙ｃ€侀噸杞藉拰宸ヤ綔杩涚▼鏁扮瓑鍙傛暟銆?    杩欎釜鍑芥暟閫氬父鍦ㄥ紑鍙戝拰娴嬭瘯鐜涓娇鐢紝鐢熶骇鐜寤鸿浣跨敤WSGI鏈嶅姟鍣ㄣ€?    
    Command line arguments:
        --host: 鏈嶅姟鍣ㄧ粦瀹氱殑涓绘満鍦板潃锛岄粯璁や负127.0.0.1
        --port: 鏈嶅姟鍣ㄧ粦瀹氱殑绔彛鍙凤紝榛樿涓?000
        --reload: 鏄惁鍚敤鑷姩閲嶈浇锛屽紑鍙戞椂鏈夌敤
        --workers: 宸ヤ綔杩涚▼鏁伴噺锛岄粯璁や负1
    """
    # 鍒涘缓鍛戒护琛屽弬鏁拌В鏋愬櫒 - 澶勭悊鍚姩鍙傛暟
    parser = argparse.ArgumentParser(description="PDF-MCP Server")  # 鍒涘缓鍙傛暟瑙ｆ瀽鍣紝璁剧疆绋嬪簭鎻忚堪
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")  # 涓绘満鍦板潃鍙傛暟锛岄粯璁ゆ湰鍦板洖鐜湴鍧€
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")  # 绔彛鍙傛暟锛屾暣鏁扮被鍨嬶紝榛樿8000
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")  # 閲嶈浇鍙傛暟锛屽竷灏旀爣蹇楋紝寮€鍙戞椂鑷姩閲嶈浇浠ｇ爜
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")  # 宸ヤ綔杩涚▼鏁板弬鏁帮紝鏁存暟绫诲瀷锛岄粯璁ゅ崟杩涚▼
    
    # 瑙ｆ瀽鍛戒护琛屽弬鏁?- 鑾峰彇鐢ㄦ埛杈撳叆鐨勫惎鍔ㄩ厤缃?    args = parser.parse_args()  # 瑙ｆ瀽sys.argv涓殑鍛戒护琛屽弬鏁?    
    # 浣跨敤uvicorn杩愯FastAPI搴旂敤 - 鍚姩ASGI鏈嶅姟鍣?    uvicorn.run(
        "pdf_mcp_server.main:app",  # 搴旂敤妯″潡璺緞锛屾寚鍚戝綋鍓嶆ā鍧楃殑app瀹炰緥
        host=args.host,  # 缁戝畾鐨勪富鏈哄湴鍧€锛屼粠鍛戒护琛屽弬鏁拌幏鍙?        port=args.port,  # 缁戝畾鐨勭鍙ｅ彿锛屼粠鍛戒护琛屽弬鏁拌幏鍙?        reload=args.reload,  # 閲嶈浇閫夐」锛屼粠鍛戒护琛屽弬鏁拌幏鍙栵紝寮€鍙戞椂鍚敤浠ｇ爜鑷姩閲嶈浇
        workers=args.workers  # 宸ヤ綔杩涚▼鏁帮紝浠庡懡浠よ鍙傛暟鑾峰彇锛岀敓浜х幆澧冨彲璁剧疆澶氳繘绋嬫彁楂樻€ц兘
    )


# 濡傛灉浣滀负涓绘ā鍧楄繍琛岋紝鍒欐墽琛屼富鍑芥暟 - Python鏍囧噯鍏ュ彛鐐规鏌?if __name__ == "__main__":  # 妫€鏌ユ槸鍚︾洿鎺ヨ繍琛屾鑴氭湰锛堣€岄潪琚鍏ワ級
    main()  # 璋冪敤涓诲嚱鏁板惎鍔ㄦ湇鍔″櫒锛屽紑濮嬪鐞哖DF澶勭悊璇锋眰





