#!/usr/bin/env python3  # 指定使用Python 3解释器运行该脚本（Unix系可用）

import sys  # 导入sys模块，用于操作Python解释器相关的系统参数（如路径）
import asyncio  # 导入asyncio，用于运行和管理异步协程
import tempfile  # 导入tempfile，用于创建临时文件/目录（用于临时配置文件）
import json  # 导入json模块，用于序列化/反序列化JSON数据
from pathlib import Path  # 导入Path，用于跨平台路径处理（更安全直观）

# 将src目录加入Python模块搜索路径，以便可以import项目内的包
# 注意：path插入到索引0意味着优先于系统site-packages，避免命名冲突时要谨慎
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_mcp_server.main import PDFMCPServer  # 从项目主模块导入PDFMCPServer类（服务器主入口）


async def test_server():  # 定义异步测试函数，用于启动和初始化服务器进行快速健康检查
    # 创建一个临时配置文件，随着进程结束自动清理（delete=False便于后续再次读取）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:  # 以写模式创建后缀为.json的临时文件
        config = {  # 构造最小可用的服务器配置字典（与服务器Config结构对齐）
            "server": {  # 服务器基础信息与运行参数配置节
                "name": "pdf-mcp-test-server",  # 服务器名称（用于日志/识别）
                "version": "1.0.0",  # 服务器版本号（非功能性）
                "description": "Test PDF processing server",  # 描述信息（可用于文档/自检）
                "max_concurrent_requests": 5,  # 并发处理请求上限（防止资源耗尽）
                "request_timeout": 60,  # 单请求超时时间（秒），防止长时间卡住
                "temp_dir": "./temp_test",  # 临时文件目录（上传/中间产物）
                "cleanup_interval": 3600  # 清理周期（秒），后台清理临时文件的时间间隔
            },
            "logging": {  # 日志配置节
                "level": "INFO",  # 日志级别（DEBUG/INFO/WARNING/ERROR）
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 日志格式模板
                "file": None  # 日志输出文件（None表示仅控制台输出）
            },
            "tools": {  # 工具开关配置（决定哪些处理能力启用）
                "text_extraction": {"enabled": True},  # 启用文本提取工具
                "table_extraction": {"enabled": False},  # 关闭表格提取工具（减少依赖/启动时间）
                "ocr": {"enabled": False},  # 关闭OCR（需要额外依赖/模型）
                "formula_recognition": {"enabled": False},  # 关闭公式识别（同上）
                "analysis": {"enabled": True}  # 启用文档分析（例如结构/标题检测等）
            }
        }
        json.dump(config, temp_file)  # 将配置字典写入临时文件（JSON格式）
        temp_file_path = temp_file.name  # 记录临时文件路径，供服务器读取配置使用
    
    try:  # 使用try/except捕获初始化过程中的异常，便于输出完整堆栈排查问题
        print("Creating PDFMCPServer...")  # 提示当前步骤：创建服务器实例
        server = PDFMCPServer(config_file=temp_file_path)  # 实例化服务器，传入临时配置文件路径
        print("Server created successfully")  # 创建成功提示
        
        print("Initializing server...")  # 提示当前步骤：初始化（通常加载模型/资源/线程等）
        await server.initialize()  # 异步初始化服务器（确保内部异步资源正确准备）
        print("Server initialized successfully")  # 初始化成功提示
        
    except Exception as e:  # 捕获所有异常（初始化可能因依赖缺失/配置错误失败）
        print(f"Error: {e}")  # 打印简要错误信息到标准输出
        import traceback  # 延迟导入traceback，避免无必要的全局导入
        traceback.print_exc()  # 打印完整堆栈，便于定位具体失败原因


if __name__ == "__main__":  # 仅当作为脚本直接运行时执行（被import则不执行）
    asyncio.run(test_server())  # 使用asyncio运行顶层异步函数，自动创建/关闭事件循环
