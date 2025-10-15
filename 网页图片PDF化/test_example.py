#!/usr/bin/env python3
"""
测试示例脚本

演示如何使用网页截图PDF工具的各种功能。
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from web_screenshot_pdf import (
    WebScreenshotPDF, 
    ScreenshotConfig, 
    OCRConfig, 
    PDFConfig
)


async def test_single_url():
    """测试单个URL处理"""
    print("🧪 测试单个URL处理...")
    
    # 配置
    screenshot_config = ScreenshotConfig(
        width=1920,
        height=1080,
        full_page=True,
        wait_time=3,
        quality=95,
        format="PNG"
    )
    
    ocr_config = OCRConfig(
        engine="auto",
        languages=["eng"],
        confidence_threshold=0.6,
        preprocess=True
    )
    
    pdf_config = PDFConfig(
        page_size="A4",
        include_ocr_text=True,
        searchable=True,
        title="测试单页PDF",
        author="测试用户"
    )
    
    # 处理
    url = "https://httpbin.org/html"  # 简单的测试页面
    output_path = Path("test_single.pdf")
    
    try:
        async with WebScreenshotPDF(browser_engine="playwright") as processor:
            result = await processor.process_urls_to_pdf(
                urls=[url],
                output_path=output_path,
                screenshot_config=screenshot_config,
                ocr_config=ocr_config,
                pdf_config=pdf_config
            )
            
            if result['success']:
                print(f"✅ 单个URL处理成功!")
                print(f"   📄 输出文件: {output_path}")
                print(f"   ⏱️  处理时间: {result['processing_time']:.2f}秒")
                if result['ocr_results']:
                    ocr_result = result['ocr_results'][0]['ocr_result']
                    print(f"   📝 OCR字符数: {ocr_result.get('character_count', 0)}")
                    print(f"   🎯 OCR置信度: {ocr_result.get('confidence', 0):.2%}")
            else:
                print(f"❌ 单个URL处理失败: {result.get('error', '未知错误')}")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")


async def test_batch_urls():
    """测试批量URL处理"""
    print("\n🧪 测试批量URL处理...")
    
    # 测试URL列表
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/xml"
    ]
    
    # 配置
    screenshot_config = ScreenshotConfig(
        width=1366,
        height=768,
        full_page=True,
        wait_time=2
    )
    
    ocr_config = OCRConfig(
        engine="auto",
        languages=["eng"],
        confidence_threshold=0.5
    )
    
    pdf_config = PDFConfig(
        page_size="A4",
        include_ocr_text=True,
        searchable=True,
        title="批量测试PDF",
        author="测试用户"
    )
    
    output_path = Path("test_batch.pdf")
    
    try:
        async with WebScreenshotPDF(browser_engine="playwright") as processor:
            result = await processor.process_urls_to_pdf(
                urls=urls,
                output_path=output_path,
                screenshot_config=screenshot_config,
                ocr_config=ocr_config,
                pdf_config=pdf_config
            )
            
            if result['success']:
                print(f"✅ 批量处理成功!")
                print(f"   📄 输出文件: {output_path}")
                print(f"   📊 处理统计: {result['processed_urls']}/{result['total_urls']}")
                print(f"   ⏱️  总时间: {result['processing_time']:.2f}秒")
                
                if result['failed_urls']:
                    print(f"   ⚠️  失败URL: {len(result['failed_urls'])}")
                    for failed in result['failed_urls']:
                        print(f"      - {failed['url']}: {failed['error']}")
            else:
                print(f"❌ 批量处理失败: {result.get('error', '未知错误')}")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")


async def test_screenshot_only():
    """测试仅截图功能"""
    print("\n🧪 测试仅截图功能...")
    
    screenshot_config = ScreenshotConfig(
        width=1920,
        height=1080,
        full_page=True,
        wait_time=3,
        format="PNG"
    )
    
    url = "https://httpbin.org/html"
    
    try:
        async with WebScreenshotPDF(browser_engine="playwright") as processor:
            screenshot_path = await processor.screenshot_url(url, screenshot_config)
            
            print(f"✅ 截图成功!")
            print(f"   📸 截图文件: {screenshot_path}")
            
            # 测试图片预处理
            processed_path = processor.preprocess_image(
                screenshot_path,
                enhance=True,
                denoise=True,
                resize_factor=0.8
            )
            
            print(f"   🔧 预处理文件: {processed_path}")
            
    except Exception as e:
        print(f"❌ 截图测试失败: {e}")


async def test_ocr_only():
    """测试仅OCR功能"""
    print("\n🧪 测试仅OCR功能...")
    
    # 首先截图
    screenshot_config = ScreenshotConfig(
        width=1920,
        height=1080,
        full_page=True,
        wait_time=3
    )
    
    url = "https://httpbin.org/html"
    
    try:
        async with WebScreenshotPDF(browser_engine="playwright") as processor:
            # 截图
            screenshot_path = await processor.screenshot_url(url, screenshot_config)
            
            # 测试不同OCR引擎
            ocr_engines = ["auto"]  # 可以添加 "tesseract", "easyocr", "paddleocr"
            
            for engine in ocr_engines:
                if engine == "auto" or processor.ocr_engines.get(engine, False):
                    print(f"   🔤 测试 {engine} OCR引擎...")
                    
                    ocr_config = OCRConfig(
                        engine=engine,
                        languages=["eng"],
                        confidence_threshold=0.5,
                        preprocess=True
                    )
                    
                    result = await processor.extract_text_ocr(screenshot_path, ocr_config)
                    
                    print(f"      ✅ 引擎: {result['engine']}")
                    print(f"      📝 字符数: {result['character_count']}")
                    print(f"      🎯 置信度: {result['confidence']:.2%}")
                    
                    if result['text']:
                        preview = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                        print(f"      📄 文本预览: {preview}")
                else:
                    print(f"   ⚠️  {engine} OCR引擎不可用")
            
    except Exception as e:
        print(f"❌ OCR测试失败: {e}")


async def test_custom_config():
    """测试自定义配置"""
    print("\n🧪 测试自定义配置...")
    
    # 自定义截图配置
    screenshot_config = ScreenshotConfig(
        width=1366,
        height=768,
        full_page=False,  # 仅视口
        wait_time=5,
        quality=80,
        format="JPEG",
        hide_elements=[".header", ".footer", ".sidebar"]  # 隐藏元素
    )
    
    # 自定义OCR配置
    ocr_config = OCRConfig(
        engine="auto",
        languages=["eng", "chi_sim"],  # 多语言
        confidence_threshold=0.7,  # 更高置信度
        preprocess=True,
        enhance_contrast=True,
        enhance_sharpness=True,
        denoise=True
    )
    
    # 自定义PDF配置
    pdf_config = PDFConfig(
        page_size="letter",  # Letter页面
        margin=30,
        include_ocr_text=True,
        searchable=True,
        compress_images=True,
        image_quality=75,
        title="自定义配置测试",
        author="高级用户",
        subject="测试自定义配置功能",
        keywords="测试,自定义,配置"
    )
    
    url = "https://httpbin.org/html"
    output_path = Path("test_custom.pdf")
    
    try:
        async with WebScreenshotPDF(browser_engine="playwright") as processor:
            result = await processor.process_urls_to_pdf(
                urls=[url],
                output_path=output_path,
                screenshot_config=screenshot_config,
                ocr_config=ocr_config,
                pdf_config=pdf_config
            )
            
            if result['success']:
                print(f"✅ 自定义配置测试成功!")
                print(f"   📄 输出文件: {output_path}")
                print(f"   ⏱️  处理时间: {result['processing_time']:.2f}秒")
            else:
                print(f"❌ 自定义配置测试失败: {result.get('error', '未知错误')}")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def test_config_management():
    """测试配置管理"""
    print("\n🧪 测试配置管理...")
    
    try:
        from config import ConfigManager, create_example_config
        
        # 创建示例配置
        create_example_config()
        print("✅ 示例配置文件创建成功")
        
        # 测试配置管理器
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"   🔧 浏览器引擎: {config.browser.engine}")
        print(f"   🔤 OCR引擎: {config.ocr.engine}")
        print(f"   📄 PDF页面大小: {config.pdf.page_size}")
        
        # 验证配置
        errors = config_manager.validate_config()
        if errors:
            print(f"   ⚠️  配置错误: {errors}")
        else:
            print("   ✅ 配置验证通过")
            
    except Exception as e:
        print(f"❌ 配置管理测试失败: {e}")


def check_dependencies():
    """检查依赖"""
    print("🔍 检查系统依赖...")
    
    # 检查浏览器
    print("\n🌐 浏览器引擎:")
    try:
        from selenium import webdriver
        print("   ✅ Selenium 可用")
    except ImportError:
        print("   ❌ Selenium 未安装")
    
    try:
        from playwright.async_api import async_playwright
        print("   ✅ Playwright 可用")
    except ImportError:
        print("   ❌ Playwright 未安装")
    
    # 检查OCR引擎
    print("\n🔤 OCR引擎:")
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Tesseract 可用 (版本: {version})")
    except Exception as e:
        print(f"   ❌ Tesseract 不可用: {e}")
    
    try:
        import easyocr
        print("   ✅ EasyOCR 可用")
    except ImportError:
        print("   ❌ EasyOCR 未安装")
    
    try:
        import paddleocr
        print("   ✅ PaddleOCR 可用")
    except ImportError:
        print("   ❌ PaddleOCR 未安装")
    
    # 检查图像处理
    print("\n🖼️  图像处理:")
    try:
        from PIL import Image
        print("   ✅ Pillow 可用")
    except ImportError:
        print("   ❌ Pillow 未安装")
    
    try:
        import cv2
        print("   ✅ OpenCV 可用")
    except ImportError:
        print("   ❌ OpenCV 未安装")
    
    # 检查PDF处理
    print("\n📄 PDF处理:")
    try:
        from reportlab.pdfgen import canvas
        print("   ✅ ReportLab 可用")
    except ImportError:
        print("   ❌ ReportLab 未安装")
    
    try:
        import fitz
        print("   ✅ PyMuPDF 可用")
    except ImportError:
        print("   ❌ PyMuPDF 未安装")


async def main():
    """主测试函数"""
    print("🚀 开始网页截图PDF工具测试")
    print("=" * 50)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 检查依赖
    check_dependencies()
    
    # 测试配置管理
    test_config_management()
    
    # 运行异步测试
    print("\n🧪 开始功能测试...")
    
    # 测试单个URL
    await test_single_url()
    
    # 测试仅截图
    await test_screenshot_only()
    
    # 测试仅OCR
    await test_ocr_only()
    
    # 测试自定义配置
    await test_custom_config()
    
    # 测试批量处理（最后执行，因为比较耗时）
    await test_batch_urls()
    
    print("\n🎉 所有测试完成!")
    print("📁 生成的文件:")
    for file_path in Path(".").glob("test_*.pdf"):
        print(f"   📄 {file_path}")
    for file_path in Path(".").glob("screenshot_*.png"):
        print(f"   📸 {file_path}")
    for file_path in Path(".").glob("processed_*.png"):
        print(f"   🔧 {file_path}")


if __name__ == "__main__":
    asyncio.run(main())