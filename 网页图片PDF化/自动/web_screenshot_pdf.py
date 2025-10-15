#!/usr/bin/env python3
"""
网页截图并合成PDF（含OCR）的完整解决方案

功能特性：
- 支持多种浏览器引擎（Selenium、Playwright）
- 智能网页截图（全页面、可视区域、元素截图）
- 图片预处理和优化
- OCR文字识别（Tesseract、EasyOCR、PaddleOCR）
- PDF合成和文本层嵌入
- 批量处理和并发支持

Author: AI Assistant
License: MIT
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import tempfile
import shutil
import os

# 图像处理
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
except ImportError as e:
    print(f"图像处理库导入失败: {e}")
    print("请安装: pip install Pillow opencv-python numpy")

# OCR引擎
try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("Tesseract未安装，请安装: pip install pytesseract")

try:
    import easyocr
except ImportError:
    easyocr = None
    print("EasyOCR未安装，请安装: pip install easyocr")

try:
    import paddleocr
except ImportError:
    paddleocr = None
    print("PaddleOCR未安装，请安装: pip install paddleocr")

# 浏览器自动化
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
except ImportError:
    webdriver = None
    print("Selenium未安装，请安装: pip install selenium")

# WebDriver管理器
try:
    from webdriver_manager.chrome import ChromeDriverManager
    webdriver_manager_available = True
except ImportError:
    webdriver_manager_available = False
    print("WebDriver Manager未安装，请安装: pip install webdriver-manager")

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None
    print("Playwright未安装，请安装: pip install playwright")

# PDF处理
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import fitz  # PyMuPDF
except ImportError as e:
    print(f"PDF处理库导入失败: {e}")
    print("请安装: pip install reportlab PyMuPDF")


@dataclass
class ScreenshotConfig:
    """截图配置"""
    width: int = 1920
    height: int = 1080
    full_page: bool = True
    wait_time: int = 3
    element_selector: Optional[str] = None
    hide_elements: List[str] = field(default_factory=list)
    mobile_emulation: bool = False
    device_name: Optional[str] = None
    quality: int = 95
    format: str = "PNG"


@dataclass
class OCRConfig:
    """OCR配置"""
    engine: str = "tesseract"  # tesseract, easyocr, paddleocr
    languages: List[str] = field(default_factory=lambda: ["eng", "chi_sim"])
    confidence_threshold: float = 0.6
    preprocess: bool = True
    dpi: int = 300


@dataclass
class PDFConfig:
    """PDF配置"""
    page_size: str = "A4"  # A4, letter, custom
    margin: int = 50
    include_ocr_text: bool = True
    searchable: bool = True
    compress: bool = True
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None


class WebScreenshotPDF:
    """网页截图并合成PDF的主类"""
    
    def __init__(self, 
                 browser_engine: str = "playwright",  # selenium, playwright
                 temp_dir: Optional[Path] = None):
        """
        初始化
        
        Args:
            browser_engine: 浏览器引擎 (selenium, playwright)
            temp_dir: 临时目录
        """
        self.browser_engine = browser_engine
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "web_screenshot_pdf"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 初始化OCR引擎
        self._init_ocr_engines()
        
        # 浏览器实例
        self.driver = None
        self.playwright_browser = None
        self.playwright_context = None
        self.playwright_page = None
    
    def _init_ocr_engines(self):
        """初始化OCR引擎"""
        self.ocr_engines = {}
        
        # Tesseract
        if pytesseract:
            try:
                # 尝试设置Tesseract路径（Windows）
                if os.name == 'nt':
                    tesseract_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                    ]
                    for path in tesseract_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            break
                
                # 测试Tesseract
                pytesseract.get_tesseract_version()
                self.ocr_engines['tesseract'] = True
                self.logger.info("Tesseract OCR 初始化成功")
            except Exception as e:
                self.logger.warning(f"Tesseract OCR 初始化失败: {e}")
                self.ocr_engines['tesseract'] = False
        
        # EasyOCR
        if easyocr:
            try:
                self.easyocr_reader = None  # 延迟初始化
                self.ocr_engines['easyocr'] = True
                self.logger.info("EasyOCR 可用")
            except Exception as e:
                self.logger.warning(f"EasyOCR 初始化失败: {e}")
                self.ocr_engines['easyocr'] = False
        
        # PaddleOCR
        if paddleocr:
            try:
                self.paddleocr_reader = None  # 延迟初始化
                self.ocr_engines['paddleocr'] = True
                self.logger.info("PaddleOCR 可用")
            except Exception as e:
                self.logger.warning(f"PaddleOCR 初始化失败: {e}")
                self.ocr_engines['paddleocr'] = False
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._init_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._cleanup_browser()
    
    async def _init_browser(self):
        """初始化浏览器"""
        # 尝试初始化指定的浏览器引擎
        if self.browser_engine == "playwright":
            if async_playwright:
                try:
                    playwright = await async_playwright().start()
                    self.playwright_browser = await playwright.chromium.launch(headless=True)
                    self.playwright_context = await self.playwright_browser.new_context()
                    self.playwright_page = await self.playwright_context.new_page()
                    self.logger.info("Playwright 浏览器初始化成功")
                    return
                except Exception as e:
                    self.logger.warning(f"Playwright 初始化失败: {e}")
            else:
                self.logger.warning("Playwright 未安装")
            
            # Playwright失败，尝试切换到Selenium
            self.logger.info("尝试切换到 Selenium Chrome 浏览器...")
            self.browser_engine = "selenium"
        
        if self.browser_engine == "selenium":
            if webdriver:
                try:
                    options = ChromeOptions()
                    options.add_argument("--headless")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    options.add_argument("--disable-gpu")
                    options.add_argument("--disable-blink-features=AutomationControlled")
                    options.add_experimental_option("excludeSwitches", ["enable-automation"])
                    options.add_experimental_option('useAutomationExtension', False)
                    
                    # 为每个实例创建唯一的用户数据目录，避免冲突
                    import uuid
                    unique_id = str(uuid.uuid4())[:8]
                    user_data_dir = self.temp_dir / f"chrome_user_data_{unique_id}"
                    user_data_dir.mkdir(exist_ok=True)
                    options.add_argument(f"--user-data-dir={user_data_dir}")
                    
                    # 其他稳定性选项
                    options.add_argument("--disable-extensions")
                    options.add_argument("--disable-plugins")
                    options.add_argument("--disable-web-security")
                    options.add_argument("--allow-running-insecure-content")
                    options.add_argument("--disable-features=VizDisplayCompositor")
                    options.add_argument("--remote-debugging-port=0")  # 随机端口
                    options.add_argument("--window-size=1920,1080")
                    options.add_argument("--start-maximized")
                    options.add_argument("--disable-background-timer-throttling")
                    options.add_argument("--disable-backgrounding-occluded-windows")
                    options.add_argument("--disable-renderer-backgrounding")
                    options.add_argument("--disable-ipc-flooding-protection")
                    # 注释掉这些可能导致问题的选项
                    # options.add_argument("--disable-images")  # 可能导致某些网站加载问题
                    # options.add_argument("--disable-javascript")  # 可能导致某些网站加载问题
                    
                    # 尝试使用webdriver-manager自动管理ChromeDriver
                    if webdriver_manager_available:
                        try:
                            service = ChromeService(ChromeDriverManager().install())
                            self.driver = webdriver.Chrome(service=service, options=options)
                            self.logger.info("Selenium Chrome 浏览器初始化成功 (使用 WebDriver Manager)")
                            return
                        except Exception as e:
                            self.logger.warning(f"WebDriver Manager 失败: {e}")
                    
                    # 如果webdriver-manager失败，尝试使用系统PATH中的ChromeDriver
                    self.driver = webdriver.Chrome(options=options)
                    self.logger.info("Selenium Chrome 浏览器初始化成功 (使用系统 ChromeDriver)")
                    return
                except Exception as e:
                    self.logger.error(f"Selenium 浏览器初始化失败: {e}")
                    # 提供更详细的错误信息和解决方案
                    error_msg = str(e).lower()
                    if "chromedriver" in error_msg or "chrome" in error_msg:
                        raise ValueError(
                            f"Chrome WebDriver 初始化失败。\n"
                            f"解决方案：\n"
                            f"1. 确保已安装 Google Chrome 浏览器\n"
                            f"2. 运行: pip install webdriver-manager\n"
                            f"3. 或手动下载 ChromeDriver 并添加到系统PATH\n"
                            f"错误详情: {e}"
                        )
                    else:
                        raise ValueError(f"Selenium 初始化失败: {e}")
            else:
                raise ValueError("Selenium 未安装。请安装: pip install selenium")
        
        # 如果所有引擎都失败了
        raise ValueError(f"无法初始化任何浏览器引擎。请确保安装了 Selenium 或 Playwright，并配置了相应的浏览器驱动。")
    
    async def _cleanup_browser(self):
        """清理浏览器资源"""
        try:
            if self.playwright_page:
                await self.playwright_page.close()
            if self.playwright_context:
                await self.playwright_context.close()
            if self.playwright_browser:
                await self.playwright_browser.close()
            
            if self.driver:
                self.driver.quit()
                
            self.logger.info("浏览器资源清理完成")
        except Exception as e:
            self.logger.error(f"浏览器资源清理失败: {e}")
    
    async def screenshot_url(self, 
                           url: str, 
                           config: ScreenshotConfig = None) -> Path:
        """
        截取网页截图
        
        Args:
            url: 网页URL
            config: 截图配置
            
        Returns:
            截图文件路径
        """
        if config is None:
            config = ScreenshotConfig()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.temp_dir / f"screenshot_{timestamp}.{config.format.lower()}"
        
        try:
            if self.browser_engine == "playwright":
                await self._screenshot_with_playwright(url, screenshot_path, config)
            elif self.browser_engine == "selenium":
                await self._screenshot_with_selenium(url, screenshot_path, config)
            else:
                raise ValueError(f"不支持的浏览器引擎: {self.browser_engine}")
            
            self.logger.info(f"网页截图完成: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            self.logger.error(f"网页截图失败: {e}")
            raise
    
    async def _screenshot_with_playwright(self, 
                                        url: str, 
                                        output_path: Path, 
                                        config: ScreenshotConfig):
        """使用Playwright截图"""
        if not self.playwright_page:
            raise RuntimeError("Playwright 未初始化")
        
        # 设置视口大小
        await self.playwright_page.set_viewport_size({
            "width": config.width,
            "height": config.height
        })
        
        # 访问页面
        await self.playwright_page.goto(url, wait_until="networkidle")
        
        # 等待页面加载
        await asyncio.sleep(config.wait_time)
        
        # 隐藏指定元素
        for selector in config.hide_elements:
            try:
                await self.playwright_page.evaluate(f"""
                    document.querySelectorAll('{selector}').forEach(el => el.style.display = 'none');
                """)
            except Exception as e:
                self.logger.warning(f"隐藏元素失败 {selector}: {e}")
        
        # 截图
        screenshot_options = {
            "path": str(output_path),
            "full_page": config.full_page,
            "quality": config.quality if config.format.upper() == "JPEG" else None
        }
        
        if config.element_selector:
            element = await self.playwright_page.query_selector(config.element_selector)
            if element:
                await element.screenshot(**screenshot_options)
            else:
                raise ValueError(f"未找到元素: {config.element_selector}")
        else:
            await self.playwright_page.screenshot(**screenshot_options)
    
    async def _screenshot_with_selenium(self, 
                                      url: str, 
                                      output_path: Path, 
                                      config: ScreenshotConfig):
        """使用Selenium截图"""
        if not self.driver:
            raise RuntimeError("Selenium 未初始化")
        
        # 设置窗口大小
        self.driver.set_window_size(config.width, config.height)
        
        # 访问页面
        self.driver.get(url)
        
        # 等待页面加载
        await asyncio.sleep(config.wait_time)
        
        # 隐藏指定元素
        for selector in config.hide_elements:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    self.driver.execute_script("arguments[0].style.display = 'none';", element)
            except Exception as e:
                self.logger.warning(f"隐藏元素失败 {selector}: {e}")
        
        # 截图
        if config.element_selector:
            try:
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, config.element_selector))
                )
                element.screenshot(str(output_path))
            except Exception as e:
                raise ValueError(f"元素截图失败 {config.element_selector}: {e}")
        else:
            if config.full_page:
                # 全页面截图
                total_height = self.driver.execute_script("return document.body.scrollHeight")
                self.driver.set_window_size(config.width, total_height)
                await asyncio.sleep(1)
            
            self.driver.save_screenshot(str(output_path))
    
    def preprocess_image(self, 
                        image_path: Path, 
                        enhance: bool = True,
                        denoise: bool = True,
                        resize_factor: float = 1.0) -> Path:
        """
        图片预处理
        
        Args:
            image_path: 图片路径
            enhance: 是否增强图片
            denoise: 是否降噪
            resize_factor: 缩放因子
            
        Returns:
            处理后的图片路径
        """
        try:
            # 读取图片
            image = Image.open(image_path)
            
            # 缩放
            if resize_factor != 1.0:
                new_size = (int(image.width * resize_factor), 
                           int(image.height * resize_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转换为RGB（如果需要）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 图片增强
            if enhance:
                # 对比度增强
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                # 锐度增强
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
            
            # 降噪处理
            if denoise and cv2 is not None:
                # 转换为OpenCV格式
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # 高斯模糊降噪
                cv_image = cv2.GaussianBlur(cv_image, (3, 3), 0)
                
                # 转换回PIL格式
                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 保存处理后的图片
            processed_path = image_path.parent / f"processed_{image_path.name}"
            image.save(processed_path, quality=95, optimize=True)
            
            self.logger.info(f"图片预处理完成: {processed_path}")
            return processed_path
            
        except Exception as e:
            self.logger.error(f"图片预处理失败: {e}")
            return image_path  # 返回原图片
    
    async def extract_text_ocr(self, 
                              image_path: Path, 
                              config: OCRConfig = None) -> Dict[str, Any]:
        """
        OCR文字识别
        
        Args:
            image_path: 图片路径
            config: OCR配置
            
        Returns:
            OCR结果字典
        """
        if config is None:
            config = OCRConfig()
        
        # 预处理图片
        if config.preprocess:
            processed_image_path = self.preprocess_image(image_path)
        else:
            processed_image_path = image_path
        
        # 选择OCR引擎
        if config.engine == "auto":
            # 自动选择可用的OCR引擎
            if self.ocr_engines.get('tesseract', False):
                config.engine = "tesseract"
            elif self.ocr_engines.get('easyocr', False):
                config.engine = "easyocr"
            elif self.ocr_engines.get('paddleocr', False):
                config.engine = "paddleocr"
            else:
                raise RuntimeError("没有可用的OCR引擎")
        
        # 执行OCR
        try:
            if config.engine == "tesseract":
                result = await self._ocr_with_tesseract(processed_image_path, config)
            elif config.engine == "easyocr":
                result = await self._ocr_with_easyocr(processed_image_path, config)
            elif config.engine == "paddleocr":
                result = await self._ocr_with_paddleocr(processed_image_path, config)
            else:
                raise ValueError(f"不支持的OCR引擎: {config.engine}")
            
            self.logger.info(f"OCR识别完成，引擎: {config.engine}")
            return result
            
        except Exception as e:
            self.logger.error(f"OCR识别失败: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": config.engine,
                "error": str(e)
            }
    
    async def _ocr_with_tesseract(self, 
                                 image_path: Path, 
                                 config: OCRConfig) -> Dict[str, Any]:
        """使用Tesseract进行OCR"""
        if not pytesseract:
            raise RuntimeError("Tesseract 不可用")
        
        try:
            # 设置语言
            lang = '+'.join(config.languages)
            
            # 提取文本
            text = pytesseract.image_to_string(
                str(image_path), 
                lang=lang,
                config='--psm 1 --oem 3'
            )
            
            # 获取置信度信息
            data = pytesseract.image_to_data(
                str(image_path), 
                lang=lang,
                output_type=pytesseract.Output.DICT
            )
            
            # 计算平均置信度
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": text.strip(),
                "confidence": avg_confidence / 100.0,
                "engine": "tesseract",
                "word_count": len(text.split()),
                "character_count": len(text)
            }
            
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR 失败: {e}")
    
    async def _ocr_with_easyocr(self, 
                               image_path: Path, 
                               config: OCRConfig) -> Dict[str, Any]:
        """使用EasyOCR进行OCR"""
        if not easyocr:
            raise RuntimeError("EasyOCR 不可用")
        
        try:
            # 延迟初始化EasyOCR
            if self.easyocr_reader is None:
                self.easyocr_reader = easyocr.Reader(config.languages, gpu=False)
            
            # 执行OCR
            results = self.easyocr_reader.readtext(str(image_path))
            
            # 提取文本和置信度
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence >= config.confidence_threshold:
                    texts.append(text)
                    confidences.append(confidence)
            
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "engine": "easyocr",
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
                "details": results
            }
            
        except Exception as e:
            raise RuntimeError(f"EasyOCR 失败: {e}")
    
    async def _ocr_with_paddleocr(self, 
                                 image_path: Path, 
                                 config: OCRConfig) -> Dict[str, Any]:
        """使用PaddleOCR进行OCR"""
        if not paddleocr:
            raise RuntimeError("PaddleOCR 不可用")
        
        try:
            # 延迟初始化PaddleOCR
            if self.paddleocr_reader is None:
                # 根据语言设置选择模型
                lang = 'ch' if 'chi_sim' in config.languages else 'en'
                self.paddleocr_reader = paddleocr.PaddleOCR(
                    use_angle_cls=True, 
                    lang=lang,
                    show_log=False
                )
            
            # 执行OCR
            results = self.paddleocr_reader.ocr(str(image_path), cls=True)
            
            # 提取文本和置信度
            texts = []
            confidences = []
            
            for line in results[0] if results and results[0] else []:
                if line:
                    bbox, (text, confidence) = line
                    if confidence >= config.confidence_threshold:
                        texts.append(text)
                        confidences.append(confidence)
            
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "engine": "paddleocr",
                "word_count": len(full_text.split()),
                "character_count": len(full_text),
                "details": results
            }
            
        except Exception as e:
            raise RuntimeError(f"PaddleOCR 失败: {e}")
    
    def create_pdf_with_images_and_text(self, 
                                       images_and_texts: List[Tuple[Path, str]], 
                                       output_path: Path,
                                       config: PDFConfig = None) -> Path:
        """
        创建包含图片和OCR文本的PDF
        
        Args:
            images_and_texts: (图片路径, OCR文本) 的列表
            output_path: 输出PDF路径
            config: PDF配置
            
        Returns:
            PDF文件路径
        """
        if config is None:
            config = PDFConfig()
        
        try:
            # 设置页面大小
            if config.page_size == "A4":
                page_size = A4
            elif config.page_size == "letter":
                page_size = letter
            else:
                page_size = A4  # 默认A4
            
            # 创建PDF
            c = canvas.Canvas(str(output_path), pagesize=page_size)
            page_width, page_height = page_size
            
            # 设置PDF元数据
            if config.title:
                c.setTitle(config.title)
            if config.author:
                c.setAuthor(config.author)
            if config.subject:
                c.setSubject(config.subject)
            
            # 处理每个图片和文本
            for i, (image_path, ocr_text) in enumerate(images_and_texts):
                if i > 0:
                    c.showPage()  # 新页面
                
                # 添加图片
                try:
                    img = Image.open(image_path)
                    img_width, img_height = img.size
                    
                    # 计算缩放比例以适应页面
                    available_width = page_width - 2 * config.margin
                    available_height = page_height - 2 * config.margin
                    
                    scale_x = available_width / img_width
                    scale_y = available_height / img_height
                    scale = min(scale_x, scale_y, 1.0)  # 不放大
                    
                    new_width = img_width * scale
                    new_height = img_height * scale
                    
                    # 居中放置图片
                    x = (page_width - new_width) / 2
                    y = (page_height - new_height) / 2
                    
                    c.drawImage(str(image_path), x, y, new_width, new_height)
                    
                    # 如果启用OCR文本层，添加不可见文本
                    if config.include_ocr_text and ocr_text.strip():
                        # 设置文本为透明（不可见但可搜索）
                        c.setFillColorRGB(0, 0, 0, alpha=0)
                        c.setFont("Helvetica", 12)
                        
                        # 将OCR文本添加到页面底部（不可见）
                        text_lines = ocr_text.split('\n')
                        y_text = 50
                        for line in text_lines[:10]:  # 限制行数
                            if line.strip():
                                c.drawString(config.margin, y_text, line.strip())
                                y_text -= 15
                    
                except Exception as e:
                    self.logger.error(f"添加图片到PDF失败 {image_path}: {e}")
                    continue
            
            # 保存PDF
            c.save()
            
            # 如果需要创建可搜索PDF，使用OCR处理
            if config.searchable and config.include_ocr_text:
                searchable_path = self._create_searchable_pdf(output_path, images_and_texts)
                if searchable_path != output_path:
                    shutil.move(searchable_path, output_path)
            
            self.logger.info(f"PDF创建完成: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"PDF创建失败: {e}")
            raise
    
    def _create_searchable_pdf(self, 
                              pdf_path: Path, 
                              images_and_texts: List[Tuple[Path, str]]) -> Path:
        """创建可搜索的PDF（使用PyMuPDF）"""
        try:
            # 打开PDF
            doc = fitz.open(str(pdf_path))
            
            # 为每页添加文本层
            for page_num, (image_path, ocr_text) in enumerate(images_and_texts):
                if page_num < len(doc):
                    page = doc[page_num]
                    
                    # 添加OCR文本作为不可见文本层
                    if ocr_text.strip():
                        # 将文本分割成行
                        lines = ocr_text.split('\n')
                        y_pos = 50
                        
                        for line in lines:
                            if line.strip():
                                # 添加文本（设置为透明）
                                page.insert_text(
                                    (50, y_pos),
                                    line.strip(),
                                    fontsize=12,
                                    color=(1, 1, 1),  # 白色（不可见）
                                    overlay=False
                                )
                                y_pos += 15
            
            # 保存可搜索PDF
            searchable_path = pdf_path.parent / f"searchable_{pdf_path.name}"
            doc.save(str(searchable_path))
            doc.close()
            
            return searchable_path
            
        except Exception as e:
            self.logger.error(f"创建可搜索PDF失败: {e}")
            return pdf_path
    
    async def process_urls_to_pdf(self, 
                                 urls: List[str],
                                 output_path: Path,
                                 screenshot_config: ScreenshotConfig = None,
                                 ocr_config: OCRConfig = None,
                                 pdf_config: PDFConfig = None) -> Dict[str, Any]:
        """
        批量处理URL并生成PDF
        
        Args:
            urls: URL列表
            output_path: 输出PDF路径
            screenshot_config: 截图配置
            ocr_config: OCR配置
            pdf_config: PDF配置
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        
        if screenshot_config is None:
            screenshot_config = ScreenshotConfig()
        if ocr_config is None:
            ocr_config = OCRConfig()
        if pdf_config is None:
            pdf_config = PDFConfig()
        
        images_and_texts = []
        results = {
            "success": True,
            "total_urls": len(urls),
            "processed_urls": 0,
            "failed_urls": [],
            "ocr_results": [],
            "output_path": str(output_path),
            "processing_time": 0
        }
        
        try:
            for i, url in enumerate(urls):
                try:
                    self.logger.info(f"处理URL {i+1}/{len(urls)}: {url}")
                    
                    # 截图
                    screenshot_path = await self.screenshot_url(url, screenshot_config)
                    
                    # OCR识别
                    ocr_result = await self.extract_text_ocr(screenshot_path, ocr_config)
                    
                    # 添加到结果
                    images_and_texts.append((screenshot_path, ocr_result.get("text", "")))
                    results["ocr_results"].append({
                        "url": url,
                        "screenshot_path": str(screenshot_path),
                        "ocr_result": ocr_result
                    })
                    results["processed_urls"] += 1
                    
                except Exception as e:
                    self.logger.error(f"处理URL失败 {url}: {e}")
                    results["failed_urls"].append({"url": url, "error": str(e)})
                    continue
            
            # 创建PDF
            if images_and_texts:
                self.create_pdf_with_images_and_text(
                    images_and_texts, 
                    output_path, 
                    pdf_config
                )
                self.logger.info(f"批量处理完成，生成PDF: {output_path}")
            else:
                results["success"] = False
                results["error"] = "没有成功处理的URL"
            
        except Exception as e:
            self.logger.error(f"批量处理失败: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        finally:
            # 清理临时文件
            self._cleanup_temp_files()
            
            results["processing_time"] = time.time() - start_time
        
        return results
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for file_path in self.temp_dir.glob("screenshot_*"):
                file_path.unlink(missing_ok=True)
            for file_path in self.temp_dir.glob("processed_*"):
                file_path.unlink(missing_ok=True)
            self.logger.info("临时文件清理完成")
        except Exception as e:
            self.logger.warning(f"临时文件清理失败: {e}")


# 示例使用函数
async def main():
    """主函数示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试URL列表
    urls = [
        "https://www.example.com",
        "https://www.github.com",
        "https://www.stackoverflow.com"
    ]
    
    # 输出路径
    output_path = Path("网页截图合成.pdf")
    
    # 配置
    screenshot_config = ScreenshotConfig(
        width=1920,
        height=1080,
        full_page=True,
        wait_time=3
    )
    
    ocr_config = OCRConfig(
        engine="auto",
        languages=["eng", "chi_sim"],
        confidence_threshold=0.6
    )
    
    pdf_config = PDFConfig(
        page_size="A4",
        include_ocr_text=True,
        searchable=True,
        title="网页截图合集",
        author="Web Screenshot PDF"
    )
    
    # 处理
    async with WebScreenshotPDF(browser_engine="playwright") as processor:
        result = await processor.process_urls_to_pdf(
            urls=urls,
            output_path=output_path,
            screenshot_config=screenshot_config,
            ocr_config=ocr_config,
            pdf_config=pdf_config
        )
        
        print(f"处理结果: {json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())