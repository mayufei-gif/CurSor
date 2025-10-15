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
import re
import csv
import subprocess
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
    import img2pdf
except ImportError as e:
    print(f"PDF处理库导入失败: {e}")
    print("请安装: pip install reportlab PyMuPDF img2pdf")


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
    
    async def _wait_image_loaded_playwright(self, page, image_selector: str, prev_src: Optional[str] = None, timeout: int = 10000):
        """等待图片加载完成（Playwright）"""
        try:
            # 优先等待 src 变化
            if prev_src:
                try:
                    await page.wait_for_function(
                        """(sel, prev) => {
                            const el = document.querySelector(sel);
                            return el && el.src && el.src !== prev && el.complete && el.naturalWidth > 10;
                        }""",
                        image_selector, prev_src, timeout=timeout
                    )
                    return
                except:
                    pass
            
            # 兜底：只要加载完成
            await page.wait_for_function(
                """(sel) => {
                    const el = document.querySelector(sel);
                    return el && el.complete && el.naturalWidth > 10;
                }""",
                image_selector, timeout=timeout
            )
        except Exception as e:
            self.logger.warning(f"等待图片加载超时: {e}")
    
    async def _scroll_lazy_until_stable_playwright(self, page, timeout: int = 3000):
        """懒加载滚动直到页面稳定（Playwright）"""
        try:
            end = time.time() + (timeout / 1000)
            last_height = -1
            stable_count = 0
            
            while time.time() < end:
                # 获取当前页面高度
                current_height = await page.evaluate("() => document.documentElement.scrollHeight")
                
                if current_height == last_height:
                    stable_count += 1
                    if stable_count >= 3:  # 连续3次高度不变，认为稳定
                        break
                else:
                    stable_count = 0
                
                last_height = current_height
                
                # 滚动一个视口高度
                await page.evaluate("() => window.scrollBy(0, window.innerHeight)")
                await asyncio.sleep(0.3)
                
            # 滚动回顶部
            await page.evaluate("() => window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)
            
        except Exception as e:
            self.logger.warning(f"懒加载滚动失败: {e}")
    
    # 交互式裁剪的JavaScript代码
    PICK_BOX_JS = """() => new Promise(resolve => {
        const overlay = document.createElement('div');
        Object.assign(overlay.style, {
            position: 'fixed',
            left: '0',
            top: '0',
            right: '0',
            bottom: '0',
            zIndex: '2147483647',
            cursor: 'crosshair',
            background: 'rgba(0,0,0,0.1)'
        });
        document.body.appendChild(overlay);
        
        let startX = 0, startY = 0, rect = null;
        
        overlay.addEventListener('mousedown', e => {
            startX = e.clientX;
            startY = e.clientY;
            rect = document.createElement('div');
            Object.assign(rect.style, {
                position: 'fixed',
                border: '2px dashed #ff0000',
                background: 'rgba(255,0,0,0.08)',
                left: startX + 'px',
                top: startY + 'px',
                zIndex: '2147483648',
                width: '0px',
                height: '0px',
                pointerEvents: 'none'
            });
            document.body.appendChild(rect);
            overlay.addEventListener('mousemove', onMove);
            overlay.addEventListener('mouseup', onUp);
        });
        
        function onMove(e) {
            const x = Math.min(e.clientX, startX);
            const y = Math.min(e.clientY, startY);
            const w = Math.abs(e.clientX - startX);
            const h = Math.abs(e.clientY - startY);
            Object.assign(rect.style, {
                left: x + 'px',
                top: y + 'px',
                width: w + 'px',
                height: h + 'px'
            });
        }
        
        function onUp(e) {
            overlay.removeEventListener('mousemove', onMove);
            overlay.removeEventListener('mouseup', onUp);
            const r = rect.getBoundingClientRect();
            
            const msg = document.createElement('div');
            Object.assign(msg.style, {
                position: 'fixed',
                bottom: '20px',
                right: '20px',
                padding: '10px 15px',
                background: '#333',
                color: '#fff',
                borderRadius: '6px',
                zIndex: '2147483648',
                fontFamily: 'Arial, sans-serif',
                fontSize: '14px'
            });
            msg.textContent = '按 Enter 确认选择，按 Esc 取消';
            document.body.appendChild(msg);
            
            function cleanup() {
                [overlay, rect, msg].forEach(n => n && n.remove());
            }
            
            function onKey(ev) {
                if (ev.key === 'Enter') {
                    cleanup();
                    resolve({
                        left: r.left,
                        top: r.top,
                        width: r.width,
                        height: r.height
                    });
                } else if (ev.key === 'Escape') {
                    cleanup();
                    resolve(null);
                }
                window.removeEventListener('keydown', onKey);
            }
            window.addEventListener('keydown', onKey);
        }
    });"""
    
    async def pick_crop_box(self, page):
        """交互式选择裁剪区域"""
        try:
            self.logger.info("请在页面上拖拽选择裁剪区域，按Enter确认，按Esc取消")
            coords = await page.evaluate(self.PICK_BOX_JS)
            
            if not coords:
                self.logger.info("用户取消了裁剪区域选择")
                return None
            
            # 获取设备像素比
            dpr = await page.evaluate("() => window.devicePixelRatio") or 1
            
            # 转换为实际像素坐标
            left = int(round(coords['left'] * dpr))
            top = int(round(coords['top'] * dpr))
            right = left + int(round(coords['width'] * dpr))
            bottom = top + int(round(coords['height'] * dpr))
            
            crop_box = (left, top, right, bottom)
            self.logger.info(f"选择的裁剪区域: {crop_box}")
            return crop_box
            
        except Exception as e:
            self.logger.error(f"交互式裁剪失败: {e}")
            return None
    
    async def _smart_screenshot_playwright(self, page, output_path: Path, config: ScreenshotConfig):
        """智能截图（Playwright）：元素优先，否则全页+裁剪"""
        try:
            # 优先尝试元素截图
            if config.image_selector:
                element = await page.query_selector(config.image_selector)
                if element:
                    self.logger.info(f"使用元素截图: {config.image_selector}")
                    await element.screenshot(path=str(output_path))
                    return output_path
                else:
                    self.logger.warning(f"未找到指定元素: {config.image_selector}")
            
            # 全页截图
            self.logger.info("使用全页截图")
            if config.crop_box:
                # 需要裁剪，先截图到临时文件
                temp_path = output_path.parent / f"temp_full_{output_path.name}"
                await page.screenshot(path=str(temp_path), full_page=True)
                
                # 裁剪图片
                with Image.open(temp_path) as img:
                    cropped = img.crop(config.crop_box)
                    cropped.save(output_path)
                
                # 删除临时文件
                temp_path.unlink(missing_ok=True)
                self.logger.info(f"已裁剪图片: {config.crop_box}")
            else:
                # 直接全页截图
                await page.screenshot(path=str(output_path), full_page=config.full_page)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"智能截图失败: {e}")
            raise
    
    async def capture_by_clicks(self, start_url: str, output_dir: Path, config: ScreenshotConfig,
                                max_pages: int = 2000, mapping_csv: Optional[Path] = None):
        """点击翻页循环截图"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if mapping_csv is None:
            mapping_csv = output_dir / "mapping.csv"
        
        # 写入CSV头部
        header_written = mapping_csv.exists()
        with mapping_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not header_written:
                writer.writerow(["seq", "page_num", "filename", "url", "timestamp"])
        
        if self.browser_engine != "playwright":
            raise ValueError("点击翻页功能目前只支持Playwright引擎")
        
        page = self.playwright_page
        await page.goto(start_url, wait_until="networkidle")
        await asyncio.sleep(0.4)
        
        # 交互式裁剪（如果启用且未设置裁剪区域）
        if config.interactive_crop and not config.crop_box:
            self.logger.info("启动交互式裁剪...")
            crop_box = await self.pick_crop_box(page)
            if crop_box:
                config.crop_box = crop_box
                self.logger.info(f"设置裁剪区域: {crop_box}")
        
        seq = 1
        processed_urls = set()  # 防止重复处理相同URL
        
        for page_index in range(max_pages):
            try:
                current_url = page.url
                
                # 检查是否已处理过此URL
                if current_url in processed_urls:
                    self.logger.warning(f"检测到重复URL，可能已到末页: {current_url}")
                    break
                
                processed_urls.add(current_url)
                
                # 懒加载处理
                if config.lazy_scroll:
                    await self._scroll_lazy_until_stable_playwright(page, int(config.lazy_timeout * 1000))
                
                # 等待图片加载
                prev_src = None
                if config.image_selector:
                    element = await page.query_selector(config.image_selector)
                    if element:
                        prev_src = await element.get_attribute("src")
                        await self._wait_image_loaded_playwright(
                            page, config.image_selector, 
                            timeout=config.image_load_timeout * 1000
                        )
                
                # 解析页码
                page_num = self._extract_page_number(current_url)
                filename = f"{page_num:04d}.png" if page_num else f"{seq:04d}.png"
                output_path = output_dir / filename
                
                # 截图（如果文件不存在）
                if not output_path.exists():
                    await self._smart_screenshot_playwright(page, output_path, config)
                    
                    # 记录到CSV
                    with mapping_csv.open("a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            seq, page_num or "", str(output_path), 
                            current_url, datetime.utcnow().isoformat()
                        ])
                    
                    self.logger.info(f"[{seq}] 已保存: {output_path}")
                else:
                    self.logger.info(f"[{seq}] 跳过已存在: {output_path}")
                
                # 尝试翻到下一页
                if not await self._navigate_to_next_page(page, config, prev_src):
                    self.logger.info("无法翻到下一页，结束处理")
                    break
                
                seq += 1
                
            except Exception as e:
                self.logger.error(f"处理第{seq}页时出错: {e}")
                break
        
        self.logger.info(f"翻页截图完成，共处理 {seq} 页")
        return output_dir
    
    async def _navigate_to_next_page(self, page, config: ScreenshotConfig, prev_src: Optional[str] = None) -> bool:
        """导航到下一页"""
        try:
            # 记录当前状态
            prev_url = page.url
            
            # 方法1: 点击下一页按钮
            if config.next_selector:
                next_button = await page.query_selector(config.next_selector)
                if next_button:
                    self.logger.debug(f"点击下一页按钮: {config.next_selector}")
                    await next_button.click()
                    
                    # 等待页面变化
                    if await self._wait_for_page_change(page, prev_url, prev_src, config):
                        return True
            
            # 方法2: 使用键盘箭头键
            if config.use_arrow_keys:
                self.logger.debug("使用右箭头键翻页")
                await page.keyboard.press("ArrowRight")
                
                # 等待页面变化
                if await self._wait_for_page_change(page, prev_url, prev_src, config):
                    return True
            
            self.logger.warning("所有翻页方法都失败")
            return False
            
        except Exception as e:
            self.logger.error(f"翻页失败: {e}")
            return False
    
    async def _wait_for_page_change(self, page, prev_url: str, prev_src: Optional[str], config: ScreenshotConfig) -> bool:
        """等待页面变化"""
        try:
            # 优先等待图片src变化
            if config.image_selector and prev_src:
                try:
                    await self._wait_image_loaded_playwright(
                        page, config.image_selector, prev_src, 
                        timeout=config.page_change_timeout * 1000
                    )
                    return True
                except:
                    pass
            
            # 兜底：等待URL变化
            try:
                await page.wait_for_function(
                    "oldUrl => location.href !== oldUrl", 
                    prev_url, 
                    timeout=config.page_change_timeout * 1000
                )
                return True
            except:
                pass
            
            # 最后尝试：短暂等待
            await asyncio.sleep(0.5)
            return page.url != prev_url
            
        except Exception as e:
            self.logger.warning(f"等待页面变化失败: {e}")
            return False
    
    def _extract_page_number(self, url: str) -> Optional[int]:
        """从URL中提取页码"""
        patterns = [
            r'/n(\d+)(?:/|$)',      # /n123/ 或 /n123
            r'[?&]page=(\d+)',      # ?page=123 或 &page=123
            r'[?&]p=(\d+)',         # ?p=123 或 &p=123
            r'#page=(\d+)',         # #page=123
            r'/page/(\d+)',         # /page/123
            r'/(\d+)(?:/|$)',       # /123/ 或 /123 (通用数字)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return int(match.group(1))
        
        return None
    
    async def process_clicks_to_pdf(self, start_url: str, screenshot_config: ScreenshotConfig, pdf_config: PDFConfig):
        """完整的点击翻页到PDF工作流"""
        try:
            # 第一步：点击翻页截图
            self.logger.info("开始点击翻页截图...")
            temp_dir = Path(tempfile.mkdtemp(prefix="web_shots_"))
            mapping_csv = temp_dir / "mapping.csv"
            
            images_dir = await self.capture_by_clicks(
                start_url, temp_dir, screenshot_config, 
                max_pages=pdf_config.max_pages, mapping_csv=mapping_csv
            )
            
            # 获取所有图片文件
            image_files = sorted(images_dir.glob("*.png"))
            if not image_files:
                raise ValueError("没有找到任何截图文件")
            
            self.logger.info(f"共截取 {len(image_files)} 张图片")
            
            # 第二步：预览（如果启用）
            if pdf_config.enable_preview:
                self.logger.info("打开文件夹进行预览...")
                if os.name == 'nt':  # Windows
                    os.startfile(str(images_dir))
                else:  # Linux/Mac
                    subprocess.run(['xdg-open', str(images_dir)], check=False)
                
                input("请检查截图是否正确、顺序是否合适。按 Enter 继续合并为PDF...")
            
            # 第三步：合并PDF
            self.logger.info("开始合并PDF...")
            pdf_config.output_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_filename = pdf_config.title or "merged_book.pdf"
            if not pdf_filename.endswith('.pdf'):
                pdf_filename += '.pdf'
            
            pdf_path = pdf_config.output_dir / pdf_filename
            
            # 使用img2pdf合并
            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert([str(img) for img in image_files]))
            
            self.logger.info(f"PDF已保存到: {pdf_path}")
            
            # 第四步：询问是否进行OCR
            perform_ocr = pdf_config.auto_ocr
            if not perform_ocr and pdf_config.enable_preview:
                response = input("是否对PDF进行OCR文字识别？(y/N): ").strip().lower()
                perform_ocr = response in ['y', 'yes', '是']
            
            ocr_path = None
            if perform_ocr:
                self.logger.info("开始OCR处理...")
                pdf_config.ocr_output_dir.mkdir(parents=True, exist_ok=True)
                
                ocr_filename = pdf_config.title or "merged_book_ocr.pdf"
                if not ocr_filename.endswith('.pdf'):
                    ocr_filename += '.pdf'
                if not ocr_filename.endswith('_ocr.pdf'):
                    ocr_filename = ocr_filename.replace('.pdf', '_ocr.pdf')
                
                ocr_path = pdf_config.ocr_output_dir / ocr_filename
                
                # 使用OCRmyPDF进行OCR
                cmd = [
                    'ocrmypdf',
                    '-l', 'chi_sim+eng',  # 中英文
                    '--deskew',           # 纠正倾斜
                    '--clean',            # 清理图像
                    '--optimize', '1',    # 优化级别
                    str(pdf_path),
                    str(ocr_path)
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    self.logger.info(f"OCR PDF已保存到: {ocr_path}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"OCR处理失败: {e}")
                    self.logger.error(f"错误输出: {e.stderr}")
                    ocr_path = None
                except FileNotFoundError:
                    self.logger.error("OCRmyPDF未安装，请安装: pip install ocrmypdf")
                    ocr_path = None
            
            # 第五步：清理临时文件（如果不保留）
            if not pdf_config.keep_temp_images:
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.info("已清理临时文件")
                except Exception as e:
                    self.logger.warning(f"清理临时文件失败: {e}")
            
            # 返回结果
            result = {
                "success": True,
                "images_dir": str(images_dir) if pdf_config.keep_temp_images else None,
                "pdf_path": str(pdf_path),
                "ocr_path": str(ocr_path) if ocr_path else None,
                "image_count": len(image_files),
                "mapping_csv": str(mapping_csv) if pdf_config.keep_temp_images else None
            }
            
            self.logger.info("PDF工作流完成！")
            self.logger.info(f"原始PDF: {pdf_path}")
            if ocr_path:
                self.logger.info(f"OCR PDF: {ocr_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"PDF工作流失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def open_output_directories(self, pdf_config: PDFConfig):
        """打开输出目录"""
        try:
            if os.name == 'nt':  # Windows
                if pdf_config.output_dir.exists():
                    os.startfile(str(pdf_config.output_dir))
                if pdf_config.ocr_output_dir.exists():
                    os.startfile(str(pdf_config.ocr_output_dir))
            else:  # Linux/Mac
                if pdf_config.output_dir.exists():
                    subprocess.run(['xdg-open', str(pdf_config.output_dir)], check=False)
                if pdf_config.ocr_output_dir.exists():
                    subprocess.run(['xdg-open', str(pdf_config.ocr_output_dir)], check=False)
        except Exception as e:
            self.logger.error(f"打开目录失败: {e}")
    
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