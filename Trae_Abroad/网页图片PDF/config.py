#!/usr/bin/env python3
"""
配置管理模块

提供项目的配置管理功能，支持环境变量、配置文件和默认配置。
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class BrowserConfig:
    """浏览器配置"""
    engine: str = "playwright"  # selenium, playwright
    headless: bool = True
    window_width: int = 1920
    window_height: int = 1080
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    timeout: int = 30
    wait_time: int = 3
    
    # Chrome特定配置
    chrome_binary_path: Optional[str] = None
    chrome_driver_path: Optional[str] = None
    
    # Firefox特定配置
    firefox_binary_path: Optional[str] = None
    firefox_driver_path: Optional[str] = None


@dataclass
class ScreenshotConfig:
    """截图配置"""
    full_page: bool = False  # 改为False，优先使用元素截图
    quality: int = 95
    format: str = "PNG"  # PNG, JPEG, WEBP
    dpi: int = 300
    
    # 元素截图
    element_selector: Optional[str] = None
    hide_elements: List[str] = field(default_factory=list)
    
    # 移动端模拟
    mobile_emulation: bool = False
    device_name: Optional[str] = None
    
    # 图片处理
    auto_crop: bool = False
    crop_margin: int = 10
    resize_factor: float = 1.0
    
    # 新增：元素与翻页配置
    image_selector: Optional[str] = None  # 图片元素选择器
    next_selector: Optional[str] = None   # 下一页按钮选择器
    prev_selector: Optional[str] = None   # 上一页按钮选择器
    use_arrow_keys: bool = False          # 使用键盘左右键翻页
    
    # 新增：裁剪配置
    interactive_crop: bool = True         # 启用交互式裁剪
    crop_box: Optional[tuple] = None      # 裁剪区域 (left,top,right,bottom)
    
    # 新增：懒加载配置
    lazy_scroll: bool = True              # 启用懒加载滚动
    lazy_timeout: float = 3.0             # 懒加载等待超时时间（秒）
    
    # 新增：页面等待配置
    image_load_timeout: int = 10          # 图片加载超时时间（秒）
    page_change_timeout: int = 5          # 页面变化检测超时时间（秒）


@dataclass
class OCRConfig:
    """OCR配置"""
    engine: str = "auto"  # auto, tesseract, easyocr, paddleocr
    languages: List[str] = field(default_factory=lambda: ["eng", "chi_sim"])
    confidence_threshold: float = 0.6
    
    # 预处理
    preprocess: bool = True
    enhance_contrast: bool = True
    enhance_sharpness: bool = True
    denoise: bool = True
    
    # Tesseract特定配置
    tesseract_cmd: Optional[str] = None
    tesseract_config: str = "--psm 1 --oem 3"
    
    # EasyOCR特定配置
    easyocr_gpu: bool = False
    easyocr_model_storage_directory: Optional[str] = None
    
    # PaddleOCR特定配置
    paddleocr_use_angle_cls: bool = True
    paddleocr_use_gpu: bool = False


@dataclass
class PDFConfig:
    """PDF配置"""
    page_size: str = "A4"  # A4, letter, custom
    margin: int = 50
    
    # 文本层
    include_ocr_text: bool = True
    searchable: bool = True
    text_layer_opacity: float = 0.0  # 0.0 = 完全透明
    
    # 压缩
    compress_images: bool = True
    image_quality: int = 85
    
    # 元数据
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: str = "Web Screenshot PDF Tool"
    
    # 自定义页面大小（当page_size为custom时）
    custom_width: float = 595.276  # A4宽度（点）
    custom_height: float = 841.890  # A4高度（点）
    
    # 新增：Windows路径默认值
    output_dir: Path = Path(r"G:\E盘\工作项目文件\电子版涂书\书籍\PDF涂书")
    ocr_output_dir: Path = Path(r"G:\E盘\工作项目文件\电子版涂书\论文\OCRmyPDF")
    
    # 新增：批量处理配置
    max_pages: int = 2000             # 最大页面数限制
    enable_preview: bool = True       # 启用合并前预览
    auto_ocr: bool = False           # 自动执行OCR（不询问）
    keep_temp_images: bool = False   # 保留临时图片文件


@dataclass
class ProcessingConfig:
    """处理配置"""
    max_concurrent_tasks: int = 3
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # 超时设置
    screenshot_timeout: int = 30
    ocr_timeout: int = 60
    pdf_creation_timeout: int = 120
    
    # 临时文件
    temp_dir: Optional[str] = None
    cleanup_temp_files: bool = True
    keep_screenshots: bool = False
    
    # 日志
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class AppConfig:
    """应用主配置"""
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    screenshot: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file or Path("config.json")
        self._config = AppConfig()
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        # 1. 加载默认配置
        self._config = AppConfig()
        
        # 2. 从配置文件加载
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self._update_config_from_dict(config_data)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        
        # 3. 从环境变量加载
        self._load_from_env()
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """从字典更新配置"""
        if "browser" in config_data:
            self._update_dataclass(self._config.browser, config_data["browser"])
        
        if "screenshot" in config_data:
            self._update_dataclass(self._config.screenshot, config_data["screenshot"])
        
        if "ocr" in config_data:
            self._update_dataclass(self._config.ocr, config_data["ocr"])
        
        if "pdf" in config_data:
            self._update_dataclass(self._config.pdf, config_data["pdf"])
        
        if "processing" in config_data:
            self._update_dataclass(self._config.processing, config_data["processing"])
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """更新数据类对象"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 浏览器配置
        if os.getenv("BROWSER_ENGINE"):
            self._config.browser.engine = os.getenv("BROWSER_ENGINE")
        
        if os.getenv("BROWSER_HEADLESS"):
            self._config.browser.headless = os.getenv("BROWSER_HEADLESS").lower() == "true"
        
        # OCR配置
        if os.getenv("OCR_ENGINE"):
            self._config.ocr.engine = os.getenv("OCR_ENGINE")
        
        if os.getenv("OCR_LANGUAGES"):
            self._config.ocr.languages = os.getenv("OCR_LANGUAGES").split(",")
        
        if os.getenv("TESSERACT_CMD"):
            self._config.ocr.tesseract_cmd = os.getenv("TESSERACT_CMD")
        
        # 处理配置
        if os.getenv("MAX_CONCURRENT_TASKS"):
            self._config.processing.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS"))
        
        if os.getenv("LOG_LEVEL"):
            self._config.processing.log_level = os.getenv("LOG_LEVEL")
        
        if os.getenv("TEMP_DIR"):
            self._config.processing.temp_dir = os.getenv("TEMP_DIR")
    
    def save_config(self, config_file: Optional[Path] = None):
        """保存配置到文件"""
        file_path = config_file or self.config_file
        
        try:
            config_dict = {
                "browser": asdict(self._config.browser),
                "screenshot": asdict(self._config.screenshot),
                "ocr": asdict(self._config.ocr),
                "pdf": asdict(self._config.pdf),
                "processing": asdict(self._config.processing)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"配置已保存到: {file_path}")
            
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def get_config(self) -> AppConfig:
        """获取配置"""
        return self._config
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                if isinstance(value, dict):
                    self._update_dataclass(getattr(self._config, key), value)
                else:
                    setattr(self._config, key, value)
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self._config = AppConfig()
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 验证浏览器引擎
        if self._config.browser.engine not in ["selenium", "playwright"]:
            errors.append(f"不支持的浏览器引擎: {self._config.browser.engine}")
        
        # 验证OCR引擎
        if self._config.ocr.engine not in ["auto", "tesseract", "easyocr", "paddleocr"]:
            errors.append(f"不支持的OCR引擎: {self._config.ocr.engine}")
        
        # 验证页面大小
        if self._config.pdf.page_size not in ["A4", "letter", "custom"]:
            errors.append(f"不支持的页面大小: {self._config.pdf.page_size}")
        
        # 验证图片格式
        if self._config.screenshot.format not in ["PNG", "JPEG", "WEBP"]:
            errors.append(f"不支持的图片格式: {self._config.screenshot.format}")
        
        # 验证数值范围
        if not 0 <= self._config.ocr.confidence_threshold <= 1:
            errors.append("OCR置信度阈值必须在0-1之间")
        
        if not 0 <= self._config.pdf.text_layer_opacity <= 1:
            errors.append("PDF文本层透明度必须在0-1之间")
        
        return errors


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """获取全局配置"""
    return config_manager.get_config()


def update_config(**kwargs):
    """更新全局配置"""
    config_manager.update_config(**kwargs)


def save_config(config_file: Optional[Path] = None):
    """保存全局配置"""
    config_manager.save_config(config_file)


def load_config(config_file: Path):
    """加载配置文件"""
    global config_manager
    config_manager = ConfigManager(config_file)


# 示例配置文件生成
def create_example_config():
    """创建示例配置文件"""
    example_config = {
        "browser": {
            "engine": "playwright",
            "headless": True,
            "window_width": 1920,
            "window_height": 1080,
            "timeout": 30,
            "wait_time": 3
        },
        "screenshot": {
            "full_page": True,
            "quality": 95,
            "format": "PNG",
            "dpi": 300,
            "hide_elements": [
                ".advertisement",
                ".popup",
                "#cookie-banner"
            ]
        },
        "ocr": {
            "engine": "auto",
            "languages": ["eng", "chi_sim"],
            "confidence_threshold": 0.6,
            "preprocess": True,
            "tesseract_config": "--psm 1 --oem 3"
        },
        "pdf": {
            "page_size": "A4",
            "margin": 50,
            "include_ocr_text": True,
            "searchable": True,
            "compress_images": True,
            "image_quality": 85,
            "title": "网页截图合集",
            "author": "Web Screenshot PDF Tool"
        },
        "processing": {
            "max_concurrent_tasks": 3,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "cleanup_temp_files": True,
            "log_level": "INFO"
        }
    }
    
    with open("config.example.json", 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
    
    print("示例配置文件已创建: config.example.json")


if __name__ == "__main__":
    # 创建示例配置
    create_example_config()
    
    # 测试配置管理器
    config = get_config()
    print(f"当前配置: {config}")
    
    # 验证配置
    errors = config_manager.validate_config()
    if errors:
        print(f"配置错误: {errors}")
    else:
        print("配置验证通过")