# 网页截图并合成PDF（含OCR）完整解决方案

一个功能强大的网页截图工具，支持自动截图、OCR文字识别和PDF合成，提供命令行、GUI和API多种使用方式。

## 🌟 主要特性

### 📸 网页截图
- **多浏览器引擎支持**: Selenium、Playwright
- **智能截图模式**: 全页面、可视区域、指定元素
- **高质量输出**: 支持PNG、JPEG、WEBP格式
- **自定义配置**: 窗口大小、等待时间、隐藏元素

### 🔤 OCR文字识别
- **多OCR引擎**: Tesseract、EasyOCR、PaddleOCR
- **多语言支持**: 中文、英文等多种语言
- **智能预处理**: 图像增强、降噪、对比度调整
- **高精度识别**: 可配置置信度阈值

### 📄 PDF生成
- **可搜索PDF**: 嵌入OCR文本层，支持全文搜索
- **灵活布局**: A4、Letter等多种页面大小
- **批量合成**: 多页面自动合并
- **元数据支持**: 标题、作者、主题等信息

### 🚀 使用方式
- **命令行工具**: 适合自动化脚本和批处理
- **图形界面**: 简单易用的GUI界面
- **Python API**: 集成到其他项目中

## 📦 安装

### 1. 克隆项目
```bash
git clone <repository-url>
cd 网页图片PDF
```

### 2. 安装Python依赖
```bash
pip install -r requirements.txt
```

### 3. 安装浏览器驱动

#### Playwright (推荐)
```bash
playwright install chromium
```

#### Selenium
下载对应浏览器的WebDriver：
- [ChromeDriver](https://chromedriver.chromium.org/)
- [GeckoDriver (Firefox)](https://github.com/mozilla/geckodriver/releases)

### 4. 安装OCR引擎

#### Tesseract
**Windows:**
1. 下载安装 [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. 添加到系统PATH或设置环境变量 `TESSERACT_CMD`

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

#### EasyOCR & PaddleOCR
```bash
pip install easyocr paddleocr
```

## 🚀 快速开始

### 命令行使用

#### 单个URL处理
```bash
python cli.py single "https://www.example.com" -o output.pdf
```

#### 批量处理
```bash
# 创建URL列表文件
echo "https://www.example.com" > urls.txt
echo "https://www.github.com" >> urls.txt

# 批量处理
python cli.py batch urls.txt -o batch_output.pdf
```

#### 仅截图
```bash
python cli.py screenshot "https://www.example.com" -o screenshot.png
```

#### OCR识别
```bash
python cli.py ocr image.png -o extracted_text.txt
```

### GUI界面使用
```bash
python gui.py
```

### Python API使用
```python
import asyncio
from pathlib import Path
from web_screenshot_pdf import WebScreenshotPDF, ScreenshotConfig, OCRConfig, PDFConfig

async def main():
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
        title="网页截图合集"
    )
    
    # 处理
    urls = ["https://www.example.com", "https://www.github.com"]
    
    async with WebScreenshotPDF(browser_engine="playwright") as processor:
        result = await processor.process_urls_to_pdf(
            urls=urls,
            output_path=Path("output.pdf"),
            screenshot_config=screenshot_config,
            ocr_config=ocr_config,
            pdf_config=pdf_config
        )
        
        print(f"处理完成: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚙️ 配置

### 生成示例配置文件
```bash
python cli.py config-example
```

### 配置文件示例 (config.json)
```json
{
  "browser": {
    "engine": "playwright",
    "headless": true,
    "window_width": 1920,
    "window_height": 1080,
    "timeout": 30,
    "wait_time": 3
  },
  "screenshot": {
    "full_page": true,
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
    "preprocess": true,
    "tesseract_config": "--psm 1 --oem 3"
  },
  "pdf": {
    "page_size": "A4",
    "margin": 50,
    "include_ocr_text": true,
    "searchable": true,
    "compress_images": true,
    "image_quality": 85,
    "title": "网页截图合集",
    "author": "Web Screenshot PDF Tool"
  },
  "processing": {
    "max_concurrent_tasks": 3,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "cleanup_temp_files": true,
    "log_level": "INFO"
  }
}
```

### 环境变量配置
```bash
# 浏览器配置
export BROWSER_ENGINE=playwright
export BROWSER_HEADLESS=true

# OCR配置
export OCR_ENGINE=tesseract
export OCR_LANGUAGES=eng,chi_sim
export TESSERACT_CMD=/usr/bin/tesseract

# 处理配置
export MAX_CONCURRENT_TASKS=3
export LOG_LEVEL=INFO
export TEMP_DIR=/tmp/web_screenshot
```

## 📚 Internet Archive 专用适配器

针对 Internet Archive (archive.org) 网站的专门适配器，支持自动翻页截图和PDF生成。

### 特性
- **手动登录**: 首屏暂停，支持手动登录/借阅操作
- **交互式裁剪**: 首张图片鼠标框选页面区域，后续自动应用
- **多种翻页方式**: URL递增、键盘方向键、滚轮翻页
- **智能等待**: 每页等待图片加载稳定后截图
- **自动合并**: 截图完成后自动合并PDF，可选OCR

### 使用方法

#### 基本用法
```bash
# 处理 Internet Archive 书籍
python cli.py ia --id isbn_9780965083409 --start-n 7

# 指定结束页面
python cli.py ia --id isbn_9780965083409 --start-n 7 --end-n 387

# 使用键盘方向键翻页
python cli.py ia --id isbn_9780965083409 --start-n 7 --mode arrow

# 使用滚轮翻页
python cli.py ia --id isbn_9780965083409 --start-n 7 --mode scroll
```

#### 参数说明
- `--id`: Internet Archive 的 identifier（必需）
- `--start-n`: 起始页面编号（默认: 7）
- `--end-n`: 结束页面编号（默认: 自动检测）
- `--mode`: 翻页方式（url/arrow/scroll，默认: url）
- `--viewport-w`: 浏览器窗口宽度（默认: 1600）
- `--viewport-h`: 浏览器窗口高度（默认: 1200）

#### 工作流程
1. **首次运行**: 浏览器打开目标页面，暂停等待手动操作
2. **手动设置**: 登录/借阅 → 切换到单页模式(1up) → 缩放并居中显示第一页
3. **交互式裁剪**: 弹出首张截图，用鼠标拖拽选择页面区域，回车确认
4. **自动处理**: 程序自动翻页截图，应用统一裁剪区域
5. **预览合并**: 截图完成后打开图片目录预览，确认后合并PDF
6. **可选OCR**: 询问是否执行OCR，生成可搜索PDF

#### 输出目录
- **原图**: `G:\E盘\工作项目文件\电子版涂书\书籍\原图\<id>\`
- **PDF**: `G:\E盘\工作项目文件\电子版涂书\书籍\PDF涂书\<id>.pdf`
- **OCR PDF**: `G:\E盘\工作项目文件\电子版涂书\论文\OCRmyPDF\<id>_OCR.pdf`

### 注意事项
- 需要 Python 内置的 tkinter 支持交互式裁剪
- 首次使用需要手动登录，程序不会保存登录状态
- 建议在良好的网络环境下使用，避免图片加载失败
- 支持断点续传，可以通过调整 `--start-n` 参数继续未完成的任务

## 📋 命令行参考

### 全局选项
- `--config, -c`: 指定配置文件路径
- `--log-level`: 设置日志级别 (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: 指定日志文件路径

### single 命令
处理单个URL并生成PDF。

```bash
python cli.py single [OPTIONS] URL
```

**选项:**
- `--output, -o`: 输出PDF文件路径 (默认: screenshot.pdf)
- `--browser`: 浏览器引擎 (selenium/playwright)
- `--full-page/--viewport`: 全页面截图或仅视口 (默认: 全页面)
- `--width`: 浏览器窗口宽度 (默认: 1920)
- `--height`: 浏览器窗口高度 (默认: 1080)
- `--wait`: 页面加载等待时间，秒 (默认: 3)
- `--quality`: 图片质量 1-100 (默认: 95)
- `--format`: 图片格式 PNG/JPEG/WEBP (默认: PNG)
- `--ocr-engine`: OCR引擎 (auto/tesseract/easyocr/paddleocr)
- `--ocr-lang`: OCR语言，逗号分隔 (如: eng,chi_sim)
- `--no-ocr`: 禁用OCR
- `--title`: PDF标题
- `--author`: PDF作者

### batch 命令
批量处理URL列表。

```bash
python cli.py batch [OPTIONS] URLS_FILE
```

**选项:**
- `--output, -o`: 输出PDF文件路径 (默认: batch_screenshots.pdf)
- `--browser`: 浏览器引擎
- `--concurrent`: 并发任务数
- `--delay`: URL之间的延迟，秒
- `--retry`: 重试次数
- `--no-ocr`: 禁用OCR
- `--title`: PDF标题
- `--author`: PDF作者
- `--report`: 生成处理报告文件路径

### screenshot 命令
仅截图，不进行OCR和PDF生成。

```bash
python cli.py screenshot [OPTIONS] URL
```

**选项:**
- `--output, -o`: 输出图片文件路径 (默认: screenshot.png)
- `--browser`: 浏览器引擎
- `--full-page/--viewport`: 全页面截图或仅视口
- `--width`: 浏览器窗口宽度
- `--height`: 浏览器窗口高度
- `--wait`: 页面加载等待时间，秒
- `--element`: 截图特定元素的CSS选择器
- `--hide`: 隐藏元素的CSS选择器 (可多次使用)

### ocr 命令
对图片进行OCR文字识别。

```bash
python cli.py ocr [OPTIONS] IMAGE_PATH
```

**选项:**
- `--engine`: OCR引擎 (auto/tesseract/easyocr/paddleocr)
- `--lang`: OCR语言，逗号分隔
- `--output, -o`: 输出文本文件路径
- `--confidence`: 置信度阈值 0.0-1.0

### 工具命令
- `config-example`: 生成示例配置文件
- `check`: 检查系统环境和依赖
  - `--check-browser`: 检查浏览器环境
  - `--check-ocr`: 检查OCR引擎
  - `--check-all`: 检查所有依赖

## 🔧 高级功能

### 自定义元素截图
```bash
# 截图特定元素
python cli.py screenshot "https://example.com" --element ".main-content" -o content.png

# 隐藏广告元素
python cli.py screenshot "https://example.com" --hide ".ad" --hide ".popup" -o clean.png
```

### 批量处理配置
```bash
# 高并发处理
python cli.py batch urls.txt --concurrent 5 --retry 3 -o output.pdf

# 添加延迟避免被限制
python cli.py batch urls.txt --delay 2 -o output.pdf
```

### OCR语言配置
```bash
# 中英文混合识别
python cli.py single "https://example.com" --ocr-lang "eng,chi_sim" -o output.pdf

# 仅英文识别
python cli.py single "https://example.com" --ocr-lang "eng" -o output.pdf
```

## 🐛 故障排除

### 常见问题

#### 1. 浏览器驱动问题
```bash
# 检查浏览器环境
python cli.py check --check-browser

# Playwright安装浏览器
playwright install chromium

# Selenium下载对应版本的WebDriver
```

#### 2. OCR引擎问题
```bash
# 检查OCR环境
python cli.py check --check-ocr

# Tesseract路径问题 (Windows)
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

#### 3. 内存不足
```bash
# 减少并发数
python cli.py batch urls.txt --concurrent 1

# 分批处理大量URL
split -l 10 large_urls.txt batch_
```

#### 4. 网络超时
```bash
# 增加等待时间
python cli.py single "https://slow-site.com" --wait 10

# 配置文件中设置超时
{
  "browser": {
    "timeout": 60,
    "wait_time": 10
  }
}
```

### 日志调试
```bash
# 启用详细日志
python cli.py --log-level DEBUG single "https://example.com"

# 保存日志到文件
python cli.py --log-file debug.log single "https://example.com"
```

## 📊 性能优化

### 1. 浏览器选择
- **Playwright**: 更快的启动速度，更好的现代网页支持
- **Selenium**: 更广泛的浏览器支持，更成熟的生态

### 2. OCR引擎选择
- **Tesseract**: 最成熟，支持最多语言
- **EasyOCR**: 更好的中文支持，无需额外配置
- **PaddleOCR**: 最快的处理速度，优秀的中文识别

### 3. 并发配置
```json
{
  "processing": {
    "max_concurrent_tasks": 3,  // 根据CPU核心数调整
    "retry_attempts": 3,
    "retry_delay": 1.0
  }
}
```

### 4. 图片优化
```json
{
  "screenshot": {
    "quality": 85,  // 降低质量减少文件大小
    "format": "JPEG",  // JPEG比PNG文件更小
    "dpi": 200  // 降低DPI减少处理时间
  }
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd 网页图片PDF

# 安装开发依赖
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8

# 运行测试
pytest

# 代码格式化
black .

# 代码检查
flake8 .
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Playwright](https://playwright.dev/) - 现代网页自动化
- [Selenium](https://selenium.dev/) - 网页自动化框架
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - OCR引擎
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - 简单易用的OCR
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 高性能OCR
- [ReportLab](https://www.reportlab.com/) - PDF生成库
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF处理库

## 📞 支持

如有问题或建议，请：
1. 查看 [FAQ](#故障排除)
2. 搜索现有 [Issues](../../issues)
3. 创建新的 [Issue](../../issues/new)

---

**Happy Screenshot & OCR! 📸🔤📄**