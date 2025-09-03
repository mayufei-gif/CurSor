# PDF-MCP Server

一个基于 Model Context Protocol (MCP) 的强大PDF处理服务器，提供文本提取、表格识别、OCR、数学公式识别和智能文档分析功能。

## 🚀 特性

### 核心功能
- **文本提取**: 支持 PyMuPDF 和 pdfplumber 两种引擎
- **表格提取**: 集成 Camelot 和 Tabula 进行精确表格识别
- **OCR识别**: 基于 Tesseract 和 OCRmyPDF 的光学字符识别
- **公式识别**: 使用 LaTeX-OCR 和 pix2tex 进行数学公式提取
- **文档分析**: 智能检测文档类型和结构
- **完整管道**: 一站式PDF处理解决方案

### 技术特点
- **MCP协议**: 完全兼容 Model Context Protocol 标准
- **异步处理**: 高性能异步架构
- **智能路由**: 根据文档类型自动选择最佳处理策略
- **批量处理**: 支持多文件并发处理
- **可配置**: 灵活的配置系统
- **容器化**: 完整的 Docker 支持

## 📦 安装

### 系统依赖
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr ghostscript poppler-utils

# macOS
brew install tesseract ghostscript poppler

# Windows
# 请下载并安装 Tesseract OCR 和 Ghostscript
```

### Python环境
```bash
# 克隆项目
git clone <repository-url>
cd pdf-mcp-server

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 或使用开发模式安装
pip install -e .[dev]
```

### GPU支持（可选）
```bash
# 安装GPU版本以加速公式识别
pip install -e .[gpu]
```

## 🔧 配置

### 环境变量
```bash
# 可选：设置模型缓存目录
export TRANSFORMERS_CACHE=/path/to/cache

# 可选：设置Tesseract路径（Windows）
export TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# 可选：启用GROBID服务
export GROBID_URL="http://localhost:8070"
```

### 配置文件
创建 `config.yaml`：
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  debug: false

processing:
  max_file_size: 100MB
  temp_dir: "/tmp/pdf_mcp"
  
ocr:
  language: "eng+chi_sim"  # Tesseract语言包
  dpi: 300
  
formula:
  model: "pix2tex"  # 或 "latex-ocr"
  device: "auto"  # "cpu", "cuda", "auto"
  
tables:
  primary_engine: "camelot"  # "camelot" 或 "tabula"
  fallback_engine: "tabula"
```

## 🚀 使用方法

### 启动服务器
```bash
# 直接启动
python -m pdf_mcp_server.main

# 或使用uvicorn
uvicorn pdf_mcp_server.main:app --host 0.0.0.0 --port 8000

# 开发模式
uvicorn pdf_mcp_server.main:app --reload
```

### MCP客户端调用
```python
import asyncio
from mcp import ClientSession

async def main():
    async with ClientSession("http://localhost:8000") as session:
        # 提取文本
        result = await session.call_tool("read_text", {
            "file_path": "/path/to/document.pdf",
            "pages": [1, 2, 3]  # 可选：指定页面
        })
        
        # 提取表格
        tables = await session.call_tool("extract_tables", {
            "file_path": "/path/to/document.pdf",
            "output_format": "json"  # "csv", "json", "dataframe"
        })
        
        # 识别公式
        formulas = await session.call_tool("extract_formulas", {
            "file_path": "/path/to/document.pdf",
            "confidence_threshold": 0.8
        })
        
        # 完整处理
        full_result = await session.call_tool("full_pipeline", {
            "file_path": "/path/to/document.pdf",
            "include_ocr": True,
            "include_formulas": True,
            "include_grobid": False
        })

asyncio.run(main())
```

### REST API调用
```bash
# 上传并处理PDF
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F "mode=full_pipeline" \
  -F "include_formulas=true"

# 处理本地文件
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "mode": "extract_tables",
    "output_format": "json"
  }'
```

## 📊 输出格式

### 统一JSON响应
```json
{
  "status": "success",
  "file_info": {
    "filename": "document.pdf",
    "pages": 10,
    "file_size": 1024000,
    "is_scanned": false
  },
  "content": {
    "text": {
      "full_text": "文档完整文本...",
      "pages": [
        {"page": 1, "text": "第一页文本...", "bbox": [...]},
        {"page": 2, "text": "第二页文本...", "bbox": [...]}
      ]
    },
    "tables": [
      {
        "page": 1,
        "table_id": 0,
        "bbox": [100, 200, 400, 300],
        "data": [["列1", "列2"], ["数据1", "数据2"]],
        "confidence": 0.95
      }
    ],
    "formulas": [
      {
        "page": 2,
        "formula_id": 0,
        "bbox": [150, 250, 350, 280],
        "latex": "E = mc^2",
        "confidence": 0.92,
        "image_path": "/tmp/formula_0.png"
      }
    ],
    "metadata": {
      "title": "文档标题",
      "author": "作者",
      "creation_date": "2024-01-01",
      "processing_time": 2.5,
      "engines_used": ["pdfplumber", "camelot", "pix2tex"]
    }
  }
}
```

## 🐳 Docker部署

```bash
# 构建镜像
docker build -t pdf-mcp-server .

# 运行容器
docker run -d \
  --name pdf-mcp \
  -p 8000:8000 \
  -v /path/to/pdfs:/app/data \
  -v /path/to/cache:/app/cache \
  pdf-mcp-server

# 使用docker-compose
docker-compose up -d
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_text_extraction.py

# 生成覆盖率报告
pytest --cov=pdf_mcp_server --cov-report=html
```

## 📈 性能优化

### GPU加速
- 公式识别模型支持GPU加速
- 建议使用CUDA 11.8+和对应的PyTorch版本

### 内存管理
- 大文件自动分页处理
- 可配置临时文件清理策略
- 支持流式处理减少内存占用

### 并发处理
- 支持异步处理多个PDF文件
- 可配置工作进程数量
- 内置任务队列和优先级管理

## 🔒 安全考虑

- 文件大小限制和类型验证
- 路径遍历攻击防护
- 临时文件自动清理
- 可配置的文件访问白名单
- 敏感信息处理日志脱敏

## 🤝 贡献

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF文本提取
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF布局分析
- [Camelot](https://github.com/camelot-dev/camelot) - 表格提取
- [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) - 公式识别
- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) - OCR处理
- [GROBID](https://github.com/kermitt2/grobid) - 学术文档解析