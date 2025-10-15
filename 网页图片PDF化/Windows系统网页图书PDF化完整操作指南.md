# Windows系统网页图书PDF化完整操作指南

## 📋 目录
1. [系统要求](#系统要求)
2. [环境准备](#环境准备)
3. [快速启动](#快速启动)
4. [详细操作流程](#详细操作流程)
5. [功能说明](#功能说明)
6. [故障排除](#故障排除)
7. [输出文件说明](#输出文件说明)

## 🖥️ 系统要求

### 必需软件
- **Windows 10/11** (推荐)
- **Python 3.8+** (推荐 3.12)
- **Google Chrome浏览器** (最新版本)
- **PowerShell 5.0+** (Windows自带)

### Python依赖包
程序会自动检查以下依赖：
- `playwright` - 浏览器自动化
- `Pillow (PIL)` - 图像处理
- `click` - 命令行界面
- `tkinter` - GUI界面 (Python自带)

## 🔧 环境准备

### 1. 解压项目文件
将 `网页图片PDF` 文件夹解压到任意位置，例如：
```
D:\Tools\网页图片PDF\
```

### 2. 安装Python依赖
打开PowerShell，切换到项目目录：
```powershell
cd "D:\Tools\网页图片PDF"
pip install -r requirements.txt
```

### 3. 安装Playwright浏览器
```powershell
playwright install chromium
```

## 🚀 快速启动

### 方法一：一键启动脚本
1. 打开PowerShell
2. 切换到项目目录：
   ```powershell
   cd "D:\Tools\网页图片PDF"
   ```
3. 执行启动命令：
   ```powershell
   # 关闭现有Chrome进程
   Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force
   
   # 启动Chrome调试模式
   & "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=".\temp_chrome_debug"
   
   # 等待3秒让Chrome完全启动
   Start-Sleep -Seconds 3
   
   # 验证调试端口
   Invoke-WebRequest http://localhost:9222/json/version | Select-Object -ExpandProperty StatusDescription
   
   # 启动浏览器控制程序
   python cli.py browser --connect-existing --debug-port 9222 --start-url blank --viewport-w 1600 --viewport-h 1200
   ```

### 方法二：分步启动
1. **启动Chrome调试模式**
   ```powershell
   & "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=".\temp_chrome_debug"
   ```

2. **验证连接**（新开PowerShell窗口）
   ```powershell
   cd "D:\Tools\网页图片PDF"
   Invoke-WebRequest http://localhost:9222/json/version
   ```
   看到 `StatusCode: 200` 表示成功

3. **启动控制程序**
   ```powershell
   python cli.py browser --connect-existing --debug-port 9222 --start-url blank --viewport-w 1600 --viewport-h 1200
   ```

## 📖 详细操作流程

### 第一步：程序启动
执行启动命令后，你会看到：
```
🚀 启动通用浏览器控制
📋 起始URL: blank
📐 视口大小: 1600x1200
✅ 成功连接到现有Chrome实例
✅ 浏览器已启动，保持空白页面

🌟 浏览器已准备就绪！
📋 请选择操作：
   1. 截图当前页面
   2. 🚀 自动化批量截图 (20页/批次)
   3. 继续在浏览器中操作
   4. 完成并退出

请输入选择 (1/2/3/4):
```

### 第二步：准备网页图书
1. 在Chrome浏览器中：
   - 输入图书网站URL
   - 登录账户
   - 找到要截图的书籍
   - 调整页面缩放到合适比例（推荐67%）
   - 确保书籍内容清晰可见

2. 回到终端，选择操作模式

### 第三步：选择截图模式

#### 模式1：单页截图
- 输入 `1` - 截图当前页面
- 程序会生成预览图
- 在弹出的窗口中框选书籍内容区域
- 程序自动放大并截取高质量图片
- 可选择继续截图更多页面

#### 模式2：自动化批量截图（推荐）
- 输入 `2` - 自动化批量截图
- 确认开始：输入 `y`
- 第一次会让你选择截图区域
- 程序自动执行：
  - 截图当前页面
  - 使用键盘右键翻页
  - 重复20次（一个批次）
- 批次完成后选择：
  - `1` - 继续下一批次
  - `2` - 调整截图区域后继续
  - `3` - 终止截图
  - `4` - 合并所有批次为一个PDF

### 第四步：文件输出
程序会自动：
1. 按顺序重命名图片：`page_001.png`, `page_002.png`...
2. 保存到批次文件夹：`G:\E盘\工作项目文件\电子版涂书\书籍\原图\batch_01_20250115_143022\`
3. 生成PDF文件：`G:\E盘\工作项目文件\电子版涂书\书籍\PDF涂书\batch_01_20250115_143022.pdf`
4. 可选OCR处理：`G:\E盘\工作项目文件\电子版涂书\论文\OCRmyPDF\`

## 🎛️ 功能说明

### 自动化批量截图特点
- **区域记忆**：第一次选择区域后，后续页面自动使用相同区域
- **智能放大**：自动计算最佳缩放比例，确保文字清晰
- **批次管理**：每20页一个批次，便于管理大量页面
- **自动翻页**：使用键盘右键模拟翻页操作
- **容错机制**：翻页失败或截图失败时的处理
- **文件组织**：按时间戳创建文件夹，图片按顺序命名

### 截图质量优化
- 自动检测书籍区域
- 智能放大确保文字清晰
- 支持高分辨率截图
- 自动恢复页面原始状态

### 文件管理
- 自动创建目录结构
- 按批次组织文件
- 支持PDF合并
- 可选OCR文字识别

## 🔧 故障排除

### Chrome路径问题
如果提示找不到Chrome，尝试以下路径：
```powershell
# 64位系统
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

# 32位系统或特殊安装
& "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

# 用户目录安装
& "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

### 端口被占用
```powershell
# 使用不同端口
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9223

# 对应修改启动命令
python cli.py browser --connect-existing --debug-port 9223 --start-url blank
```

### 连接失败
```powershell
# 检查端口状态
netstat -an | findstr 9222

# 重启Chrome
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force
# 然后重新启动Chrome调试模式
```

### Python依赖问题
```powershell
# 重新安装依赖
pip install --upgrade -r requirements.txt

# 安装Playwright
playwright install chromium

# 检查Python版本
python --version
```

### GUI界面问题
如果截图区域选择窗口无法显示：
- 确保安装了tkinter（Python自带）
- 检查显示器分辨率设置
- 尝试以管理员身份运行PowerShell

## 📁 输出文件说明

### 目录结构
```
G:\E盘\工作项目文件\电子版涂书\
├── 书籍\
│   ├── 原图\                    # 原始截图文件
│   │   ├── batch_01_20250115_143022\
│   │   │   ├── page_001.png
│   │   │   ├── page_002.png
│   │   │   └── ...
│   │   └── batch_02_20250115_144530\
│   └── PDF涂书\                 # PDF文件
│       ├── batch_01_20250115_143022.pdf
│       ├── batch_02_20250115_144530.pdf
│       └── complete_book_20250115_150000.pdf
└── 论文\
    └── OCRmyPDF\               # OCR处理后的PDF
        └── complete_book_ocr_20250115_150000.pdf
```

### 文件命名规则
- **图片文件**：`page_XXX.png`（XXX为3位数字，如001、002）
- **批次PDF**：`batch_XX_YYYYMMDD_HHMMSS.pdf`
- **完整PDF**：`complete_book_YYYYMMDD_HHMMSS.pdf`
- **OCR PDF**：`complete_book_ocr_YYYYMMDD_HHMMSS.pdf`

## 🎯 使用技巧

### 最佳实践
1. **页面准备**：
   - 调整浏览器缩放到67%（推荐）
   - 确保书籍内容完整显示
   - 关闭不必要的浏览器插件

2. **截图区域选择**：
   - 尽量选择书籍主要内容区域
   - 避免包含页面边框和导航栏
   - 选择区域要稍大一些，程序会自动优化

3. **批量截图**：
   - 建议每次处理20-50页
   - 可以分多个批次进行
   - 最后统一合并为完整PDF

4. **文件管理**：
   - 定期清理临时文件
   - 备份重要的PDF文件
   - 使用有意义的文件名

### 高级功能
- **OCR文字识别**：支持中英文混合识别
- **PDF合并**：可将多个批次合并为一个文件
- **自定义输出目录**：可修改配置文件指定输出路径
- **截图质量调整**：可在配置中调整图片质量和格式

## 📞 技术支持

### 常见问题
1. **程序无响应**：按Ctrl+C中断，重新启动
2. **截图质量差**：调整浏览器缩放比例
3. **翻页失败**：检查网页是否支持键盘翻页
4. **文件过大**：可以分批次处理，最后合并

### 配置文件
可以修改 `config.py` 文件来自定义：
- 输出目录路径
- 截图质量设置
- OCR语言设置
- 浏览器参数

---

## 📝 版本信息
- **版本**：2.0
- **更新日期**：2025年1月
- **兼容性**：Windows 10/11, Python 3.8+
- **作者**：AI Assistant

---

**注意**：使用本工具时请遵守相关网站的使用条款和版权法律法规。