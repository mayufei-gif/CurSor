# Windows PowerShell 网页图书PDF化完整操作指南

## 🎯 概述
本指南详细说明如何在Windows PowerShell环境下使用网页图书PDF化工具，将在线图书转换为高质量PDF文件。

## 📋 系统要求
- Windows 10/11 操作系统
- PowerShell 5.0 或更高版本
- Google Chrome 浏览器
- Python 3.8+ 环境
- 至少 2GB 可用磁盘空间

## 🚀 完整操作流程

### 第一步：环境准备

#### 1.1 打开PowerShell
```powershell
# 方法1：按 Win + R，输入 powershell，回车
# 方法2：在开始菜单搜索 "PowerShell"
# 方法3：在文件资源管理器地址栏输入 powershell
```

#### 1.2 切换到项目目录
```powershell
cd "g:\E盘\工作项目文件\AI_Agent\Trae_Abroad\网页图片PDF"
```

#### 1.3 验证环境
```powershell
# 检查Python版本
python --version

# 检查项目文件
ls cli.py

# 检查Chrome是否安装
Get-Process chrome -ErrorAction SilentlyContinue
```

### 第二步：启动Chrome调试模式

#### 2.1 关闭现有Chrome进程
```powershell
# 强制关闭所有Chrome进程
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force
```

#### 2.2 启动Chrome调试模式
```powershell
# 启动Chrome并开启远程调试
Start-Process "chrome.exe" -ArgumentList "--remote-debugging-port=9222", "--user-data-dir=G:\E盘\工作项目文件\AI_Agent\Trae_Abroad\.chrome_profiles\IA"
```

#### 2.3 验证调试端口
```powershell
# 等待2秒让Chrome完全启动
Start-Sleep -Seconds 2

# 验证调试端口是否可用
Invoke-WebRequest -Uri "http://localhost:9222/json" -UseBasicParsing
```

### 第三步：启动浏览器控制程序

#### 3.1 启动控制程序
```powershell
python cli.py browser --connect-existing --debug-port 9222 --start-url blank --viewport-w 1600 --viewport-h 1200
```

#### 3.2 程序启动成功标志
看到以下输出表示启动成功：
```
🚀 启动通用浏览器控制
✅ 成功连接到现有Chrome实例
✅ 浏览器已启动，保持空白页面
🌟 浏览器已准备就绪！

📋 请选择操作：
   1. 截图当前页面
   2. 🚀 自动化批量截图 (20页/批次)
   3. 继续在浏览器中操作
   4. 完成并退出
```

### 第四步：网页操作准备

#### 4.1 在Chrome浏览器中操作
1. **导航到图书网站**：在地址栏输入图书网站URL
2. **登录账户**：输入用户名和密码登录
3. **找到目标图书**：搜索并打开要下载的图书
4. **调整页面**：
   - 设置合适的缩放比例（建议100%-125%）
   - 确保图书内容完整显示
   - 导航到第一页

#### 4.2 页面准备检查清单
- ✅ 图书页面已完全加载
- ✅ 页面缩放比例合适
- ✅ 图书内容清晰可见
- ✅ 位于要截图的起始页面

### 第五步：执行截图操作

#### 5.1 选择操作模式
在PowerShell终端中输入选择：

**推荐：自动化批量截图**
```
请输入选择 (1/2/3/4): 2
```

#### 5.2 自动化批量截图流程

**步骤1：选择截图区域**
```
📸 开始批量截图...
🖼️ 正在生成预览截图...
✅ 预览截图已保存
📋 请在弹出的窗口中选择要截图的区域
```
- 在弹出的图片选择窗口中，用鼠标拖拽选择图书内容区域
- 避免选择页面边框、导航栏等无关内容
- 点击"确认选择"按钮

**步骤2：自动批量处理**
```
✅ 截图区域已选择: (x1, y1) -> (x2, y2)
🚀 开始批次 1 的自动化截图...
📸 正在截取第 1 页...
⏭️ 自动翻页到下一页...
📸 正在截取第 2 页...
...
📸 正在截取第 20 页...
✅ 批次 1 完成！共截取 20 页
```

**步骤3：批次控制**
每完成20页后，程序会询问：
```
📋 批次控制选项：
   1. 继续下一批次 (20页)
   2. 调整截图区域后继续
   3. 终止截图并生成PDF
   4. 仅合并当前批次为PDF

请选择 (1/2/3/4):
```

#### 5.3 操作选择说明

**选择1 - 继续下一批次**：
- 使用相同截图区域继续截取下20页
- 适用于图书格式一致的情况

**选择2 - 调整截图区域**：
- 重新选择截图区域
- 适用于图书格式发生变化的情况

**选择3 - 终止并生成PDF**：
- 停止截图，将所有已截取的图片合并为一个PDF
- 适用于图书截取完成的情况

**选择4 - 仅合并当前批次**：
- 只将当前批次的20页合并为PDF
- 继续截取下一批次

### 第六步：文件输出和管理

#### 6.1 输出文件位置
```
📁 图片文件保存位置：
G:\E盘\工作项目文件\电子版涂书\书籍\原图\batch_X\

📁 PDF文件保存位置：
G:\E盘\工作项目文件\电子版涂书\书籍\PDF涂书\
```

#### 6.2 文件命名规则
```
图片文件：page_001.png, page_002.png, ...
PDF文件：complete_book_YYYYMMDD_HHMMSS.pdf
```

#### 6.3 验证输出文件
```powershell
# 检查图片文件
ls "G:\E盘\工作项目文件\电子版涂书\书籍\原图"

# 检查PDF文件
ls "G:\E盘\工作项目文件\电子版涂书\书籍\PDF涂书"
```

## 🔧 一键启动脚本使用

### 使用启动脚本
```powershell
# 执行一键启动脚本
.\启动脚本.ps1
```

### 脚本功能
- 自动检查环境
- 自动关闭现有Chrome进程
- 自动启动Chrome调试模式
- 自动启动浏览器控制程序

## ⚠️ 重要注意事项

### 操作注意事项
1. **不要关闭Chrome浏览器窗口**：程序运行期间保持Chrome窗口开启
2. **不要手动刷新页面**：可能导致连接中断
3. **确保网络稳定**：避免图片加载失败
4. **磁盘空间充足**：每页图片约1-5MB

### 翻页机制说明
- 程序使用键盘右箭头键模拟翻页
- 每次翻页后等待2秒确保页面加载完成
- 支持大多数在线图书阅读器

### 截图质量优化
- 建议使用1600x1200或更高分辨率
- 页面缩放比例100%-125%最佳
- 确保图书内容完整显示在可视区域

## 🛠️ 故障排除

### 常见问题及解决方案

#### 问题1：Chrome启动失败
```powershell
# 解决方案：手动指定Chrome路径
$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
Start-Process $chromePath -ArgumentList "--remote-debugging-port=9222"
```

#### 问题2：端口占用
```powershell
# 检查端口占用
netstat -ano | findstr :9222

# 终止占用进程
taskkill /PID <进程ID> /F
```

#### 问题3：连接失败
```powershell
# 重新启动整个流程
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
# 重新执行启动命令
```

#### 问题4：截图区域选择失败
- 确保安装了tkinter：`pip install tk`
- 确保安装了Pillow：`pip install Pillow`
- 重新选择截图区域

#### 问题5：自动翻页失败
- 检查网页是否支持键盘导航
- 确认当前页面焦点在图书阅读区域
- 手动点击图书内容区域后重试

## 📊 性能指标

### 处理速度
- 单页截图：2-5秒
- 20页批次：约2-3分钟
- PDF生成：10-30秒（取决于页数）

### 资源占用
- 内存使用：约200-500MB
- CPU使用：中等（截图时较高）
- 磁盘IO：中等

## 🔄 完整操作示例

```powershell
# 1. 切换目录
cd "g:\E盘\工作项目文件\AI_Agent\Trae_Abroad\网页图片PDF"

# 2. 关闭Chrome
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force

# 3. 启动Chrome调试模式
Start-Process "chrome.exe" -ArgumentList "--remote-debugging-port=9222", "--user-data-dir=G:\E盘\工作项目文件\AI_Agent\Trae_Abroad\.chrome_profiles\IA"

# 4. 等待启动
Start-Sleep -Seconds 3

# 5. 启动控制程序
python cli.py browser --connect-existing --debug-port 9222 --start-url blank --viewport-w 1600 --viewport-h 1200

# 6. 在程序中选择操作
# 输入: 2 (自动化批量截图)

# 7. 选择截图区域并开始自动化处理
```

## 📞 技术支持

如遇到问题，请检查：
1. Python和依赖包是否正确安装
2. Chrome浏览器版本是否兼容
3. 网络连接是否稳定
4. 磁盘空间是否充足

---

**版本信息**：v2.0  
**更新日期**：2025年1月  
**兼容性**：Windows 10/11, Chrome 90+, Python 3.8+