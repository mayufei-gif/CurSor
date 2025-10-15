"""
网页截图PDF工具启动脚本

提供简单的启动选项，方便用户快速使用工具。
"""

import sys
import os
import subprocess
from pathlib import Path


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    网页截图PDF生成器                          ║
║                Web Screenshot PDF Generator                  ║
╠══════════════════════════════════════════════════════════════╣
║  🌐 支持多种浏览器引擎 (Selenium, Playwright)                ║
║  🔤 集成多种OCR引擎 (Tesseract, EasyOCR, PaddleOCR)         ║
║  📄 生成可搜索PDF文档                                        ║
║  🚀 提供命令行、GUI和API接口                                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python版本: {sys.version.split()[0]}")


def check_dependencies():
    """检查基本依赖"""
    required_packages = [
        'click',
        'asyncio',
        'pathlib',
        'PIL',
        'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True


def show_menu():
    """显示主菜单"""
    menu = """
请选择运行模式:

1. 🖥️  启动图形界面 (GUI)
2. 💻 使用命令行工具 (CLI)
3. 🔄 点击翻页抓取 (新功能!)
4. 🧪 运行测试示例
5. ⚙️  生成示例配置文件
6. 🔍 检查系统环境
7. 📁 打开输出目录
8. 📚 查看帮助文档
9. 🚪 退出

请输入选项 (1-9): """
    
    return input(menu).strip()


def run_gui():
    """启动GUI界面"""
    print("🖥️  启动图形界面...")
    try:
        subprocess.run([sys.executable, "gui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ GUI启动失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到gui.py文件")


def run_cli():
    """运行命令行工具"""
    print("💻 命令行工具选项:")
    print("1. 处理单个URL")
    print("2. 批量处理URL")
    print("3. 仅截图")
    print("4. 仅OCR识别")
    print("5. 查看CLI帮助")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        url = input("请输入URL: ").strip()
        output = input("输出文件名 (默认: output.pdf): ").strip() or "output.pdf"
        
        cmd = [sys.executable, "cli.py", "single", url, "-o", output]
        
        # 询问是否启用OCR
        ocr_choice = input("是否启用OCR? (y/n, 默认: y): ").strip().lower()
        if ocr_choice == "n":
            cmd.append("--no-ocr")
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif choice == "2":
        urls_file = input("URL列表文件路径: ").strip()
        if not Path(urls_file).exists():
            print("❌ 文件不存在")
            return
        
        output = input("输出文件名 (默认: batch.pdf): ").strip() or "batch.pdf"
        
        cmd = [sys.executable, "cli.py", "batch", urls_file, "-o", output]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif choice == "3":
        url = input("请输入URL: ").strip()
        output = input("输出图片名 (默认: screenshot.png): ").strip() or "screenshot.png"
        
        cmd = [sys.executable, "cli.py", "screenshot", url, "-o", output]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif choice == "4":
        image_path = input("图片文件路径: ").strip()
        if not Path(image_path).exists():
            print("❌ 文件不存在")
            return
        
        cmd = [sys.executable, "cli.py", "ocr", image_path]
        
        output = input("输出文本文件 (可选): ").strip()
        if output:
            cmd.extend(["-o", output])
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif choice == "5":
        subprocess.run([sys.executable, "cli.py", "--help"])
    
    else:
        print("❌ 无效选择")

def run_clicks_capture():
    """运行点击翻页抓取"""
    print("🔄 点击翻页抓取配置:")
    
    # 获取基本参数
    start_url = input("起始URL: ").strip()
    if not start_url:
        print("❌ URL不能为空")
        return
    
    next_selector = input("下一页按钮选择器 (可选): ").strip()
    image_selector = input("图片元素选择器 (可选): ").strip()
    
    use_arrow_keys = input("使用键盘箭头键翻页? (y/n, 默认: n): ").strip().lower() == 'y'
    interactive_crop = input("启用交互式裁剪? (y/n, 默认: y): ").strip().lower() != 'n'
    
    pdf_title = input("PDF标题 (默认: 点击翻页抓取): ").strip() or "点击翻页抓取"
    max_pages = input("最大页面数 (默认: 100): ").strip() or "100"
    
    auto_ocr = input("自动执行OCR? (y/n, 默认: n): ").strip().lower() == 'y'
    keep_images = input("保留临时图片? (y/n, 默认: n): ").strip().lower() == 'y'
    
    # 构建命令
    cmd = [
        sys.executable, "cli.py", "capture-by-clicks",
        "--start-url", start_url,
        "--pdf-title", pdf_title,
        "--max-pages", max_pages
    ]
    
    if next_selector:
        cmd.extend(["--next-selector", next_selector])
    
    if image_selector:
        cmd.extend(["--image-selector", image_selector])
    
    if use_arrow_keys:
        cmd.append("--use-arrow-keys")
    
    if not interactive_crop:
        cmd.append("--no-interactive-crop")
    
    if auto_ocr:
        cmd.append("--auto-ocr")
    
    if keep_images:
        cmd.append("--keep-images")
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("⚠️  注意：此功能需要Playwright引擎，请确保已安装")
    
    confirm = input("确认执行? (y/n): ").strip().lower()
    if confirm == 'y':
        subprocess.run(cmd)
    else:
        print("❌ 已取消")

def open_output_directories():
    """打开输出目录"""
    try:
        from config import get_config
        config = get_config()
        
        print("📁 打开输出目录...")
        
        # 创建目录（如果不存在）
        config.pdf.output_dir.mkdir(parents=True, exist_ok=True)
        config.pdf.ocr_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PDF输出目录: {config.pdf.output_dir}")
        print(f"OCR输出目录: {config.pdf.ocr_output_dir}")
        
        # 打开目录
        if os.name == 'nt':  # Windows
            os.startfile(str(config.pdf.output_dir))
            os.startfile(str(config.pdf.ocr_output_dir))
            print("✅ 已在文件管理器中打开目录")
        else:  # Linux/Mac
            subprocess.run(['xdg-open', str(config.pdf.output_dir)], check=False)
            subprocess.run(['xdg-open', str(config.pdf.ocr_output_dir)], check=False)
            print("✅ 已在文件管理器中打开目录")
            
    except Exception as e:
        print(f"❌ 打开目录失败: {e}")


def run_test():
    """运行测试示例"""
    print("🧪 运行测试示例...")
    try:
        subprocess.run([sys.executable, "test_example.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到test_example.py文件")


def generate_config():
    """生成示例配置文件"""
    print("⚙️  生成示例配置文件...")
    try:
        subprocess.run([sys.executable, "cli.py", "config-example"], check=True)
        print("✅ 示例配置文件已生成: config.example.json")
        print("💡 可以复制为 config.json 并根据需要修改")
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成配置文件失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到cli.py文件")


def check_environment():
    """检查系统环境"""
    print("🔍 检查系统环境...")
    try:
        subprocess.run([sys.executable, "cli.py", "check", "--check-all"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 环境检查失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到cli.py文件")


def show_help():
    """显示帮助信息"""
    help_text = """
📚 网页截图PDF工具帮助

🎯 主要功能:
  • 网页自动截图
  • 🔄 点击翻页抓取 (新功能!)
  • 🖱️  交互式裁剪
  • OCR文字识别
  • PDF文档生成
  • 批量处理支持

🚀 快速开始:
  1. 安装依赖: pip install -r requirements.txt
  2. 安装浏览器: playwright install chromium
  3. 安装OCR引擎: 参考README.md
  4. 运行工具: python run.py

💻 命令行使用:
  • 单个URL: python cli.py single "https://example.com"
  • 批量处理: python cli.py batch urls.txt
  • 🔄 点击翻页: python cli.py capture-by-clicks --start-url "https://example.com/page/1"
  • 仅截图: python cli.py screenshot "https://example.com"
  • OCR识别: python cli.py ocr image.png

🖥️  图形界面:
  • 启动GUI: python gui.py
  • 简单易用的图形界面
  • 支持拖拽操作
  • 🔄 点击翻页标签页

🔄 点击翻页功能:
  • 自动点击下一页按钮
  • 支持键盘箭头键翻页
  • 🖱️  交互式裁剪区域选择
  • 懒加载滚动支持
  • 智能页面变化检测
  • 自动合并PDF + OCR

⚙️  配置文件:
  • 生成示例: python cli.py config-example
  • 配置文件: config.json
  • 环境变量支持

📁 输出目录:
  • PDF文件: G:\\E盘\\工作项目文件\\电子版涂书\\书籍\\PDF涂书
  • OCR文件: G:\\E盘\\工作项目文件\\电子版涂书\\论文\\OCRmyPDF

🔍 故障排除:
  • 检查环境: python cli.py check
  • 查看日志: --log-level DEBUG
  • 常见问题: 参考README.md

📖 更多信息:
  • 项目文档: README.md
  • 示例代码: test_example.py
  • 配置说明: config.example.json
"""
    print(help_text)


def main():
    """主函数"""
    print_banner()
    
    # 检查Python版本
    check_python_version()
    
    # 检查基本依赖
    print("\n🔍 检查基本依赖...")
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装所需依赖")
        sys.exit(1)
    
    # 主循环
    while True:
        try:
            choice = show_menu()
            
            if choice == "1":
                run_gui()
            elif choice == "2":
                run_cli()
            elif choice == "3":
                run_clicks_capture()
            elif choice == "4":
                run_test()
            elif choice == "5":
                generate_config()
            elif choice == "6":
                check_environment()
            elif choice == "7":
                open_output_directories()
            elif choice == "8":
                show_help()
            elif choice == "9":
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请输入1-9")
            
            # 询问是否继续
            if choice in ["1", "2", "3", "4", "5", "6"]:
                continue_choice = input("\n按Enter继续，或输入'q'退出: ").strip().lower()
                if continue_choice == 'q':
                    print("👋 再见!")
                    break
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            continue


if __name__ == "__main__":
    main()