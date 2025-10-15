#!/usr/bin/env python3
"""
Chrome浏览器测试脚本

测试Selenium Chrome是否能正常工作，并进行简单的网页截图测试。
"""

import asyncio
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from web_screenshot_pdf import WebScreenshotPDF, ScreenshotConfig


async def test_chrome_browser():
    """测试Chrome浏览器功能"""
    print("🧪 开始测试Chrome浏览器...")
    
    try:
        # 创建截图配置
        config = ScreenshotConfig(
            width=1366,
            height=768,
            full_page=True,
            wait_time=3,
            format="PNG"
        )
        
        # 测试URL
        test_url = "https://httpbin.org/html"
        
        print(f"📱 测试URL: {test_url}")
        print("🔧 使用Selenium Chrome引擎...")
        
        # 使用Selenium引擎
        async with WebScreenshotPDF(browser_engine="selenium") as processor:
            print("✅ 浏览器初始化成功!")
            
            # 进行截图
            screenshot_path = await processor.screenshot_url(test_url, config)
            
            print(f"📸 截图完成: {screenshot_path}")
            print(f"📁 文件大小: {screenshot_path.stat().st_size / 1024:.1f} KB")
            
            # 检查文件是否存在
            if screenshot_path.exists():
                print("✅ Chrome浏览器测试成功!")
                print(f"🎉 您现在可以使用Chrome浏览器进行网页截图了!")
                return True
            else:
                print("❌ 截图文件未生成")
                return False
                
    except Exception as e:
        print(f"❌ Chrome浏览器测试失败: {e}")
        print("\n🔧 可能的解决方案:")
        print("1. 确保已安装Google Chrome浏览器")
        print("2. 检查网络连接")
        print("3. 尝试重新运行测试")
        return False


async def main():
    """主函数"""
    print("🚀 Chrome浏览器功能测试")
    print("=" * 40)
    
    success = await test_chrome_browser()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 测试完成! Chrome浏览器工作正常!")
        print("💡 您现在可以在GUI中使用Chrome浏览器了!")
    else:
        print("❌ 测试失败，请检查错误信息并重试")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)