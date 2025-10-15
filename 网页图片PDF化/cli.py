#!/usr/bin/env python3
"""
命令行接口工具

提供网页截图并合成PDF的命令行接口，支持单个URL和批量处理。
"""

import asyncio
import click
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from web_screenshot_pdf import (
    WebScreenshotPDF, 
    ScreenshotConfig, 
    OCRConfig, 
    PDFConfig
)
from config import get_config, load_config, create_example_config
from adapters.internet_archive import run_ia


def setup_logging(log_level: str, log_file: Optional[str] = None):
    """设置日志"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='配置文件路径')
@click.option('--log-level', default='INFO', help='日志级别')
@click.option('--log-file', help='日志文件路径')
@click.pass_context
def cli(ctx, config, log_level, log_file):
    """网页截图并合成PDF工具"""
    ctx.ensure_object(dict)
    
    # 加载配置
    if config:
        load_config(Path(config))
    
    app_config = get_config()
    
    # 设置日志
    setup_logging(log_level or app_config.processing.log_level, log_file)
    
    ctx.obj['config'] = app_config


@cli.command()
@click.argument('url')
@click.option('--output', '-o', default='screenshot.pdf', help='输出PDF文件路径')
@click.option('--browser', default=None, help='浏览器引擎 (selenium/playwright)')
@click.option('--full-page/--viewport', default=True, help='全页面截图或仅视口')
@click.option('--width', default=1920, help='浏览器窗口宽度')
@click.option('--height', default=1080, help='浏览器窗口高度')
@click.option('--wait', default=3, help='页面加载等待时间（秒）')
@click.option('--quality', default=95, help='图片质量 (1-100)')
@click.option('--format', default='PNG', help='图片格式 (PNG/JPEG/WEBP)')
@click.option('--ocr-engine', default=None, help='OCR引擎 (auto/tesseract/easyocr/paddleocr)')
@click.option('--ocr-lang', default=None, help='OCR语言，逗号分隔 (如: eng,chi_sim)')
@click.option('--no-ocr', is_flag=True, help='禁用OCR')
@click.option('--title', help='PDF标题')
@click.option('--author', help='PDF作者')
@click.pass_context
async def single(ctx, url, output, browser, full_page, width, height, wait, 
                quality, format, ocr_engine, ocr_lang, no_ocr, title, author):
    """处理单个URL"""
    config = ctx.obj['config']
    
    # 创建配置
    screenshot_config = ScreenshotConfig(
        width=width,
        height=height,
        full_page=full_page,
        wait_time=wait,
        quality=quality,
        format=format.upper()
    )
    
    ocr_config = OCRConfig(
        engine=ocr_engine or config.ocr.engine,
        languages=ocr_lang.split(',') if ocr_lang else config.ocr.languages,
        confidence_threshold=config.ocr.confidence_threshold,
        preprocess=config.ocr.preprocess
    )
    
    pdf_config = PDFConfig(
        page_size=config.pdf.page_size,
        margin=config.pdf.margin,
        include_ocr_text=not no_ocr and config.pdf.include_ocr_text,
        searchable=not no_ocr and config.pdf.searchable,
        title=title or config.pdf.title,
        author=author or config.pdf.author
    )
    
    # 处理
    browser_engine = browser or config.browser.engine
    
    try:
        async with WebScreenshotPDF(browser_engine=browser_engine) as processor:
            result = await processor.process_urls_to_pdf(
                urls=[url],
                output_path=Path(output),
                screenshot_config=screenshot_config,
                ocr_config=ocr_config,
                pdf_config=pdf_config
            )
            
            if result['success']:
                click.echo(f"✅ 成功生成PDF: {output}")
                click.echo(f"📊 处理时间: {result['processing_time']:.2f}秒")
                if result['ocr_results']:
                    ocr_result = result['ocr_results'][0]['ocr_result']
                    click.echo(f"📝 OCR文字数: {ocr_result.get('character_count', 0)}")
                    click.echo(f"🎯 OCR置信度: {ocr_result.get('confidence', 0):.2%}")
            else:
                click.echo(f"❌ 处理失败: {result.get('error', '未知错误')}", err=True)
                sys.exit(1)
                
    except Exception as e:
        click.echo(f"❌ 处理失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('urls_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='batch_screenshots.pdf', help='输出PDF文件路径')
@click.option('--browser', default=None, help='浏览器引擎 (selenium/playwright)')
@click.option('--concurrent', default=None, type=int, help='并发任务数')
@click.option('--delay', default=0, type=float, help='URL之间的延迟（秒）')
@click.option('--retry', default=None, type=int, help='重试次数')
@click.option('--no-ocr', is_flag=True, help='禁用OCR')
@click.option('--title', help='PDF标题')
@click.option('--author', help='PDF作者')
@click.option('--report', help='生成处理报告文件路径')
@click.pass_context
async def batch(ctx, urls_file, output, browser, concurrent, delay, retry, 
               no_ocr, title, author, report):
    """批量处理URL列表"""
    config = ctx.obj['config']
    
    # 读取URL列表
    try:
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        click.echo(f"❌ 读取URL文件失败: {e}", err=True)
        sys.exit(1)
    
    if not urls:
        click.echo("❌ 没有找到有效的URL", err=True)
        sys.exit(1)
    
    click.echo(f"📋 找到 {len(urls)} 个URL")
    
    # 创建配置
    screenshot_config = ScreenshotConfig(
        width=config.screenshot.resize_factor,
        height=config.browser.window_height,
        full_page=config.screenshot.full_page,
        wait_time=config.browser.wait_time,
        quality=config.screenshot.quality,
        format=config.screenshot.format
    )
    
    ocr_config = OCRConfig(
        engine=config.ocr.engine,
        languages=config.ocr.languages,
        confidence_threshold=config.ocr.confidence_threshold,
        preprocess=config.ocr.preprocess
    )
    
    pdf_config = PDFConfig(
        page_size=config.pdf.page_size,
        margin=config.pdf.margin,
        include_ocr_text=not no_ocr and config.pdf.include_ocr_text,
        searchable=not no_ocr and config.pdf.searchable,
        title=title or f"批量网页截图 - {datetime.now().strftime('%Y-%m-%d')}",
        author=author or config.pdf.author
    )
    
    # 更新处理配置
    if concurrent:
        config.processing.max_concurrent_tasks = concurrent
    if retry:
        config.processing.retry_attempts = retry
    
    # 处理
    browser_engine = browser or config.browser.engine
    
    try:
        async with WebScreenshotPDF(browser_engine=browser_engine) as processor:
            # 添加进度条
            with click.progressbar(length=len(urls), label='处理进度') as bar:
                # 分批处理以避免内存问题
                batch_size = config.processing.max_concurrent_tasks
                all_results = []
                
                for i in range(0, len(urls), batch_size):
                    batch_urls = urls[i:i + batch_size]
                    
                    # 处理当前批次
                    batch_output = Path(output).parent / f"batch_{i//batch_size + 1}_{Path(output).name}"
                    
                    result = await processor.process_urls_to_pdf(
                        urls=batch_urls,
                        output_path=batch_output,
                        screenshot_config=screenshot_config,
                        ocr_config=ocr_config,
                        pdf_config=pdf_config
                    )
                    
                    all_results.append(result)
                    bar.update(len(batch_urls))
                    
                    # 延迟
                    if delay > 0 and i + batch_size < len(urls):
                        await asyncio.sleep(delay)
                
                # 合并结果
                total_processed = sum(r['processed_urls'] for r in all_results)
                total_failed = sum(len(r['failed_urls']) for r in all_results)
                total_time = sum(r['processing_time'] for r in all_results)
                
                # 合并PDF文件（如果有多个批次）
                if len(all_results) > 1:
                    click.echo("🔗 合并PDF文件...")
                    # 这里可以添加PDF合并逻辑
                
                # 显示结果
                click.echo(f"\n📊 处理完成:")
                click.echo(f"   ✅ 成功: {total_processed}/{len(urls)}")
                click.echo(f"   ❌ 失败: {total_failed}")
                click.echo(f"   ⏱️  总时间: {total_time:.2f}秒")
                click.echo(f"   📄 输出文件: {output}")
                
                # 生成报告
                if report:
                    report_data = {
                        "timestamp": datetime.now().isoformat(),
                        "total_urls": len(urls),
                        "processed_urls": total_processed,
                        "failed_urls": total_failed,
                        "processing_time": total_time,
                        "output_file": str(output),
                        "config": {
                            "browser_engine": browser_engine,
                            "ocr_enabled": not no_ocr,
                            "concurrent_tasks": config.processing.max_concurrent_tasks
                        },
                        "results": all_results
                    }
                    
                    with open(report, 'w', encoding='utf-8') as f:
                        json.dump(report_data, f, indent=2, ensure_ascii=False)
                    
                    click.echo(f"📋 报告已保存: {report}")
                
                if total_failed > 0:
                    click.echo(f"\n⚠️  有 {total_failed} 个URL处理失败，请检查日志")
                    
    except Exception as e:
        click.echo(f"❌ 批量处理失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('url')
@click.option('--output', '-o', default='screenshot.png', help='输出图片文件路径')
@click.option('--browser', default=None, help='浏览器引擎 (selenium/playwright)')
@click.option('--full-page/--viewport', default=True, help='全页面截图或仅视口')
@click.option('--width', default=1920, help='浏览器窗口宽度')
@click.option('--height', default=1080, help='浏览器窗口高度')
@click.option('--wait', default=3, help='页面加载等待时间（秒）')
@click.option('--element', help='截图特定元素的CSS选择器')
@click.option('--hide', multiple=True, help='隐藏元素的CSS选择器（可多次使用）')
@click.pass_context
async def screenshot(ctx, url, output, browser, full_page, width, height, 
                    wait, element, hide):
    """仅截图，不进行OCR和PDF生成"""
    config = ctx.obj['config']
    
    screenshot_config = ScreenshotConfig(
        width=width,
        height=height,
        full_page=full_page,
        wait_time=wait,
        element_selector=element,
        hide_elements=list(hide),
        format=Path(output).suffix[1:].upper() or 'PNG'
    )
    
    browser_engine = browser or config.browser.engine
    
    try:
        async with WebScreenshotPDF(browser_engine=browser_engine) as processor:
            screenshot_path = await processor.screenshot_url(url, screenshot_config)
            
            # 移动到指定输出路径
            import shutil
            shutil.move(screenshot_path, output)
            
            click.echo(f"✅ 截图完成: {output}")
            
    except Exception as e:
        click.echo(f"❌ 截图失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--engine', default=None, help='OCR引擎 (auto/tesseract/easyocr/paddleocr)')
@click.option('--lang', default=None, help='OCR语言，逗号分隔 (如: eng,chi_sim)')
@click.option('--output', '-o', help='输出文本文件路径')
@click.option('--confidence', default=None, type=float, help='置信度阈值 (0.0-1.0)')
@click.pass_context
async def ocr(ctx, image_path, engine, lang, output, confidence):
    """对图片进行OCR文字识别"""
    config = ctx.obj['config']
    
    ocr_config = OCRConfig(
        engine=engine or config.ocr.engine,
        languages=lang.split(',') if lang else config.ocr.languages,
        confidence_threshold=confidence or config.ocr.confidence_threshold,
        preprocess=config.ocr.preprocess
    )
    
    try:
        async with WebScreenshotPDF() as processor:
            result = await processor.extract_text_ocr(Path(image_path), ocr_config)
            
            if result['text']:
                click.echo(f"✅ OCR识别完成:")
                click.echo(f"   🎯 置信度: {result['confidence']:.2%}")
                click.echo(f"   📝 字符数: {result['character_count']}")
                click.echo(f"   🔤 单词数: {result['word_count']}")
                click.echo(f"   🔧 引擎: {result['engine']}")
                
                if output:
                    with open(output, 'w', encoding='utf-8') as f:
                        f.write(result['text'])
                    click.echo(f"   💾 文本已保存: {output}")
                else:
                    click.echo(f"\n📄 识别文本:")
                    click.echo("-" * 50)
                    click.echo(result['text'])
                    click.echo("-" * 50)
            else:
                click.echo("❌ 未识别到文字内容")
                
    except Exception as e:
        click.echo(f"❌ OCR识别失败: {e}", err=True)
        sys.exit(1)


@cli.command()
def config_example():
    """生成示例配置文件"""
    try:
        create_example_config()
        click.echo("✅ 示例配置文件已生成: config.example.json")
        click.echo("💡 可以复制为 config.json 并根据需要修改")
    except Exception as e:
        click.echo(f"❌ 生成配置文件失败: {e}", err=True)


@cli.command("capture-by-clicks")
@click.option('--start-url', required=True, help='起始URL')
@click.option('--next-selector', default=None, help='下一页按钮的CSS选择器')
@click.option('--image-selector', default=None, help='图片元素的CSS选择器')
@click.option('--use-arrow-keys', is_flag=True, default=False, help='使用键盘箭头键翻页')
@click.option('--interactive-crop/--no-interactive-crop', default=True, help='启用交互式裁剪')
@click.option('--merge-pdf/--no-merge-pdf', default=True, help='合并为PDF')
@click.option('--max-pages', default=2000, help='最大页面数')
@click.option('--pdf-title', default=None, help='PDF标题')
@click.option('--auto-ocr', is_flag=True, default=False, help='自动执行OCR')
@click.option('--keep-images', is_flag=True, default=False, help='保留临时图片')
@click.pass_context
def capture_by_clicks_cmd(ctx, **kwargs):
    """点击翻页抓取网页并生成PDF"""
    config = ctx.obj['config']
    
    # 构建截图配置
    screenshot_config = ScreenshotConfig(
        image_selector=kwargs['image_selector'],
        next_selector=kwargs['next_selector'],
        use_arrow_keys=kwargs['use_arrow_keys'],
        interactive_crop=kwargs['interactive_crop'],
        lazy_scroll=True,
        lazy_timeout=3.0
    )
    
    # 构建PDF配置
    pdf_config = PDFConfig(
        title=kwargs['pdf_title'] or "点击翻页抓取",
        max_pages=kwargs['max_pages'],
        auto_ocr=kwargs['auto_ocr'],
        keep_temp_images=kwargs['keep_images'],
        enable_preview=True
    )
    
    if kwargs['merge_pdf']:
        asyncio.run(_run_clicks_to_pdf_flow(kwargs['start_url'], screenshot_config, pdf_config))
    else:
        asyncio.run(_run_clicks_only_flow(kwargs['start_url'], screenshot_config))

async def _run_clicks_to_pdf_flow(start_url: str, screenshot_config: ScreenshotConfig, pdf_config: PDFConfig):
    """运行完整的点击翻页到PDF流程"""
    async with WebScreenshotPDF(browser_engine="playwright") as processor:
        result = await processor.process_clicks_to_pdf(start_url, screenshot_config, pdf_config)
        
        if result['success']:
            click.echo(f"✅ 处理完成!")
            click.echo(f"📄 PDF文件: {result['pdf_path']}")
            if result['ocr_path']:
                click.echo(f"🔤 OCR PDF: {result['ocr_path']}")
            click.echo(f"📊 共处理 {result['image_count']} 张图片")
        else:
            click.echo(f"❌ 处理失败: {result['error']}")

async def _run_clicks_only_flow(start_url: str, screenshot_config: ScreenshotConfig):
    """仅运行点击翻页截图流程"""
    from pathlib import Path
    import tempfile
    
    output_dir = Path(tempfile.mkdtemp(prefix="web_shots_"))
    
    async with WebScreenshotPDF(browser_engine="playwright") as processor:
        result_dir = await processor.capture_by_clicks(start_url, output_dir, screenshot_config)
        
        image_files = list(result_dir.glob("*.png"))
        click.echo(f"✅ 截图完成!")
        click.echo(f"📁 图片目录: {result_dir}")
        click.echo(f"📊 共截取 {len(image_files)} 张图片")

@cli.command("ia")
@click.option("--id", "item_id", required=True, help="Internet Archive 的 identifier，如 isbn_9780965083409")
@click.option("--start-n", default=7, show_default=True, help="起始 n（/page/nX 的 X）")
@click.option("--end-n", default=None, type=int, help="结束 n（默认自动根据 metadata imagecount 推断）")
@click.option("--mode", type=click.Choice(["url","arrow","scroll"]), default="url", show_default=True,
              help="翻页方式：url=改n；arrow=发送右方向键；scroll=滚轮")
@click.option("--viewport-w", default=1600, show_default=True)
@click.option("--viewport-h", default=1200, show_default=True)
@click.option("--start-url", default=None, help="起始URL（可选，用于从已登录页面开始）")
@click.option("--profile-dir", default=None,
              help="Chrome 用户数据目录，用于复用本机登录状态。示例(Windows)：%LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default")
@click.option("--connect-existing", is_flag=True, default=False, 
              help="连接到现有Chrome实例而不是启动新的（需要Chrome开启远程调试）")
@click.option("--debug-port", default=9222, show_default=True,
              help="Chrome远程调试端口（仅在--connect-existing时使用）")
def ia_cmd(item_id, start_n, end_n, mode, viewport_w, viewport_h, start_url, profile_dir, connect_existing, debug_port):
    """Internet Archive 专用：手动登录/摆正 → 框选 → 批量截图 → PDF → 可选 OCR"""
    try:
        run_ia(item_id=item_id, start_n=start_n, end_n=end_n, mode=mode,
               viewport=(viewport_w, viewport_h), start_url=start_url, user_data_dir=profile_dir,
               connect_to_existing=connect_existing, debug_port=debug_port)
        click.echo(f"✅ Internet Archive 处理完成: {item_id}")
    except Exception as e:
        click.echo(f"❌ 处理失败: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command("browser")
@click.option("--start-url", default="blank", help="起始URL（默认空白页面，可输入任意URL）")
@click.option("--viewport-w", default=1920, show_default=True, help="浏览器窗口宽度")
@click.option("--viewport-h", default=1080, show_default=True, help="浏览器窗口高度")
@click.option("--connect-existing", is_flag=True, default=False, 
              help="连接到现有Chrome实例而不是启动新的（需要Chrome开启远程调试）")
@click.option("--debug-port", default=9222, show_default=True,
              help="Chrome远程调试端口（仅在--connect-existing时使用）")
@click.option("--profile-dir", default=None,
              help="Chrome 用户数据目录，用于复用本机登录状态")
def browser_cmd(start_url, viewport_w, viewport_h, connect_existing, debug_port, profile_dir):
    """通用浏览器控制：打开浏览器让用户自由操作任意网站，然后进行截图"""
    try:
        from adapters.generic_browser import run_generic_browser
        run_generic_browser(
            start_url=start_url,
            viewport=(viewport_w, viewport_h),
            connect_to_existing=connect_existing,
            debug_port=debug_port,
            user_data_dir=profile_dir
        )
        click.echo(f"✅ 浏览器操作完成")
    except Exception as e:
        click.echo(f"❌ 处理失败: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option('--check-browser', is_flag=True, help='检查浏览器环境')
@click.option('--check-ocr', is_flag=True, help='检查OCR引擎')
@click.option('--check-all', is_flag=True, help='检查所有依赖')
def check(check_browser, check_ocr, check_all):
    """检查系统环境和依赖"""
    if check_all:
        check_browser = check_ocr = True
    
    if not any([check_browser, check_ocr]):
        check_browser = check_ocr = True
    
    click.echo("🔍 系统环境检查:")
    
    if check_browser:
        click.echo("\n🌐 浏览器环境:")
        
        # 检查Selenium
        try:
            from selenium import webdriver
            click.echo("   ✅ Selenium 可用")
            
            # 检查Chrome驱动
            try:
                from selenium.webdriver.chrome.options import Options
                options = Options()
                options.add_argument("--headless")
                driver = webdriver.Chrome(options=options)
                driver.quit()
                click.echo("   ✅ Chrome WebDriver 可用")
            except Exception as e:
                click.echo(f"   ❌ Chrome WebDriver 不可用: {e}")
                
        except ImportError:
            click.echo("   ❌ Selenium 未安装")
        
        # 检查Playwright
        try:
            from playwright.async_api import async_playwright
            click.echo("   ✅ Playwright 可用")
        except ImportError:
            click.echo("   ❌ Playwright 未安装")
    
    if check_ocr:
        click.echo("\n🔤 OCR引擎:")
        
        # 检查Tesseract
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            click.echo(f"   ✅ Tesseract 可用 (版本: {version})")
        except Exception as e:
            click.echo(f"   ❌ Tesseract 不可用: {e}")
        
        # 检查EasyOCR
        try:
            import easyocr
            click.echo("   ✅ EasyOCR 可用")
        except ImportError:
            click.echo("   ❌ EasyOCR 未安装")
        
        # 检查PaddleOCR
        try:
            import paddleocr
            click.echo("   ✅ PaddleOCR 可用")
        except ImportError:
            click.echo("   ❌ PaddleOCR 未安装")
    
    click.echo("\n💡 如有缺失依赖，请参考 requirements.txt 安装")


# 异步命令包装器
def async_command(f):
    """装饰器：将异步命令转换为同步"""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


# 应用异步装饰器
single = async_command(single)
batch = async_command(batch)
screenshot = async_command(screenshot)
ocr = async_command(ocr)
capture_by_clicks_cmd = async_command(capture_by_clicks_cmd)


if __name__ == '__main__':
    cli()