"""
通用浏览器控制模块
支持连接到现有Chrome实例或启动新实例，让用户在任意网站操作
"""

from __future__ import annotations
import os, re, json, requests, subprocess, pathlib, time
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 检查依赖
try:
    from PIL import Image
except ImportError:
    Image = None
    print("PIL未安装，请安装: pip install Pillow")

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None
    print("Playwright未安装，请安装: pip install playwright")

# 输出目录配置
IMG_ROOT = r"G:\E盘\工作项目文件\电子版涂书\书籍\原图"
PDF_DIR  = r"G:\E盘\工作项目文件\电子版涂书\书籍\PDF涂书"
OCR_DIR  = r"G:\E盘\工作项目文件\电子版涂书\论文\OCRmyPDF"

# GUI相关
try:
    import tkinter as tk
    from PIL import ImageTk
except Exception:
    tk = None

@dataclass
class CropBox:
    left:int; top:int; right:int; bottom:int

@dataclass
class BatchScreenshotState:
    """批量截图状态管理"""
    crop_box: Optional[CropBox] = None
    page_count: int = 0
    batch_count: int = 0
    current_batch_images: List[str] = None
    all_images: List[str] = None
    
    def __post_init__(self):
        if self.current_batch_images is None:
            self.current_batch_images = []
        if self.all_images is None:
            self.all_images = []

def select_crop_box(img_path:str)->CropBox:
    """交互式选择截图区域"""
    if not tk or not Image:
        print("❌ 缺少GUI依赖，使用全屏截图")
        img = Image.open(img_path)
        return CropBox(0, 0, img.width, img.height)
    
    root = tk.Tk()
    root.title("选择截图区域")
    
    img = Image.open(img_path)
    # 缩放图片以适应屏幕
    screen_w = root.winfo_screenwidth() - 100
    screen_h = root.winfo_screenheight() - 100
    
    scale = min(screen_w/img.width, screen_h/img.height, 1.0)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    
    display_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(display_img)
    
    canvas = tk.Canvas(root, width=new_w, height=new_h)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    
    # 选择区域的变量
    start_x = start_y = end_x = end_y = 0
    rect_id = None
    
    def on_click(event):
        nonlocal start_x, start_y, rect_id
        start_x, start_y = event.x, event.y
        if rect_id:
            canvas.delete(rect_id)
    
    def on_drag(event):
        nonlocal rect_id, end_x, end_y
        end_x, end_y = event.x, event.y
        if rect_id:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(start_x, start_y, end_x, end_y, outline='red', width=2)
    
    def on_confirm():
        root.quit()
    
    canvas.bind("<Button-1>", on_click)
    canvas.bind("<B1-Motion>", on_drag)
    
    confirm_btn = tk.Button(root, text="确认选择", command=on_confirm, bg='green', fg='white', font=('Arial', 12))
    confirm_btn.pack(pady=10)
    
    root.mainloop()
    root.destroy()
    
    # 转换回原图坐标
    real_x1 = int(min(start_x, end_x) / scale)
    real_y1 = int(min(start_y, end_y) / scale)
    real_x2 = int(max(start_x, end_x) / scale)
    real_y2 = int(max(start_y, end_y) / scale)
    
    return CropBox(real_x1, real_y1, real_x2, real_y2)

def merge_to_pdf(pngs:List[str], out_pdf:str):
    """合并PNG图片为PDF"""
    if not Image:
        print("❌ PIL未安装，无法合并PDF")
        return
    
    if not pngs:
        print("❌ 没有图片可合并")
        return
    
    try:
        images = []
        for png_path in pngs:
            if os.path.exists(png_path):
                img = Image.open(png_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
        
        if images:
            images[0].save(out_pdf, save_all=True, append_images=images[1:])
            print(f"✅ PDF已保存: {out_pdf}")
        else:
            print("❌ 没有有效图片可合并")
    except Exception as e:
        print(f"❌ 合并PDF失败: {e}")

def maybe_ocr(in_pdf:str, out_pdf:str):
    """可选的OCR处理"""
    try:
        import ocrmypdf
        print(f"🔍 正在进行OCR处理...")
        ocrmypdf.ocr(in_pdf, out_pdf, language='chi_sim+eng')
        print(f"✅ OCR处理完成: {out_pdf}")
    except ImportError:
        print("⚠️ OCRmyPDF未安装，跳过OCR处理")
    except Exception as e:
        print(f"⚠️ OCR处理失败: {e}")

def auto_page_turn(page):
    """自动翻页：模拟键盘右键"""
    try:
        print("📖 正在翻页...")
        page.keyboard.press("ArrowRight")
        time.sleep(1.5)  # 等待页面加载
        print("✅ 翻页完成")
        return True
    except Exception as e:
        print(f"❌ 翻页失败: {e}")
        return False

def create_batch_folder(batch_num: int) -> str:
    """创建批次文件夹"""
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"batch_{batch_num:02d}_{timestamp}"
    folder_path = os.path.join(IMG_ROOT, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def _take_screenshot_with_memory(page, state: BatchScreenshotState, page_num: int, folder_path: str) -> str:
    """使用记忆的截图区域进行截图"""
    try:
        # 生成文件名
        filename = f"page_{page_num:03d}.png"
        file_path = os.path.join(folder_path, filename)
        
        if state.crop_box is None:
            # 第一次截图，需要选择区域
            print("\n📸 第一次截图，需要选择截图区域...")
            
            # 全页面截图用于区域选择
            preview_path = os.path.join(folder_path, f"preview_temp.png")
            page.screenshot(path=preview_path, full_page=True)
            
            # 让用户选择区域
            print("🎯 请在预览图中框选书籍内容区域...")
            state.crop_box = select_crop_box(preview_path)
            
            # 删除临时预览文件
            try:
                os.remove(preview_path)
            except:
                pass
        
        # 使用记忆的区域进行截图
        if Image and state.crop_box:
            # 计算相对位置
            preview_path = os.path.join(folder_path, f"temp_full_{page_num}.png")
            page.screenshot(path=preview_path, full_page=True)
            
            preview_img = Image.open(preview_path)
            page_width, page_height = preview_img.size
            
            # 计算选择区域的相对位置（百分比）
            rel_left = state.crop_box.left / page_width
            rel_top = state.crop_box.top / page_height
            rel_right = state.crop_box.right / page_width
            rel_bottom = state.crop_box.bottom / page_height
            
            # 获取页面实际尺寸
            page_info = page.evaluate("""() => {
                return {
                    scrollWidth: document.documentElement.scrollWidth,
                    scrollHeight: document.documentElement.scrollHeight,
                    clientWidth: document.documentElement.clientWidth,
                    clientHeight: document.documentElement.clientHeight
                };
            }""")
            
            # 计算目标区域在实际页面中的位置
            actual_left = rel_left * page_info['scrollWidth']
            actual_top = rel_top * page_info['scrollHeight']
            actual_width = (rel_right - rel_left) * page_info['scrollWidth']
            actual_height = (rel_bottom - rel_top) * page_info['scrollHeight']
            
            # 计算合适的缩放比例
            viewport_width = page.viewport_size['width']
            viewport_height = page.viewport_size['height']
            
            zoom_x = (viewport_width * 0.7) / actual_width
            zoom_y = (viewport_height * 0.7) / actual_height
            target_zoom = min(zoom_x, zoom_y, 3.0)
            
            # 执行放大和定位
            center_x = actual_left + actual_width / 2
            center_y = actual_top + actual_height / 2
            
            page.evaluate(f"""() => {{
                window.scrollTo({center_x} - window.innerWidth/2, {center_y} - window.innerHeight/2);
                document.body.style.zoom = '{target_zoom}';
                return new Promise(resolve => setTimeout(resolve, 800));
            }}""")
            
            # 截取放大后的截图
            page.screenshot(path=file_path)
            
            # 恢复原始缩放
            page.evaluate("() => { document.body.style.zoom = '1'; }")
            
            # 删除临时文件
            try:
                os.remove(preview_path)
            except:
                pass
            
            print(f"✅ 第{page_num}页截图完成: {filename}")
            return file_path
        else:
            # 回退到普通截图
            page.screenshot(path=file_path)
            print(f"✅ 第{page_num}页截图完成: {filename}")
            return file_path
            
    except Exception as e:
        print(f"❌ 第{page_num}页截图失败: {e}")
        return None

def batch_screenshot_with_auto_turn(page, pages_per_batch: int = 20) -> None:
    """自动化批量截图功能"""
    print(f"\n🚀 启动自动化批量截图功能")
    print(f"📋 每批次截图页数: {pages_per_batch}")
    print(f"📁 图片保存目录: {IMG_ROOT}")
    print(f"📄 PDF保存目录: {PDF_DIR}")
    
    state = BatchScreenshotState()
    
    while True:
        state.batch_count += 1
        print(f"\n📦 开始第 {state.batch_count} 批次截图...")
        
        # 创建批次文件夹
        batch_folder = create_batch_folder(state.batch_count)
        print(f"📁 批次文件夹: {batch_folder}")
        
        # 重置当前批次图片列表
        state.current_batch_images = []
        
        # 截图当前批次
        for i in range(pages_per_batch):
            page_num = state.page_count + 1
            print(f"\n📸 正在截图第 {page_num} 页 (批次 {state.batch_count}, 页面 {i+1}/{pages_per_batch})")
            
            # 截图
            screenshot_path = _take_screenshot_with_memory(page, state, page_num, batch_folder)
            
            if screenshot_path:
                state.current_batch_images.append(screenshot_path)
                state.all_images.append(screenshot_path)
                state.page_count += 1
            else:
                print(f"⚠️ 第 {page_num} 页截图失败，跳过")
                continue
            
            # 如果不是最后一页，则自动翻页
            if i < pages_per_batch - 1:
                if not auto_page_turn(page):
                    print("❌ 翻页失败，批次截图中断")
                    break
                    
                # 等待页面稳定
                time.sleep(0.5)
        
        # 批次完成，生成PDF
        if state.current_batch_images:
            print(f"\n📄 正在生成第 {state.batch_count} 批次PDF...")
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"batch_{state.batch_count:02d}_{timestamp}.pdf"
            pdf_path = os.path.join(PDF_DIR, pdf_filename)
            
            # 确保PDF目录存在
            os.makedirs(PDF_DIR, exist_ok=True)
            
            # 合并为PDF
            merge_to_pdf(state.current_batch_images, pdf_path)
            print(f"✅ 第 {state.batch_count} 批次PDF已生成: {pdf_filename}")
        
        # 显示批次统计
        print(f"\n📊 第 {state.batch_count} 批次完成:")
        print(f"   📸 成功截图: {len(state.current_batch_images)} 页")
        print(f"   📁 图片文件夹: {os.path.basename(batch_folder)}")
        print(f"   📄 PDF文件: {pdf_filename if state.current_batch_images else '无'}")
        print(f"   📈 总计截图: {state.page_count} 页")
        
        # 批次控制菜单
        print(f"\n🎛️ 批次控制选项:")
        print("   1. 继续下一批次 (20页)")
        print("   2. 调整截图区域后继续")
        print("   3. 终止截图")
        print("   4. 合并所有批次为一个PDF")
        
        while True:
            try:
                choice = input("\n请选择操作 (1/2/3/4): ").strip()
                
                if choice == "1":
                    print("✅ 继续下一批次...")
                    break
                elif choice == "2":
                    print("🔄 重置截图区域，下一批次将重新选择区域...")
                    state.crop_box = None  # 重置区域
                    break
                elif choice == "3":
                    print("🛑 终止批量截图...")
                    
                    # 询问是否合并所有批次
                    if len(state.all_images) > 0:
                        merge_choice = input("\n❓ 是否将所有批次合并为一个完整PDF？(y/n): ").lower().strip()
                        if merge_choice in ['y', 'yes', '是', '要']:
                            print("📄 正在合并所有批次...")
                            final_timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                            final_pdf = os.path.join(PDF_DIR, f"complete_book_{final_timestamp}.pdf")
                            merge_to_pdf(state.all_images, final_pdf)
                            
                            # 询问是否OCR
                            ocr_choice = input("\n❓ 是否对完整PDF进行OCR处理？(y/n): ").lower().strip()
                            if ocr_choice in ['y', 'yes', '是', '要']:
                                ocr_pdf = os.path.join(OCR_DIR, f"complete_book_ocr_{final_timestamp}.pdf")
                                os.makedirs(OCR_DIR, exist_ok=True)
                                maybe_ocr(final_pdf, ocr_pdf)
                    
                    print(f"\n🎉 批量截图完成！")
                    print(f"📊 总计截图: {state.page_count} 页")
                    print(f"📦 总计批次: {state.batch_count} 个")
                    print(f"📁 图片保存在: {IMG_ROOT}")
                    print(f"📄 PDF保存在: {PDF_DIR}")
                    return
                elif choice == "4":
                    if len(state.all_images) > 0:
                        print("📄 正在合并所有已截图的批次...")
                        merge_timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                        merged_pdf = os.path.join(PDF_DIR, f"merged_batches_{merge_timestamp}.pdf")
                        merge_to_pdf(state.all_images, merged_pdf)
                        print(f"✅ 合并完成: {os.path.basename(merged_pdf)}")
                    else:
                        print("❌ 没有可合并的图片")
                else:
                    print("❌ 无效选择，请输入 1、2、3 或 4")
            except KeyboardInterrupt:
                print("\n🛑 用户中断操作")
                return

def _take_screenshot(page, base_name: str = None) -> List[str]:
    """改进的截图功能：先选择区域，再放大截图"""
    if not base_name:
        # 获取当前页面URL作为文件名
        current_url = page.url
        safe_filename = re.sub(r'[^\w\-_.]', '_', current_url.split('/')[-1] or 'screenshot')
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{safe_filename}_{timestamp}"
    
    # 创建输出目录
    os.makedirs(IMG_ROOT, exist_ok=True)
    
    print("\n📸 开始智能截图流程...")
    
    # 第一步：全页面截图用于区域选择
    print("1️⃣ 正在生成预览截图...")
    preview_path = os.path.join(IMG_ROOT, f"{base_name}_preview.png")
    page.screenshot(path=preview_path, full_page=True)
    print(f"✅ 预览截图已保存: {preview_path}")
    
    # 第二步：让用户选择书籍区域
    print("\n🎯 请在预览图中框选书籍内容区域...")
    print("💡 提示：选择书籍的主要内容区域，程序会自动放大该区域进行高质量截图")
    crop_box = select_crop_box(preview_path)
    
    # 第三步：计算选择区域在页面中的相对位置
    if Image:
        preview_img = Image.open(preview_path)
        page_width, page_height = preview_img.size
        
        # 计算选择区域的相对位置（百分比）
        rel_left = crop_box.left / page_width
        rel_top = crop_box.top / page_height
        rel_right = crop_box.right / page_width
        rel_bottom = crop_box.bottom / page_height
        
        print(f"\n🔍 检测到选择区域: {crop_box.left},{crop_box.top} -> {crop_box.right},{crop_box.bottom}")
        print(f"📐 相对位置: {rel_left:.2%},{rel_top:.2%} -> {rel_right:.2%},{rel_bottom:.2%}")
    
    # 第四步：使用JavaScript滚动到选择区域并放大
    print("\n🔎 正在定位并放大选择区域...")
    
    # 获取页面实际尺寸
    page_info = page.evaluate("""() => {
        return {
            scrollWidth: document.documentElement.scrollWidth,
            scrollHeight: document.documentElement.scrollHeight,
            clientWidth: document.documentElement.clientWidth,
            clientHeight: document.documentElement.clientHeight,
            currentZoom: window.devicePixelRatio
        };
    }""")
    
    # 计算目标区域在实际页面中的位置
    actual_left = rel_left * page_info['scrollWidth']
    actual_top = rel_top * page_info['scrollHeight']
    actual_width = (rel_right - rel_left) * page_info['scrollWidth']
    actual_height = (rel_bottom - rel_top) * page_info['scrollHeight']
    
    # 计算合适的缩放比例（确保文字清晰）
    viewport_width = page.viewport_size['width']
    viewport_height = page.viewport_size['height']
    
    # 目标：让选择区域占据视口的70%，这样文字会更清晰
    zoom_x = (viewport_width * 0.7) / actual_width
    zoom_y = (viewport_height * 0.7) / actual_height
    target_zoom = min(zoom_x, zoom_y, 3.0)  # 最大放大3倍
    
    print(f"📊 页面信息: {page_info['scrollWidth']}x{page_info['scrollHeight']}")
    print(f"🎯 目标区域: {actual_left:.0f},{actual_top:.0f} 尺寸: {actual_width:.0f}x{actual_height:.0f}")
    print(f"🔍 计算缩放比例: {target_zoom:.2f}x")
    
    # 第五步：执行放大和定位
    try:
        # 先滚动到目标区域中心
        center_x = actual_left + actual_width / 2
        center_y = actual_top + actual_height / 2
        
        page.evaluate(f"""() => {{
            // 滚动到目标区域中心
            window.scrollTo({center_x} - window.innerWidth/2, {center_y} - window.innerHeight/2);
            
            // 设置缩放
            document.body.style.zoom = '{target_zoom}';
            
            // 等待渲染
            return new Promise(resolve => setTimeout(resolve, 1000));
        }}""")
        
        print("✅ 页面已放大并定位到目标区域")
        
        # 第六步：截取放大后的高质量截图
        print("\n📷 正在截取高质量截图...")
        final_screenshot_path = os.path.join(IMG_ROOT, f"{base_name}_hq.png")
        
        # 截取当前视口（已经放大的区域）
        page.screenshot(path=final_screenshot_path)
        print(f"✅ 高质量截图已保存: {final_screenshot_path}")
        
        # 恢复原始缩放
        page.evaluate("() => { document.body.style.zoom = '1'; }")
        
        return [final_screenshot_path]
        
    except Exception as e:
        print(f"⚠️ 放大截图失败，使用传统裁剪方式: {e}")
        
        # 回退到传统裁剪方式
        cropped_images = []
        if Image:
            img = Image.open(preview_path)
            cropped = img.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))
            cropped_path = os.path.join(IMG_ROOT, f"{base_name}_cropped.png")
            cropped.save(cropped_path)
            cropped_images.append(cropped_path)
            print(f"✅ 裁剪后截图已保存: {cropped_path}")
        
        return cropped_images

def run_generic_browser(
    start_url: str = "blank",
    viewport: Tuple[int,int] = (1920,1080),
    connect_to_existing: bool = False,
    debug_port: int = 9222,
    user_data_dir: Optional[str] = None,
) -> None:
    """
    通用浏览器控制函数
    
    Args:
        start_url: 起始URL
        viewport: 浏览器视口大小
        connect_to_existing: 是否连接到现有Chrome实例
        debug_port: Chrome调试端口
        user_data_dir: Chrome用户数据目录
    """
    
    if not sync_playwright:
        raise RuntimeError("Playwright未安装，请运行: pip install playwright")
    
    print(f"🚀 启动通用浏览器控制")
    print(f"📋 起始URL: {start_url}")
    print(f"📐 视口大小: {viewport[0]}x{viewport[1]}")
    
    with sync_playwright() as p:
        browser = None
        ctx = None
        
        try:
            # 连接到现有Chrome实例的逻辑
            if connect_to_existing:
                try:
                    print(f"尝试连接到现有Chrome实例 (端口: {debug_port})...")
                    print("📋 请确保你的Chrome浏览器已经启动并开启了远程调试")
                    print(f"   如果Chrome未开启调试，请用以下命令重启Chrome：")
                    print(f"   chrome.exe --remote-debugging-port={debug_port}")
                    
                    browser = p.chromium.connect_over_cdp(f"http://localhost:{debug_port}")
                    print("✅ 成功连接到现有Chrome实例")
                    
                    # 获取现有的上下文
                    contexts = browser.contexts
                    if contexts:
                        ctx = contexts[0]  # 使用第一个上下文
                        print("✅ 使用现有浏览器上下文")
                    else:
                        ctx = browser.new_context(viewport={"width":viewport[0], "height":viewport[1]})
                        print("✅ 创建新的浏览器上下文")
                        
                except Exception as e_connect:
                    print(f"❌ 连接现有Chrome失败: {e_connect}")
                    print("将回退到启动新的Chrome实例...")
                    connect_to_existing = False  # 回退到启动新实例
            
            if not connect_to_existing:
                udir = None
                if user_data_dir:
                    udir = os.path.expandvars(user_data_dir)
                    print(f"使用持久化 Chrome 用户数据目录: {udir}")
                    try:
                        ctx = p.chromium.launch_persistent_context(
                            udir,
                            headless=False,
                            channel="chrome",
                            args=[
                                '--no-sandbox', 
                                '--disable-dev-shm-usage',
                                '--disable-web-security',
                                '--disable-features=VizDisplayCompositor',
                                '--disable-extensions',
                                '--no-first-run',
                                '--disable-default-apps',
                                '--disable-popup-blocking',
                            ],
                        )
                        print("✅ Chrome 持久化上下文启动成功")
                    except Exception as e_persist:
                        print(f"❌ 持久化上下文启动失败，将使用临时上下文: {e_persist}")
                
                if ctx is None:
                    # 尝试使用 Chrome，如果失败则回退到 Chromium
                    try:
                        print("尝试启动 Chrome 浏览器...")
                        browser = p.chromium.launch(
                            headless=False,
                            channel="chrome",
                            args=[
                                '--no-sandbox', 
                                '--disable-dev-shm-usage',
                                '--disable-web-security',
                                '--disable-features=VizDisplayCompositor',
                                '--disable-extensions',
                                '--no-first-run',
                                '--disable-default-apps',
                                '--disable-popup-blocking',
                            ],
                        )
                        print("✅ Chrome 浏览器启动成功")
                    except Exception as e:
                        print(f"❌ Chrome 启动失败: {e}")
                        try:
                            print("尝试启动 Chromium 浏览器...")
                            browser = p.chromium.launch(
                                headless=False,
                                args=[
                                    '--no-sandbox', 
                                    '--disable-dev-shm-usage',
                                    '--disable-web-security',
                                    '--disable-features=VizDisplayCompositor',
                                ],
                            )
                            print("✅ Chromium 浏览器启动成功")
                        except Exception as e2:
                            print(f"❌ Chromium 也启动失败: {e2}")
                            raise RuntimeError(f"无法启动任何浏览器: Chrome={e}, Chromium={e2}")
                    ctx = browser.new_context(viewport={"width":viewport[0], "height":viewport[1]})

            # 如果连接到现有Chrome实例，尝试使用当前活动页面
            if connect_to_existing and ctx.pages:
                page = ctx.pages[-1]  # 使用最后一个（通常是当前活动的）页面
                print("✅ 使用现有浏览器页面")
                # 确保视口尺寸
                try:
                    page.set_viewport_size({"width":viewport[0], "height":viewport[1]})
                except Exception:
                    pass
            else:
                page = ctx.new_page()
                # 对持久化上下文确保视口尺寸
                try:
                    page.set_viewport_size({"width":viewport[0], "height":viewport[1]})
                except Exception:
                    pass
                print("✅ 浏览器页面创建成功")

            # 不自动跳转，让用户在地址栏自由输入
            if start_url and start_url.lower() != "blank":
                print(f"打开起始页面: {start_url}")
                page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
                print("✅ 页面加载完成")
            else:
                print("✅ 浏览器已启动，保持空白页面")
            
            # 集成的用户操作和截图阶段
            print(f"\n🌟 浏览器已准备就绪！")
            print("📋 您现在可以：")
            print("   1. 在地址栏输入任意网站URL")
            print("   2. 登录账户、借阅书籍或进行其他操作")
            print("   3. 调整页面到您想要截图的状态")
            print(f"\n🔥 重要提示：请不要关闭浏览器窗口或标签页！")
            
            # 集成的操作循环
            while True:
                print(f"\n📋 请选择操作：")
                print("   1. 截图当前页面")
                print("   2. 🚀 自动化批量截图 (20页/批次)")
                print("   3. 继续在浏览器中操作")
                print("   4. 完成并退出")

                try:
                    choice = input("\n请输入选择 (1/2/3/4): ").strip()

                    if choice == "1":
                        # 检查页面是否还存在
                        if page.is_closed():
                            print("❌ 页面已关闭，无法截图")
                            continue

                        # 截图操作
                        print("\n📸 开始截图...")
                        cropped_images = _take_screenshot(page)

                        # 询问是否需要更多截图
                        while True:
                            more = input("\n❓ 是否需要截取更多页面？(y/n): ").lower().strip()
                            if more in ['n', 'no', '否', '不']:
                                break
                            elif more in ['y', 'yes', '是', '要']:
                                print("📋 请在浏览器中导航到下一页，然后返回这里")
                                input("✅ 准备好后按回车继续截图... ")

                                # 再次截图
                                page_num = input("📝 请输入页面编号（用于文件命名）: ").strip() or "next"
                                more_images = _take_screenshot(page, f"page_{page_num}")
                                cropped_images.extend(more_images)
                            else:
                                print("请输入 y 或 n")

                        # 处理收集到的图片
                        if cropped_images:
                            # 创建PDF目录
                            os.makedirs(PDF_DIR, exist_ok=True)

                            # 生成PDF文件名
                            current_url = page.url
                            safe_filename = re.sub(r'[^\w\-_.]', '_', current_url.split('/')[-1] or 'screenshot')
                            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                            base_name = f"{safe_filename}_{timestamp}"

                            # 合并为PDF
                            pdf_path = os.path.join(PDF_DIR, f"{base_name}.pdf")
                            merge_to_pdf(cropped_images, pdf_path)

                            # 询问是否需要OCR
                            ocr_choice = input("\n❓ 是否需要进行OCR文字识别？(y/n): ").lower().strip()
                            if ocr_choice in ['y', 'yes', '是', '要']:
                                ocr_pdf = os.path.join(OCR_DIR, f"{base_name}_ocr.pdf")
                                os.makedirs(OCR_DIR, exist_ok=True)
                                maybe_ocr(pdf_path, ocr_pdf)

                            print(f"\n🎉 截图处理完成！")
                            print(f"📁 图片保存在: {IMG_ROOT}")
                            print(f"📄 PDF保存在: {PDF_DIR}")

                    elif choice == "2":
                        # 自动化批量截图
                        if page.is_closed():
                            print("❌ 页面已关闭，无法截图")
                            continue
                        
                        print("\n🚀 启动自动化批量截图模式")
                        print("💡 提示：")
                        print("   - 程序将自动翻页并截图")
                        print("   - 每20页为一个批次")
                        print("   - 第一次会让您选择截图区域")
                        print("   - 后续页面将使用相同区域")
                        print("   - 使用键盘右键进行翻页")
                        
                        confirm = input("\n❓ 确认开始自动化批量截图？(y/n): ").lower().strip()
                        if confirm in ['y', 'yes', '是', '要']:
                            batch_screenshot_with_auto_turn(page, 20)
                        else:
                            print("❌ 已取消批量截图")

                    elif choice == "3":
                        print("✅ 请继续在浏览器中操作，完成后回到这里...")
                        input("按回车继续...")
                    elif choice == "4":
                        print("✅ 操作完成，正在退出...")
                        break
                    else:
                        print("❌ 无效选择，请输入 1、2、3 或 4")

                except KeyboardInterrupt:
                    print("\n用户中断操作")
                    break
            
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            raise
        finally:
            # 清理资源
            try:
                if not connect_to_existing and ctx:
                    ctx.close()
                if not connect_to_existing and browser:
                    browser.close()
            except Exception:
                pass