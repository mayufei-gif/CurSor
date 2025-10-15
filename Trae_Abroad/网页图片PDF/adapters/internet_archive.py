# -*- coding: utf-8 -*-
"""
Internet Archive 适配器：
- 处理 https://archive.org/details/<ITEM_ID>/page/nX/mode/1up
- 首屏人工登录/借阅 + 切到单页 + 缩放 + 摆正
- 首张视口截图 → Tk 鼠标框选页面区域 → 批量套用裁剪框
- 翻页方式：url（改 n）/ arrow（右键）/ scroll（三选一）
- 每页：等待稳定 → 去遮挡 → 视口截图 → 裁剪 → 以 n 零填充命名
- 合并 PDF → 可选 OCR 到指定目录
"""

from __future__ import annotations
import os, re, json, requests, subprocess, pathlib
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 可选依赖导入
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

# 你的目标落地目录（可改为从 config 读取，这里先直连你的要求）
IMG_ROOT = r"G:\E盘\工作项目文件\电子版涂书\书籍\原图"
PDF_DIR  = r"G:\E盘\工作项目文件\电子版涂书\书籍\PDF涂书"
OCR_DIR  = r"G:\E盘\工作项目文件\电子版涂书\论文\OCRmyPDF"

# --- Tk 框选 ---
try:
    import tkinter as tk
    from PIL import ImageTk
except Exception:
    tk = None

@dataclass
class CropBox:
    left:int; top:int; right:int; bottom:int

# ---------- 小工具 ----------
def _mkdir(p:str): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def parse_n(url:str)->Optional[int]:
    m = re.search(r"/page/n(\d+)(?:/|$)", url)
    return int(m.group(1)) if m else None

def replace_n(url:str, new_n:int)->str:
    return re.sub(r"/page/n\d+", f"/page/n{new_n}", url)

def meta_imagecount(item_id:str)->Optional[int]:
    try:
        r = requests.get(f"https://archive.org/metadata/{item_id}", timeout=15)
        r.raise_for_status()
        md = r.json().get("metadata", {})
        return int(md["imagecount"]) if "imagecount" in md else None
    except Exception:
        return None

def hide_overlays(page):
    css = """
    [role="dialog"], .modal, .popup, .overlay, .ajs-message { display:none !important; }
    header, footer, #navwrap, #nav-bar, #ia-bar, .navbar, .brand { display:none !important; }
    """
    try: page.add_style_tag(content=css)
    except: pass
    page.keyboard.press("Escape"); page.keyboard.press("Escape")

def wait_stable(page):
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)

def select_crop_box(img_path:str)->CropBox:
    if tk is None:
        raise RuntimeError("需要 tkinter 才能交互式框选；请确保 Python 自带 Tk。")
    im = Image.open(img_path)
    maxw = 1280
    scale = min(1.0, maxw / im.width)
    show = im if scale==1 else im.resize((int(im.width*scale), int(im.height*scale)), Image.LANCZOS)

    root = tk.Tk(); root.title("拖拽框选【页面区域】，回车确认，Esc取消")
    tkimg = ImageTk.PhotoImage(show)
    cv = tk.Canvas(root, width=show.width, height=show.height); cv.pack()
    cv.create_image(0,0,anchor="nw",image=tkimg)
    sel=None; p0=[0,0]; p1=[0,0]
    def down(e):
        nonlocal sel,p0,p1; p0=[e.x,e.y]; p1=[e.x,e.y]
        if sel: cv.delete(sel)
        sel=cv.create_rectangle(p0[0],p0[1],p1[0],p1[1],outline="red",width=2)
    def drag(e):
        nonlocal p1; p1=[e.x,e.y]; cv.coords(sel,p0[0],p0[1],p1[0],p1[1])
    def ok(_): root.quit()
    def cancel(_): p0[:]=[0,0]; p1[:]=[0,0]; root.quit()
    cv.bind("<ButtonPress-1>", down); cv.bind("<B1-Motion>", drag)
    root.bind("<Return>", ok); root.bind("<Escape>", cancel)
    root.mainloop(); root.destroy()
    k = 1/scale
    lx,ty = int(min(p0[0],p1[0])*k), int(min(p0[1],p1[1])*k)
    rx,by = int(max(p0[0],p1[0])*k), int(max(p0[1],p1[1])*k)
    return CropBox(lx,ty,rx,by)

def merge_to_pdf(pngs:List[str], out_pdf:str):
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    if not imgs: return
    _mkdir(os.path.dirname(out_pdf))
    imgs[0].save(out_pdf, save_all=True, append_images=imgs[1:])

def maybe_ocr(in_pdf:str, out_pdf:str):
    ans = input("是否运行 OCRmyPDF？(y/N)：").strip().lower()
    if ans != "y": return
    _mkdir(os.path.dirname(out_pdf))
    lang = input("OCR 语言(默认 chi_sim+eng)：").strip() or "chi_sim+eng"
    cmd = ["ocrmypdf","--skip-text","--force-ocr","--rotate-pages","--jobs","4",
           "--language",lang,in_pdf,out_pdf]
    print("运行："," ".join(cmd))
    subprocess.run(cmd, check=False)
    print("OCR 输出：", out_pdf)

# ---------- 主流程 ----------
def run_ia(
    item_id:str,
    start_n:int,
    end_n:Optional[int]=None,
    mode:str="url",               # url / arrow / scroll
    viewport:Tuple[int,int]=(1600,1200),
    start_url:Optional[str]=None,  # 可选：自定义起始URL（如已登录页面）
    user_data_dir:Optional[str]=None,  # 可选：复用Chrome用户数据目录以保留登录态
    connect_to_existing:bool=False,  # 新增：连接到现有Chrome实例
    debug_port:int=9222,  # 新增：Chrome调试端口
) -> None:
    # 检查必要依赖
    if sync_playwright is None:
        raise RuntimeError("Playwright 未安装。请运行: pip install playwright && python -m playwright install")
    
    if Image is None:
        raise RuntimeError("PIL 未安装。请运行: pip install Pillow")
    
    if tk is None:
        raise RuntimeError("tkinter 未安装。请确保 Python 安装时包含 tkinter 模块。")
    img_dir = os.path.join(IMG_ROOT, item_id)
    tmp_dir = os.path.join(img_dir, "_tmp")
    _mkdir(img_dir); _mkdir(tmp_dir)

    with sync_playwright() as p:
        # 根据是否提供用户数据目录选择持久化或临时上下文
        browser = None
        ctx = None
        try:
            # 新增：连接到现有Chrome实例的逻辑
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
                             channel="chrome",  # 使用系统安装的 Chrome
                             args=[
                                 '--no-sandbox', 
                                 '--disable-dev-shm-usage',
                                 '--disable-web-security',  # 禁用网络安全限制
                                 '--disable-features=VizDisplayCompositor',  # 提高稳定性
                                 '--disable-extensions',  # 禁用扩展
                                 '--no-first-run',  # 跳过首次运行设置
                                 '--disable-default-apps',  # 禁用默认应用
                                 '--disable-popup-blocking',  # 禁用弹窗阻止
                             ]
                         )
                         print("✅ Chrome 浏览器启动成功")
                     except Exception as e:
                         print(f"❌ 无法启动 Chrome，回退到 Chromium: {e}")
                         try:
                             browser = p.chromium.launch(
                                 headless=False,
                                 args=[
                                     '--no-sandbox', 
                                     '--disable-dev-shm-usage',
                                     '--disable-web-security',
                                     '--disable-features=VizDisplayCompositor',
                                     '--disable-extensions',
                                     '--no-first-run',
                                     '--disable-default-apps',
                                     '--disable-popup-blocking',
                                 ]
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

            # 1) 打开起始页 —— 你手动登录/借阅 + 切 1up + 缩放 + 摆正
            if start_url:
                # 如果提供了自定义起始URL，先打开它
                print(f"打开自定义起始页面: {start_url}")
                page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
                print("✅ 页面加载完成")
                
                try:
                    print(f"\n🔥 重要提示：请不要点击浏览器右上角的菜单按钮（三个点），这会导致程序失控！")
                    print("📋 请在浏览器中完成以下操作：")
                    print(f"   1. 找到并打开书籍 {item_id}")
                    print("   2. 切换到单页模式(1up)")
                    print(f"   3. 导航到第 {start_n} 页")
                    print("   4. 缩放并把页面完整居中显示")
                    print("   5. 只使用页面内的按钮和链接，不要使用浏览器菜单")
                    input("\n✅ 完成后按回车继续… ")
                    
                    # 检查页面是否还存在
                    if page.is_closed():
                        raise RuntimeError("页面已关闭，可能是因为点击了浏览器菜单")
                        
                except KeyboardInterrupt:
                    print("\n用户中断操作")
                    return
            else:
                # 默认行为：直接打开书籍页面
                url0 = f"https://archive.org/details/{item_id}/page/n{start_n}/mode/1up"
                print(f"打开书籍页面: {url0}")
                page.goto(url0, wait_until="domcontentloaded", timeout=30000)
                print("✅ 页面加载完成")
                
                try:
                    print(f"\n🔥 重要提示：请不要点击浏览器右上角的菜单按钮（三个点），这会导致程序失控！")
                    print("📋 请在浏览器中完成以下操作：")
                    print("   1. 登录/借阅书籍")
                    print("   2. 切换到单页模式(1up)")
                    print(f"   3. 缩放并把第 {start_n} 页完整居中显示")
                    print("   4. 只使用页面内的按钮和链接，不要使用浏览器菜单")
                    input("\n✅ 完成后按回车继续… ")
                    
                    # 检查页面是否还存在
                    if page.is_closed():
                        raise RuntimeError("页面已关闭，可能是因为点击了浏览器菜单")
                        
                except KeyboardInterrupt:
                    print("\n用户中断操作")
                    return
        except Exception as e:
            print(f"❌ 浏览器操作失败: {e}")
            try:
                if ctx:
                    ctx.close()
                elif browser:
                    browser.close()
            except:
                pass
            raise

        # 2) 首张视口截图→框选裁剪区域
        hide_overlays(page); wait_stable(page)
        first_png = os.path.join(tmp_dir, "__first.png")
        page.screenshot(path=first_png, full_page=False)
        crop = select_crop_box(first_png)
        with open(os.path.join(img_dir, "crop.json"), "w", encoding="utf-8") as f:
            json.dump(crop.__dict__, f, ensure_ascii=False, indent=2)

        # 3) 计算结束 n
        if end_n is None:
            cnt = meta_imagecount(item_id)
            end_n = cnt-1 if cnt else start_n + 200  # 取不到时保守截 200 页

        # 4) 翻页并截图
        n = start_n
        saved = []
        while True:
            hide_overlays(page); wait_stable(page)
            raw = os.path.join(tmp_dir, f"__v_{n:05d}.png")
            out = os.path.join(img_dir, f"{n:05d}.png")
            page.screenshot(path=raw, full_page=False)
            im = Image.open(raw)
            l,t,r,b = max(0,crop.left), max(0,crop.top), min(im.width,crop.right), min(im.height,crop.bottom)
            im.crop((l,t,r,b)).save(out)
            saved.append(out)
            print(f"[OK] n{n} -> {out}")

            if n >= end_n: break
            if mode == "url":
                n += 1
                page.goto(replace_n(page.url, n), wait_until="domcontentloaded")
            elif mode == "arrow":
                old = parse_n(page.url) or -1
                page.keyboard.press("ArrowRight")
                # 等 n 改变
                for _ in range(60):
                    cur = parse_n(page.url) or old
                    if cur != old: n = cur; break
                    page.wait_for_timeout(200)
            else:  # scroll
                old = parse_n(page.url) or -1
                page.mouse.wheel(0, int(viewport[1]*0.85))
                for _ in range(60):
                    cur = parse_n(page.url) or old
                    if cur != old: n = cur; break
                    page.wait_for_timeout(200)

        ctx.close(); browser.close()

    # 5) 合并前预览 → 合并 → 可选 OCR
    try: os.startfile(img_dir)
    except: pass
    input("\n已完成截图，请快速预览。回车合并为 PDF… ")
    pdf_path = os.path.join(PDF_DIR, f"{item_id}.pdf")
    saved.sort()
    merge_to_pdf(saved, pdf_path)
    print("PDF 已保存：", pdf_path)
    ocr_pdf = os.path.join(OCR_DIR, f"{item_id}_OCR.pdf")
    maybe_ocr(pdf_path, ocr_pdf)