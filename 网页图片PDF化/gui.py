#!/usr/bin/env python3
"""
图形用户界面

提供简单易用的GUI界面，方便用户进行网页截图和PDF生成操作。
"""

import asyncio
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
from pathlib import Path
from datetime import datetime
import logging
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_screenshot_pdf import (
    WebScreenshotPDF, 
    ScreenshotConfig, 
    OCRConfig, 
    PDFConfig
)
from config import get_config


class WebScreenshotGUI:
    """网页截图PDF生成器GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("网页截图PDF生成器")
        self.root.geometry("800x700")
        
        # 配置
        self.config = get_config()
        
        # 设置日志
        self.setup_logging()
        
        # 创建界面
        self.create_widgets()
        
        # 处理状态
        self.is_processing = False
        
    def setup_logging(self):
        """设置日志"""
        # 创建日志处理器，输出到GUI
        self.log_handler = GUILogHandler(self)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[self.log_handler]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_widgets(self):
        """创建界面组件"""
        # 创建笔记本控件（标签页）
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 单个URL标签页
        self.create_single_tab(notebook)
        
        # 批量处理标签页
        self.create_batch_tab(notebook)
        
        # 点击翻页标签页
        self.create_clicks_tab(notebook)
        
        # 设置标签页
        self.create_settings_tab(notebook)
        
        # 日志标签页
        self.create_log_tab(notebook)
    
    def create_single_tab(self, notebook):
        """创建单个URL处理标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="单个URL")
        
        # URL输入
        ttk.Label(frame, text="网页URL:").pack(anchor=tk.W, pady=(10, 5))
        self.url_entry = ttk.Entry(frame, width=80)
        self.url_entry.pack(fill=tk.X, pady=(0, 10))
        self.url_entry.insert(0, "https://www.example.com")
        
        # 输出文件
        output_frame = ttk.Frame(frame)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_frame, text="输出PDF:").pack(anchor=tk.W)
        
        file_frame = ttk.Frame(output_frame)
        file_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.output_entry = ttk.Entry(file_frame)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.output_entry.insert(0, "screenshot.pdf")
        
        ttk.Button(file_frame, text="浏览", 
                  command=self.browse_output_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 选项框架
        options_frame = ttk.LabelFrame(frame, text="选项")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 第一行选项
        row1 = ttk.Frame(options_frame)
        row1.pack(fill=tk.X, padx=10, pady=5)
        
        # 浏览器引擎
        ttk.Label(row1, text="浏览器:").pack(side=tk.LEFT)
        # 默认使用selenium，因为它更容易安装和配置
        default_browser = "selenium"
        self.browser_var = tk.StringVar(value=default_browser)
        browser_combo = ttk.Combobox(row1, textvariable=self.browser_var, 
                                   values=["selenium", "playwright"], 
                                   state="readonly", width=12)
        browser_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        # 全页面截图
        self.fullpage_var = tk.BooleanVar(value=self.config.screenshot.full_page)
        ttk.Checkbutton(row1, text="全页面截图", 
                       variable=self.fullpage_var).pack(side=tk.LEFT, padx=(0, 20))
        
        # OCR开关
        self.ocr_var = tk.BooleanVar(value=self.config.pdf.include_ocr_text)
        ttk.Checkbutton(row1, text="启用OCR", 
                       variable=self.ocr_var).pack(side=tk.LEFT)
        
        # 第二行选项
        row2 = ttk.Frame(options_frame)
        row2.pack(fill=tk.X, padx=10, pady=5)
        
        # 窗口大小
        ttk.Label(row2, text="窗口大小:").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value=str(self.config.browser.window_width))
        width_entry = ttk.Entry(row2, textvariable=self.width_var, width=8)
        width_entry.pack(side=tk.LEFT, padx=(5, 2))
        
        ttk.Label(row2, text="×").pack(side=tk.LEFT)
        
        self.height_var = tk.StringVar(value=str(self.config.browser.window_height))
        height_entry = ttk.Entry(row2, textvariable=self.height_var, width=8)
        height_entry.pack(side=tk.LEFT, padx=(2, 20))
        
        # 等待时间
        ttk.Label(row2, text="等待时间:").pack(side=tk.LEFT)
        self.wait_var = tk.StringVar(value=str(self.config.browser.wait_time))
        wait_entry = ttk.Entry(row2, textvariable=self.wait_var, width=5)
        wait_entry.pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(row2, text="秒").pack(side=tk.LEFT, padx=(0, 20))
        
        # OCR引擎
        ttk.Label(row2, text="OCR引擎:").pack(side=tk.LEFT)
        self.ocr_engine_var = tk.StringVar(value=self.config.ocr.engine)
        ocr_combo = ttk.Combobox(row2, textvariable=self.ocr_engine_var,
                               values=["auto", "tesseract", "easyocr", "paddleocr"],
                               state="readonly", width=12)
        ocr_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # 处理按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="开始处理", 
                                       command=self.process_single_url)
        self.process_button.pack(side=tk.LEFT)
        
        self.stop_button = ttk.Button(button_frame, text="停止", 
                                    command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # 进度条
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # 状态标签
        self.status_label = ttk.Label(frame, text="就绪")
        self.status_label.pack(anchor=tk.W, pady=(5, 0))
    
    def create_batch_tab(self, notebook):
        """创建批量处理标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="批量处理")
        
        # URL列表输入
        ttk.Label(frame, text="URL列表 (每行一个URL):").pack(anchor=tk.W, pady=(10, 5))
        
        self.urls_text = scrolledtext.ScrolledText(frame, height=10, width=80)
        self.urls_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 示例URL
        example_urls = """https://www.example.com
https://www.github.com
https://www.stackoverflow.com"""
        self.urls_text.insert(tk.END, example_urls)
        
        # 文件操作按钮
        file_buttons = ttk.Frame(frame)
        file_buttons.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_buttons, text="从文件加载", 
                  command=self.load_urls_from_file).pack(side=tk.LEFT)
        ttk.Button(file_buttons, text="保存到文件", 
                  command=self.save_urls_to_file).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(file_buttons, text="清空", 
                  command=lambda: self.urls_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=(10, 0))
        
        # 输出文件
        output_frame = ttk.Frame(frame)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_frame, text="输出PDF:").pack(anchor=tk.W)
        
        file_frame = ttk.Frame(output_frame)
        file_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.batch_output_entry = ttk.Entry(file_frame)
        self.batch_output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.batch_output_entry.insert(0, "batch_screenshots.pdf")
        
        ttk.Button(file_frame, text="浏览", 
                  command=self.browse_batch_output_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 批量选项
        batch_options = ttk.LabelFrame(frame, text="批量选项")
        batch_options.pack(fill=tk.X, pady=(0, 10))
        
        options_row = ttk.Frame(batch_options)
        options_row.pack(fill=tk.X, padx=10, pady=5)
        
        # 并发数
        ttk.Label(options_row, text="并发数:").pack(side=tk.LEFT)
        self.concurrent_var = tk.StringVar(value=str(self.config.processing.max_concurrent_tasks))
        concurrent_entry = ttk.Entry(options_row, textvariable=self.concurrent_var, width=5)
        concurrent_entry.pack(side=tk.LEFT, padx=(5, 20))
        
        # 延迟
        ttk.Label(options_row, text="延迟:").pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="0")
        delay_entry = ttk.Entry(options_row, textvariable=self.delay_var, width=5)
        delay_entry.pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(options_row, text="秒").pack(side=tk.LEFT, padx=(0, 20))
        
        # 重试次数
        ttk.Label(options_row, text="重试:").pack(side=tk.LEFT)
        self.retry_var = tk.StringVar(value=str(self.config.processing.retry_attempts))
        retry_entry = ttk.Entry(options_row, textvariable=self.retry_var, width=5)
        retry_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # 处理按钮
        batch_button_frame = ttk.Frame(frame)
        batch_button_frame.pack(fill=tk.X, pady=10)
        
        self.batch_process_button = ttk.Button(batch_button_frame, text="开始批量处理", 
                                             command=self.process_batch_urls)
        self.batch_process_button.pack(side=tk.LEFT)
        
        self.batch_stop_button = ttk.Button(batch_button_frame, text="停止", 
                                          command=self.stop_processing, state=tk.DISABLED)
        self.batch_stop_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # 批量进度条
        self.batch_progress = ttk.Progressbar(frame, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=(10, 0))
        
        # 批量状态
        self.batch_status_label = ttk.Label(frame, text="就绪")
        self.batch_status_label.pack(anchor=tk.W, pady=(5, 0))
    
    def create_clicks_tab(self, notebook):
        """创建点击翻页标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="点击翻页")
        
        # 主容器
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # URL输入
        url_frame = ttk.LabelFrame(main_frame, text="起始URL")
        url_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.clicks_url_entry = ttk.Entry(url_frame)
        self.clicks_url_entry.pack(fill=tk.X, padx=10, pady=10)
        self.clicks_url_entry.insert(0, "https://example.com/page/1")
        
        # 翻页配置
        nav_frame = ttk.LabelFrame(main_frame, text="翻页配置")
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        nav_inner = ttk.Frame(nav_frame)
        nav_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # 第一行：下一页选择器
        row1 = ttk.Frame(nav_inner)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row1, text="下一页选择器:").pack(side=tk.LEFT)
        self.next_selector_entry = ttk.Entry(row1, width=40)
        self.next_selector_entry.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        self.next_selector_entry.insert(0, ".next-page, .page-next, [aria-label*='next']")
        
        # 第二行：图片选择器
        row2 = ttk.Frame(nav_inner)
        row2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row2, text="图片选择器:").pack(side=tk.LEFT)
        self.image_selector_entry = ttk.Entry(row2, width=40)
        self.image_selector_entry.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        self.image_selector_entry.insert(0, "img.main-image, .content img, .page-image")
        
        # 第三行：选项
        row3 = ttk.Frame(nav_inner)
        row3.pack(fill=tk.X, pady=(0, 5))
        
        self.use_arrow_keys_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="使用键盘箭头键翻页", 
                       variable=self.use_arrow_keys_var).pack(side=tk.LEFT, padx=(0, 20))
        
        self.interactive_crop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="启用交互式裁剪", 
                       variable=self.interactive_crop_var).pack(side=tk.LEFT)
        
        # PDF配置
        pdf_frame = ttk.LabelFrame(main_frame, text="PDF配置")
        pdf_frame.pack(fill=tk.X, pady=(0, 10))
        
        pdf_inner = ttk.Frame(pdf_frame)
        pdf_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # PDF标题和最大页数
        pdf_row1 = ttk.Frame(pdf_inner)
        pdf_row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(pdf_row1, text="PDF标题:").pack(side=tk.LEFT)
        self.pdf_title_entry = ttk.Entry(pdf_row1, width=30)
        self.pdf_title_entry.pack(side=tk.LEFT, padx=(5, 20))
        self.pdf_title_entry.insert(0, "点击翻页抓取")
        
        ttk.Label(pdf_row1, text="最大页数:").pack(side=tk.LEFT)
        self.max_pages_entry = ttk.Entry(pdf_row1, width=10)
        self.max_pages_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.max_pages_entry.insert(0, "100")
        
        # PDF选项
        pdf_row2 = ttk.Frame(pdf_inner)
        pdf_row2.pack(fill=tk.X, pady=(0, 5))
        
        self.auto_ocr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pdf_row2, text="自动执行OCR", 
                       variable=self.auto_ocr_var).pack(side=tk.LEFT, padx=(0, 20))
        
        self.keep_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pdf_row2, text="保留临时图片", 
                       variable=self.keep_images_var).pack(side=tk.LEFT, padx=(0, 20))
        
        self.enable_preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pdf_row2, text="启用预览", 
                       variable=self.enable_preview_var).pack(side=tk.LEFT)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="开始点击翻页抓取", 
                  command=self.process_clicks_to_pdf).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="仅截图（不合并PDF）", 
                  command=self.process_clicks_only).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="打开输出目录", 
                  command=self.open_output_dirs).pack(side=tk.LEFT, padx=(0, 10))
        
        self.clicks_stop_button = ttk.Button(button_frame, text="停止", 
                                           command=self.stop_processing, state=tk.DISABLED)
        self.clicks_stop_button.pack(side=tk.RIGHT)
        
        # 进度显示
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.clicks_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.clicks_progress.pack(fill=tk.X, pady=(0, 5))
        
        self.clicks_status_label = ttk.Label(progress_frame, text="就绪")
        self.clicks_status_label.pack(anchor=tk.W)
    
    def create_settings_tab(self, notebook):
        """创建设置标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="设置")
        
        # 创建滚动框架
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # PDF设置
        pdf_frame = ttk.LabelFrame(scrollable_frame, text="PDF设置")
        pdf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # PDF标题
        ttk.Label(pdf_frame, text="标题:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pdf_title_var = tk.StringVar(value=self.config.pdf.title or "")
        ttk.Entry(pdf_frame, textvariable=self.pdf_title_var, width=40).grid(row=0, column=1, padx=5, pady=2)
        
        # PDF作者
        ttk.Label(pdf_frame, text="作者:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.pdf_author_var = tk.StringVar(value=self.config.pdf.author or "")
        ttk.Entry(pdf_frame, textvariable=self.pdf_author_var, width=40).grid(row=1, column=1, padx=5, pady=2)
        
        # 页面大小
        ttk.Label(pdf_frame, text="页面大小:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.page_size_var = tk.StringVar(value=self.config.pdf.page_size)
        page_size_combo = ttk.Combobox(pdf_frame, textvariable=self.page_size_var,
                                     values=["A4", "letter", "custom"], state="readonly")
        page_size_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # OCR设置
        ocr_frame = ttk.LabelFrame(scrollable_frame, text="OCR设置")
        ocr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # OCR语言
        ttk.Label(ocr_frame, text="语言:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ocr_lang_var = tk.StringVar(value=",".join(self.config.ocr.languages))
        ttk.Entry(ocr_frame, textvariable=self.ocr_lang_var, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        # 置信度阈值
        ttk.Label(ocr_frame, text="置信度阈值:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.confidence_var = tk.StringVar(value=str(self.config.ocr.confidence_threshold))
        ttk.Entry(ocr_frame, textvariable=self.confidence_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 预处理选项
        self.preprocess_var = tk.BooleanVar(value=self.config.ocr.preprocess)
        ttk.Checkbutton(ocr_frame, text="图片预处理", variable=self.preprocess_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # 保存设置按钮
        ttk.Button(scrollable_frame, text="保存设置", command=self.save_settings).pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_log_tab(self, notebook):
        """创建日志标签页"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="日志")
        
        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(frame, height=25, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 日志控制按钮
        log_buttons = ttk.Frame(frame)
        log_buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(log_buttons, text="清空日志", 
                  command=lambda: self.log_text.delete(1.0, tk.END)).pack(side=tk.LEFT)
        ttk.Button(log_buttons, text="保存日志", 
                  command=self.save_log).pack(side=tk.LEFT, padx=(10, 0))
    
    def browse_output_file(self):
        """浏览输出文件"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)
    
    def browse_batch_output_file(self):
        """浏览批量输出文件"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.batch_output_entry.delete(0, tk.END)
            self.batch_output_entry.insert(0, filename)
    
    def load_urls_from_file(self):
        """从文件加载URL"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.urls_text.delete(1.0, tk.END)
                self.urls_text.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {e}")
    
    def save_urls_to_file(self):
        """保存URL到文件"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                content = self.urls_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("成功", "URL列表已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件失败: {e}")
    
    def save_log(self):
        """保存日志"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                content = self.log_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("成功", "日志已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存日志失败: {e}")
    
    def save_settings(self):
        """保存设置"""
        try:
            # 更新配置
            self.config.pdf.title = self.pdf_title_var.get()
            self.config.pdf.author = self.pdf_author_var.get()
            self.config.pdf.page_size = self.page_size_var.get()
            self.config.ocr.languages = [lang.strip() for lang in self.ocr_lang_var.get().split(',')]
            self.config.ocr.confidence_threshold = float(self.confidence_var.get())
            self.config.ocr.preprocess = self.preprocess_var.get()
            
            messagebox.showinfo("成功", "设置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败: {e}")
    
    def process_single_url(self):
        """处理单个URL"""
        if self.is_processing:
            return
        
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("错误", "请输入URL")
            return
        
        output_path = self.output_entry.get().strip()
        if not output_path:
            messagebox.showerror("错误", "请指定输出文件")
            return
        
        # 开始处理
        self.start_processing()
        
        # 在新线程中运行异步任务
        thread = threading.Thread(target=self._run_single_process, args=(url, output_path))
        thread.daemon = True
        thread.start()
    
    def _run_single_process(self, url, output_path):
        """运行单个URL处理（在线程中）"""
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 运行异步任务
            result = loop.run_until_complete(self._process_single_async(url, output_path))
            
            # 更新UI
            self.root.after(0, self._on_single_complete, result)
            
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))
        finally:
            loop.close()
    
    async def _process_single_async(self, url, output_path):
        """异步处理单个URL"""
        # 创建配置
        screenshot_config = ScreenshotConfig(
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            full_page=self.fullpage_var.get(),
            wait_time=int(self.wait_var.get())
        )
        
        ocr_config = OCRConfig(
            engine=self.ocr_engine_var.get(),
            languages=self.config.ocr.languages,
            confidence_threshold=self.config.ocr.confidence_threshold,
            preprocess=self.config.ocr.preprocess
        )
        
        pdf_config = PDFConfig(
            include_ocr_text=self.ocr_var.get(),
            searchable=self.ocr_var.get(),
            title=self.config.pdf.title,
            author=self.config.pdf.author
        )
        
        # 处理
        async with WebScreenshotPDF(browser_engine=self.browser_var.get()) as processor:
            result = await processor.process_urls_to_pdf(
                urls=[url],
                output_path=Path(output_path),
                screenshot_config=screenshot_config,
                ocr_config=ocr_config,
                pdf_config=pdf_config
            )
            
        return result
    
    def process_batch_urls(self):
        """处理批量URL"""
        if self.is_processing:
            return
        
        # 获取URL列表
        urls_text = self.urls_text.get(1.0, tk.END).strip()
        if not urls_text:
            messagebox.showerror("错误", "请输入URL列表")
            return
        
        urls = [line.strip() for line in urls_text.split('\n') 
                if line.strip() and not line.startswith('#')]
        
        if not urls:
            messagebox.showerror("错误", "没有有效的URL")
            return
        
        output_path = self.batch_output_entry.get().strip()
        if not output_path:
            messagebox.showerror("错误", "请指定输出文件")
            return
        
        # 开始处理
        self.start_processing()
        self.batch_progress['maximum'] = len(urls)
        self.batch_progress['value'] = 0
        
        # 在新线程中运行异步任务
        thread = threading.Thread(target=self._run_batch_process, args=(urls, output_path))
        thread.daemon = True
        thread.start()
    
    def _run_batch_process(self, urls, output_path):
        """运行批量处理（在线程中）"""
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 运行异步任务
            result = loop.run_until_complete(self._process_batch_async(urls, output_path))
            
            # 更新UI
            self.root.after(0, self._on_batch_complete, result)
            
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))
        finally:
            loop.close()
    
    async def _process_batch_async(self, urls, output_path):
        """异步处理批量URL"""
        # 创建配置
        screenshot_config = ScreenshotConfig(
            width=self.config.browser.window_width,
            height=self.config.browser.window_height,
            full_page=self.config.screenshot.full_page,
            wait_time=self.config.browser.wait_time
        )
        
        ocr_config = OCRConfig(
            engine=self.config.ocr.engine,
            languages=self.config.ocr.languages,
            confidence_threshold=self.config.ocr.confidence_threshold,
            preprocess=self.config.ocr.preprocess
        )
        
        pdf_config = PDFConfig(
            include_ocr_text=self.config.pdf.include_ocr_text,
            searchable=self.config.pdf.searchable,
            title=self.config.pdf.title,
            author=self.config.pdf.author
        )
        
        # 处理
        async with WebScreenshotPDF(browser_engine=self.config.browser.engine) as processor:
            # 逐个处理以更新进度
            processed_count = 0
            images_and_texts = []
            failed_urls = []
            
            for i, url in enumerate(urls):
                try:
                    # 更新进度
                    self.root.after(0, self._update_batch_progress, i, f"处理: {url}")
                    
                    # 截图
                    screenshot_path = await processor.screenshot_url(url, screenshot_config)
                    
                    # OCR
                    ocr_result = await processor.extract_text_ocr(screenshot_path, ocr_config)
                    
                    images_and_texts.append((screenshot_path, ocr_result.get("text", "")))
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"处理URL失败 {url}: {e}")
                    failed_urls.append({"url": url, "error": str(e)})
            
            # 创建PDF
            if images_and_texts:
                processor.create_pdf_with_images_and_text(
                    images_and_texts, 
                    Path(output_path), 
                    pdf_config
                )
            
            return {
                "success": len(images_and_texts) > 0,
                "total_urls": len(urls),
                "processed_urls": processed_count,
                "failed_urls": failed_urls,
                "output_path": output_path
            }
    
    def _update_batch_progress(self, current, status):
        """更新批量处理进度"""
        self.batch_progress['value'] = current
        self.batch_status_label.config(text=status)
    
    def start_processing(self):
        """开始处理"""
        self.is_processing = True
        self.process_button.config(state=tk.DISABLED)
        self.batch_process_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.batch_stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.status_label.config(text="处理中...")
        self.batch_status_label.config(text="处理中...")
    
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        self.process_button.config(state=tk.NORMAL)
        self.batch_process_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.batch_stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        self.status_label.config(text="已停止")
        self.batch_status_label.config(text="已停止")
    
    def _on_single_complete(self, result):
        """单个URL处理完成"""
        self.stop_processing()
        
        if result['success']:
            self.status_label.config(text=f"完成 - 用时: {result['processing_time']:.1f}秒")
            messagebox.showinfo("成功", f"PDF已生成: {result['output_path']}")
        else:
            self.status_label.config(text="处理失败")
            messagebox.showerror("失败", f"处理失败: {result.get('error', '未知错误')}")
    
    def _on_batch_complete(self, result):
        """批量处理完成"""
        self.stop_processing()
        self.batch_progress['value'] = self.batch_progress['maximum']
        
        if result['success']:
            self.batch_status_label.config(text=f"完成 - 成功: {result['processed_urls']}/{result['total_urls']}")
            messagebox.showinfo("成功", 
                              f"批量处理完成!\n"
                              f"成功: {result['processed_urls']}\n"
                              f"失败: {len(result['failed_urls'])}\n"
                              f"输出: {result['output_path']}")
        else:
            self.batch_status_label.config(text="批量处理失败")
            messagebox.showerror("失败", "批量处理失败")
    
    def _on_process_error(self, error):
        """处理错误"""
        self.stop_processing()
        self.status_label.config(text="错误")
        self.batch_status_label.config(text="错误")
        messagebox.showerror("错误", f"处理过程中发生错误: {error}")
    
    def process_clicks_to_pdf(self):
        """处理点击翻页到PDF"""
        url = self.clicks_url_entry.get().strip()
        if not url:
            messagebox.showerror("错误", "请输入起始URL")
            return
        
        # 构建配置
        screenshot_config = ScreenshotConfig(
            image_selector=self.image_selector_entry.get().strip() or None,
            next_selector=self.next_selector_entry.get().strip() or None,
            use_arrow_keys=self.use_arrow_keys_var.get(),
            interactive_crop=self.interactive_crop_var.get(),
            lazy_scroll=True,
            lazy_timeout=3.0
        )
        
        pdf_config = PDFConfig(
            title=self.pdf_title_entry.get().strip() or "点击翻页抓取",
            max_pages=int(self.max_pages_entry.get() or 100),
            auto_ocr=self.auto_ocr_var.get(),
            keep_temp_images=self.keep_images_var.get(),
            enable_preview=self.enable_preview_var.get()
        )
        
        self.clicks_progress.start()
        self.clicks_status_label.config(text="正在处理...")
        self.clicks_stop_button.config(state=tk.NORMAL)
        
        # 在新线程中运行
        thread = threading.Thread(
            target=self._run_clicks_to_pdf_process,
            args=(url, screenshot_config, pdf_config)
        )
        thread.daemon = True
        thread.start()
    
    def process_clicks_only(self):
        """仅处理点击翻页截图"""
        url = self.clicks_url_entry.get().strip()
        if not url:
            messagebox.showerror("错误", "请输入起始URL")
            return
        
        # 构建配置
        screenshot_config = ScreenshotConfig(
            image_selector=self.image_selector_entry.get().strip() or None,
            next_selector=self.next_selector_entry.get().strip() or None,
            use_arrow_keys=self.use_arrow_keys_var.get(),
            interactive_crop=self.interactive_crop_var.get(),
            lazy_scroll=True,
            lazy_timeout=3.0
        )
        
        self.clicks_progress.start()
        self.clicks_status_label.config(text="正在截图...")
        self.clicks_stop_button.config(state=tk.NORMAL)
        
        # 在新线程中运行
        thread = threading.Thread(
            target=self._run_clicks_only_process,
            args=(url, screenshot_config)
        )
        thread.daemon = True
        thread.start()
    
    def _run_clicks_to_pdf_process(self, url, screenshot_config, pdf_config):
        """运行点击翻页到PDF流程"""
        try:
            result = asyncio.run(self._process_clicks_to_pdf_async(url, screenshot_config, pdf_config))
            self.root.after(0, self._on_clicks_complete, result)
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))
    
    def _run_clicks_only_process(self, url, screenshot_config):
        """运行仅点击翻页截图流程"""
        try:
            result = asyncio.run(self._process_clicks_only_async(url, screenshot_config))
            self.root.after(0, self._on_clicks_complete, result)
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))
    
    async def _process_clicks_to_pdf_async(self, url, screenshot_config, pdf_config):
        """异步处理点击翻页到PDF"""
        async with WebScreenshotPDF(browser_engine="playwright") as processor:
            return await processor.process_clicks_to_pdf(url, screenshot_config, pdf_config)
    
    async def _process_clicks_only_async(self, url, screenshot_config):
        """异步处理仅点击翻页截图"""
        import tempfile
        from pathlib import Path
        
        output_dir = Path(tempfile.mkdtemp(prefix="web_shots_"))
        
        async with WebScreenshotPDF(browser_engine="playwright") as processor:
            result_dir = await processor.capture_by_clicks(url, output_dir, screenshot_config)
            
            image_files = list(result_dir.glob("*.png"))
            return {
                "success": True,
                "images_dir": str(result_dir),
                "image_count": len(image_files),
                "type": "screenshots_only"
            }
    
    def _on_clicks_complete(self, result):
        """点击翻页完成回调"""
        self.clicks_progress.stop()
        self.clicks_status_label.config(text="完成")
        self.clicks_stop_button.config(state=tk.DISABLED)
        
        if result.get('success'):
            if result.get('type') == 'screenshots_only':
                self.log_message(f"✅ 截图完成! 共 {result['image_count']} 张图片")
                self.log_message(f"📁 图片目录: {result['images_dir']}")
                messagebox.showinfo("完成", f"截图完成!\n共 {result['image_count']} 张图片\n目录: {result['images_dir']}")
            else:
                self.log_message(f"✅ PDF生成完成! 共 {result['image_count']} 张图片")
                self.log_message(f"📄 PDF文件: {result['pdf_path']}")
                if result.get('ocr_path'):
                    self.log_message(f"🔤 OCR PDF: {result['ocr_path']}")
                
                msg = f"PDF生成完成!\n共 {result['image_count']} 张图片\nPDF: {result['pdf_path']}"
                if result.get('ocr_path'):
                    msg += f"\nOCR PDF: {result['ocr_path']}"
                messagebox.showinfo("完成", msg)
        else:
            self.log_message(f"❌ 处理失败: {result.get('error', '未知错误')}")
            messagebox.showerror("错误", f"处理失败:\n{result.get('error', '未知错误')}")
    
    def open_output_dirs(self):
        """打开输出目录"""
        try:
            from config import get_config
            config = get_config()
            
            # 创建目录（如果不存在）
            config.pdf.output_dir.mkdir(parents=True, exist_ok=True)
            config.pdf.ocr_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 打开目录
            if os.name == 'nt':  # Windows
                os.startfile(str(config.pdf.output_dir))
                os.startfile(str(config.pdf.ocr_output_dir))
            else:  # Linux/Mac
                import subprocess
                subprocess.run(['xdg-open', str(config.pdf.output_dir)], check=False)
                subprocess.run(['xdg-open', str(config.pdf.ocr_output_dir)], check=False)
            
            self.log_message("📁 已打开输出目录")
        except Exception as e:
            self.log_message(f"❌ 打开目录失败: {e}")
            messagebox.showerror("错误", f"打开目录失败:\n{e}")
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)


class GUILogHandler(logging.Handler):
    """GUI日志处理器"""
    
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # 在主线程中更新GUI
            self.gui.root.after(0, self.gui.log_message, msg)
        except Exception:
            pass


def main():
    """主函数"""
    root = tk.Tk()
    app = WebScreenshotGUI(root)
    
    # 设置窗口图标（如果有的话）
    try:
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    # 运行GUI
    root.mainloop()


if __name__ == "__main__":
    main()