#!/usr/bin/env python3
"""
MCP客户端测试脚本
用于测试PDF-MCP服务器的功能
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

class MCPClient:
    def __init__(self, server_command):
        self.server_command = server_command
        self.process = None
    
    async def start_server(self):
        """启动MCP服务器进程"""
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print("MCP服务器已启动")
    
    async def send_request(self, method, params=None):
        """发送MCP请求"""
        if not self.process:
            raise RuntimeError("服务器未启动")
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # 读取响应
        response_line = await self.process.stdout.readline()
        if response_line:
            try:
                response = json.loads(response_line.decode().strip())
                return response
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始响应: {response_line.decode()}")
                return None
        return None
    
    async def initialize(self):
        """初始化MCP连接"""
        # 发送初始化请求
        response = await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
        
        if response and "result" in response:
            print("MCP连接初始化成功")
            # 发送initialized通知
            await self.send_request("notifications/initialized")
            return True
        else:
            print(f"MCP初始化失败: {response}")
            return False
    
    async def close(self):
        """关闭服务器进程"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print("MCP服务器已关闭")

async def test_pdf_processing():
    """测试PDF处理功能"""
    # PDF文件路径
    pdf_path = "G:\\E盘\\工作项目文件\\AI_Agent\\Trae_Abroad\\MCP_run_PDF\\14531.pdf"
    
    # 检查PDF文件是否存在
    if not Path(pdf_path).exists():
        print(f"错误: PDF文件不存在: {pdf_path}")
        return
    
    # 启动MCP客户端
    server_command = [sys.executable, "-m", "src.pdf_mcp_server.main"]
    client = MCPClient(server_command)
    
    try:
        await client.start_server()
        
        # 等待服务器启动
        await asyncio.sleep(2)
        
        # 初始化MCP连接
        if not await client.initialize():
            print("无法初始化MCP连接")
            return
        
        print("\n=== 测试PDF-MCP服务器功能 ===")
        print(f"处理PDF文件: {pdf_path}")
        
        # 测试1: 检测PDF类型
        print("\n1. 检测PDF类型...")
        response = await client.send_request("tools/call", {
            "name": "detect_pdf_type",
            "arguments": {
                "file_path": pdf_path
            }
        })
        if response:
            print(f"响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        # 测试2: 提取元数据
        print("\n2. 提取PDF元数据...")
        response = await client.send_request("tools/call", {
            "name": "extract_metadata",
            "arguments": {
                "file_path": pdf_path
            }
        })
        if response:
            print(f"响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        # 测试3: 读取文本内容（前500字符）
        print("\n3. 读取PDF文本内容...")
        response = await client.send_request("tools/call", {
            "name": "read_text",
            "arguments": {
                "file_path": pdf_path,
                "pages": [1, 2],  # 只读取前两页
                "max_chars": 500
            }
        })
        if response:
            print(f"响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        # 测试4: 分析PDF
        print("\n4. 分析PDF结构...")
        response = await client.send_request("tools/call", {
            "name": "analyze_pdf",
            "arguments": {
                "file_path": pdf_path,
                "analysis_type": "basic"
            }
        })
        if response:
            print(f"响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_pdf_processing())