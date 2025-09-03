#!/usr/bin/env python3
"""
扫描 MCP/PDF 目录下所有 .py 文件，找出“没有任何有效注释/文档字符串”的文件。

判定规则（尽量接近“没加注释”的语义）：
- 模块级 docstring 存在即视为“有注释”；
- 存在以 # 开头的注释行且非 shebang(#!) 与 非编码标记(# -*- coding: ... -*)，则视为“有注释”；
- 仅有 shebang 或编码标记不视为“有注释”。
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT  # MCP/PDF

ENCODING_RE = re.compile(r"coding[:=]", re.IGNORECASE)

def has_meaningful_comments(text: str) -> bool:
    # 模块 docstring 检测
    try:
        mod = ast.parse(text)
        if ast.get_docstring(mod):
            return True
    except Exception:
        # 解析失败时仅依赖行级判断
        pass

    for line in text.splitlines():
        s = line.lstrip()
        if not s.startswith('#'):
            continue
        # 排除 shebang 与 编码标记
        if s.startswith('#!'):
            continue
        if ENCODING_RE.search(s):
            continue
        # 认为存在有效注释
        return True
    return False


def main() -> None:
    py_files = sorted([
        p for p in TARGET.rglob('*.py')
        if p.is_file()
        and '/.venv/' not in p.as_posix()
        and '/site-packages/' not in p.as_posix()
    ])
    no_comment = []
    for p in py_files:
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if not has_meaningful_comments(text):
            no_comment.append(p)
    for p in no_comment:
        print(p.as_posix())


if __name__ == '__main__':
    main()
