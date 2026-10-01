---
title: "RAG 文档切片：按标题切分 Markdown 并生成 Chunk"
date: 2026-10-01 13:22:21
tags:
  - RAG
  - Markdown
  - Python
---

在 RAG（检索增强生成）系统中，用户上传的文档不能直接交给 LLM 处理。上下文窗口有限，整篇文档还会带来检索噪声、较高的向量化成本，以及被长文本稀释的语义信息。

切分质量会直接影响后续检索和回答：切分太粗会带来噪声，切分太细会破坏上下文。比较实用的做法是：**先按 Markdown 标题切分，再对超长章节递归切分，最后合并同一父标题下的短片段**。

本文给出一个可以直接运行的实现。每个 Chunk 都会保留 `file_title`、`parent_title` 和 `part` 等元信息，方便写入向量数据库后追溯原文。

## 准备环境

需要 Python 3.10 或更高版本。安装依赖：

```bash
python -m pip install langchain-text-splitters
```

假设目录如下：

```text
.
├── chunk_markdown.py
└── input/
    └── report.md
```

## 处理流程

1. **预处理**：校验文件标题和正文，并统一换行符。
2. **按标题初切**：识别 1～6 级 Markdown 标题，同时跳过代码块中的标题。
3. **无标题兜底**：没有标题时，把全文作为一个章节处理。
4. **精细化**：用 `RecursiveCharacterTextSplitter` 切分超长章节，并合并同一父标题下的短 Chunk。
5. **输出结果**：打印统计信息，并将 Chunk 保存为可读的 JSON 文件。

## 完整代码

将下面的代码保存为 `chunk_markdown.py`：

```python
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

Chunk = dict[str, Any]
TITLE_RE = re.compile(r"^\s{0,3}#{1,6}\s+.+?\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]


def split_by_headers(content: str, file_title: str) -> list[Chunk]:
    """按 Markdown 标题切分正文，并忽略代码块中的标题。"""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[Chunk] = []
    current_title = ""
    current_lines: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(
                {
                    "title": current_title,
                    "content": text,
                    "file_title": file_title,
                    "parent_title": current_title,
                }
            )

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            current_lines.append(line)
            continue

        if fence is None and TITLE_RE.match(line):
            flush()
            current_title = line.strip()
            current_lines = [current_title]
        else:
            current_lines.append(line)

    flush()

    if not sections:
        return [
            {
                "title": "无标题",
                "content": content.strip(),
                "file_title": file_title,
                "parent_title": "无标题",
            }
        ]
    return sections


def split_long_section(section: Chunk, max_length: int) -> list[Chunk]:
    """将超长章节切成不超过 max_length 的子 Chunk。"""
    content = str(section.get("content", "")).strip()
    if len(content) <= max_length:
        return [dict(section)]

    title = str(section.get("title", ""))
    prefix = f"{title}\n\n" if title else ""
    available_length = max_length - len(prefix)
    if available_length <= 0:
        raise ValueError(f"标题长度已达到或超过 max_length：{title}")

    body = content
    if title and body.startswith(title):
        body = body[len(title):].lstrip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_length,
        chunk_overlap=0,
        separators=SEPARATORS,
    )

    chunks: list[Chunk] = []
    for part, text in enumerate(splitter.split_text(body), start=1):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            {
                "title": f"{title}-{part}" if title else f"chunk-{part}",
                "content": f"{prefix}{text}".strip(),
                "parent_title": title or "无标题",
                "part": part,
                "file_title": section.get("file_title", ""),
            }
        )
    return chunks


def merge_short_sections(
    sections: list[Chunk], min_length: int, max_length: int
) -> list[Chunk]:
    """只合并同一父标题下、且合并后不超过上限的相邻 Chunk。"""
    merged: list[Chunk] = []
    current: Chunk | None = None

    for section in sections:
        if current is None:
            current = dict(section)
            continue

        current_text = str(current["content"])
        next_text = str(section["content"])
        same_parent = current.get("parent_title") == section.get("parent_title")
        can_merge = (
            len(current_text) < min_length
            and same_parent
            and len(current_text) + 2 + len(next_text) <= max_length
        )

        if can_merge:
            current["content"] = f"{current_text}\n\n{next_text}".strip()
            if "part" in section:
                current["part"] = section["part"]
        else:
            merged.append(current)
            current = dict(section)

    if current is not None:
        merged.append(current)
    return merged


def chunk_markdown(
    markdown_path: Path,
    output_path: Path,
    max_length: int = 1500,
    min_length: int = 500,
) -> list[Chunk]:
    """读取 Markdown，切分并写出 JSON，返回最终 Chunk 列表。"""
    if max_length <= 0 or min_length < 0 or min_length > max_length:
        raise ValueError("需要满足：max_length > 0 且 0 <= min_length <= max_length")
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在：{markdown_path}")

    content = markdown_path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError("Markdown 文件内容不能为空")

    file_title = markdown_path.stem
    sections = split_by_headers(content, file_title)
    refined: list[Chunk] = []
    for section in sections:
        refined.extend(split_long_section(section, max_length))

    chunks = merge_short_sections(refined, min_length, max_length)
    for chunk in chunks:
        chunk.setdefault("parent_title", chunk.get("title", ""))
        chunk.setdefault("file_title", file_title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("原始文本：%d 行，最终生成 %d 个 Chunk", content.count("\n") + 1, len(chunks))
    logger.info("Chunk JSON 已保存：%s", output_path)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="按 Markdown 标题切分文档并输出 Chunk JSON")
    parser.add_argument("markdown", type=Path, help="待切分的 Markdown 文件")
    parser.add_argument(
        "-o", "--output", type=Path, help="输出 JSON 路径，默认与 Markdown 同目录"
    )
    parser.add_argument("--max-length", type=int, default=1500, help="Chunk 最大字符数")
    parser.add_argument("--min-length", type=int, default=500, help="触发合并的最小字符数")
    args = parser.parse_args()

    markdown_path = args.markdown.resolve()
    output_path = args.output or markdown_path.with_name(f"{markdown_path.stem}_chunks.json")
    chunk_markdown(
        markdown_path,
        output_path.resolve(),
        max_length=args.max_length,
        min_length=args.min_length,
    )


if __name__ == "__main__":
    main()
```

运行默认配置：

```bash
python chunk_markdown.py input/report.md
```

输出文件为 `input/report_chunks.json`。也可以调整 Chunk 长度和输出位置：

```bash
python chunk_markdown.py input/report.md \
  --max-length 1200 \
  --min-length 300 \
  --output output/report_chunks.json
```

## 关键实现说明

### 为什么要跳过代码块

代码示例中可能包含 `# 注释` 或类似标题的文本。如果不记录代码围栏状态，切分器会把这些内容误认为 Markdown 标题，导致章节结构被破坏。示例同时支持反引号和波浪号围栏。

### 为什么要保留父标题

超长章节会生成 `标题-1`、`标题-2` 这样的子 Chunk，但它们都保留同一个 `parent_title`。写入向量数据库后，检索结果可以通过父标题恢复上下文；`file_title` 则用于定位原始文档。

### `RecursiveCharacterTextSplitter` 的切分顺序

分隔符按“段落、换行、中文标点、英文标点、空格”的顺序尝试。优先使用更大的语义单元，只有在仍然超长时才继续细分；如果所有分隔符都无法切开，分割器会在长度上限处硬切。

### 合并短 Chunk 的边界

程序只合并相邻且 `parent_title` 相同的 Chunk，并检查合并后仍不超过 `max_length`。因此不会跨章节拼接，也不会因为“过短合并”重新产生超长 Chunk。

生成的 JSON 可以直接作为后续 Embedding 和向量数据库写入步骤的输入。实际项目中还可以根据 Embedding 模型的 token 上限，把 `max_length` 从字符数进一步换算为 token 数。
