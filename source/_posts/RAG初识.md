---
title: "RAG 初识：让大语言模型基于知识库回答问题"
date: 2026-09-28 21:05:04
tags:
  - RAG
  - LLM
---

**RAG（Retrieval-Augmented Generation，检索增强生成）是一种让大语言模型基于外部知识库回答问题的技术。**

RAG 的核心思路很简单：先从知识库中找出与问题最相关的内容，再把这些内容和用户问题一起交给 LLM，让模型基于检索到的上下文生成答案。

## 本系列使用的项目框架

后面的文章都会在同一个项目中继续完善。下面是整个系列的目录规划：脚本按章节添加，数据文件由运行命令生成；当前第 5 章已经提供可运行的向量化与入库入口，检索和回答仍放在后续章节。

```text
RAG/
├── .env                  # 本地密钥和服务地址，不提交到 Git
├── .env.example          # 配置项示例
├── data/
│   ├── input/             # 原始文件，例如 report.pdf
│   ├── parsed/            # MinerU 解压后的 Markdown 和图片
│   │   └── report/
│   │       ├── full.md
│   │       ├── full_new.md
│   │       └── images/
│   └── chunks/            # 切片后的 JSON 文件
│       └── report_chunks.json
├── models/
│   └── bge-m3/            # 本地 Embedding 模型（可选）
├── scripts/
│   ├── mini_rag.py        # 第 1 章：检索与提示词组装演示
│   ├── parse_pdf.py       # 第 2 章：PDF 解析
│   ├── process_images.py  # 第 3 章：图片处理
│   ├── chunk_markdown.py  # 第 4 章：文档切片
│   ├── embed_and_insert.py # 第 5 章：向量化与入库
│   └── ask.py             # 后续：检索、生成与引用（规划）
├── config/                # Embedding、Milvus 等配置
├── utils/                 # 可复用的项目工具
├── tool/                  # 日志等基础设施
├── requirements.txt
└── README.md
```

每一章都会说明“本章新增什么、输入在哪里、如何运行、输出是什么”。下文所有相对路径都以项目根目录 `RAG/` 为起点。例如项目放在 `D:/Desktop/RAG`，就先在终端进入这个目录，再执行各章命令。系列统一使用 Python 3.10 或更高版本，依赖按章节安装；新配置追加到根目录的 `.env`，保留前面已填写的配置。

| 章节 | 新增脚本 | 输入 → 输出 |
| --- | --- | --- |
| 本篇 | `scripts/mini_rag.py` | 内置示例资料 → 控制台中的召回结果和提示词 |
| {% post_link RAG文档解析1 文档解析 %} | `scripts/parse_pdf.py` | `data/input/report.pdf` → `data/parsed/report/full.md` 与 `images/` |
| {% post_link RAG文档解析2 图片处理 %} | `scripts/process_images.py` | `data/parsed/report/full.md` → 同目录的 `full_new.md` |
| {% post_link RAG文档切片 文档切片 %} | `scripts/chunk_markdown.py` | `full_new.md` → `data/chunks/report_chunks.json` |
| {% post_link RAG向量化与向量数据库 向量化与入库 %} | `scripts/embed_and_insert.py` | Chunk JSON → Milvus 中的向量与元数据 |
| 检索与回答（规划） | `scripts/ask.py` | 用户问题 → 检索片段 → 带来源的答案 |

先跑通本篇的小例子，再沿着同一份文档完成后续步骤。看到中间产物时，可以打开它检查处理结果，再进入下一章。

## RAG 的架构流程

![RAG 整体架构流程图](https://www.runoob.com/wp-content/uploads/2026/06/11-rag-architecture.svg)

一个基础的 RAG 系统通常包含两个阶段：知识库构建，以及检索召回与生成。

### 知识库构建

知识来源可以是 PDF、DOCX、TXT、Markdown 等。实际使用时，通常会先把不同格式的文档统一解析为结构清晰的文本，再进行清洗和切分。Markdown 不是 RAG 的硬性要求，但它保留了标题、段落和列表等结构，便于后续处理，也方便人工检查。

文档切分是因为 LLM 的上下文窗口和注意力都有限。按标题和段落将长文档切成较小的片段后，检索时只需将相关内容放入上下文，可以减少无关信息和 token 消耗，同时保留章节结构，方便溯源。

切分后的片段会经过 Embedding 模型向量化，并写入向量数据库。向量数据库保存了片段内容及其向量，供后续相似度检索使用。

### 检索召回与生成

当问题依赖模型训练数据之外、会变化且需要溯源的知识时，RAG 尤其有用。一次基本的检索生成流程如下：

1. 对用户问题进行必要的改写或补充。
2. 使用 Embedding 模型将问题向量化，并在向量数据库中召回相似片段。
3. 将用户问题和召回片段组装成提示词，发送给 LLM。
4. LLM 根据上下文生成最终答案，并在需要时附上来源。

## 一个零依赖的最小示例

下面的示例使用 Python 标准库完成“关键词召回 + 提示词组装”。它不依赖向量数据库或第三方 API，可以直接运行，用来理解 RAG 的基本数据流。实际项目中，可以将 `retrieve` 替换为向量检索，并将生成的提示词交给具体的 LLM。

将下面的代码保存为 `scripts/mini_rag.py`，在项目根目录运行：

```bash
python scripts/mini_rag.py
```

```python
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    content: str
    source: str


DOCUMENTS = [
    Document("RAG 先从知识库召回相关片段，再将片段交给大语言模型生成答案。", "rag.md"),
    Document("Embedding 模型可以把文本转换为向量，向量数据库据此完成相似度检索。", "embedding.md"),
    Document("切分文档时应尽量保留标题和段落关系，便于检索和结果溯源。", "chunking.md"),
]


def tokenize(text: str) -> set[str]:
    """提取中文字符和英文单词，作为演示用的关键词。"""
    return set(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", text.lower()))


def retrieve(query: str, documents: list[Document], top_k: int = 2) -> list[Document]:
    """按关键词重合数量召回片段；生产环境可替换为向量检索。"""
    query_terms = tokenize(query)
    scored = [
        (len(query_terms & tokenize(document.content)), document)
        for document in documents
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for score, document in scored[:top_k] if score > 0]


def build_prompt(query: str, contexts: list[Document]) -> str:
    context_text = "\n\n".join(
        f"[{document.source}]\n{document.content}" for document in contexts
    )
    return (
        "你是一个严谨的问答助手。请只依据下面的参考资料回答问题；"
        "如果资料不足，请明确说明。\n\n"
        f"参考资料：\n{context_text}\n\n"
        f"问题：{query}\n"
    )


if __name__ == "__main__":
    question = "RAG 是如何利用 Embedding 找到相关内容的？"
    contexts = retrieve(question, DOCUMENTS)

    print("召回片段：")
    for document in contexts:
        print(f"- {document.source}: {document.content}")

    print("\n发送给 LLM 的提示词：\n")
    print(build_prompt(question, contexts))
```

这个示例省略了真正的向量模型、向量数据库和 LLM 调用，但完整展示了 RAG 的关键连接：**问题 → 检索 → 上下文组装 → 生成提示词**。在生产环境中，还需要补充文档解析、切分策略、Embedding 模型、向量数据库、权限控制和答案引用等部分。
