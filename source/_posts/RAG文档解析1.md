---
title: "RAG 文档解析：使用 MinerU 将 PDF 转换为 Markdown"
date: 2026-09-29 18:59:29
topic: rag
tags:
  - RAG
  - MinerU
  - Python
---

在 RAG（Retrieval-Augmented Generation，检索增强生成）流程中，文档解析是检索前最容易被忽略的一步。

PDF、Word、Excel 和扫描件需要先转换成结构清晰、可检索的文本。多栏 PDF 的阅读顺序、表格的行列关系、标题层级以及 OCR 错误，如果在这一步没有处理好，后续更换 Embedding 模型或向量数据库也无法恢复已经丢失的信息。

本文使用 MinerU API，将 PDF 上传、解析、轮询、下载和 Markdown 提取串成一个完整流程。

## 准备 MinerU 配置

先在 [MinerU API 管理页面](https://mineru.net/apiManage/token) 创建 Token，然后在项目根目录创建 `.env` 文件：

```dotenv
MINERU_API_TOKEN=你的_API_Token
MINERU_BASE_URL=https://mineru.net/api/v4
```

安装运行依赖：

```bash
pip install python-dotenv requests
```

在上一篇的项目框架中，本章只新增 `scripts/parse_pdf.py`，并使用 `data/input/` 保存原始 PDF：

```text
RAG/
├── data/
│   └── input/
│       └── report.pdf
└── scripts/
    └── parse_pdf.py
```

## 完整代码

将下面的代码保存为 `scripts/parse_pdf.py`。配置好 Token 并准备 PDF 后，在项目根目录运行。脚本保留 MinerU 返回的 ZIP 文件和解压目录 `data/parsed/report/`；下一章使用其中的 `full.md` 和同级 `images/`。额外复制的 `data/parsed/report.md` 只用于查看文本，相对图片链接仍以原解压目录为准。

```python
from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_config() -> tuple[str, str]:
    """读取并校验 MinerU 配置。"""
    load_dotenv()

    base_url = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4").rstrip("/")
    token = os.getenv("MINERU_API_TOKEN")

    if not token:
        raise ValueError("未找到 MINERU_API_TOKEN，请先在 .env 中配置 API Token")

    return base_url, token


def request_upload_url(
    base_url: str,
    token: str,
    pdf_path: Path,
) -> tuple[str, str]:
    """申请上传地址并返回 (signed_url, batch_id)。"""
    response = requests.post(
        f"{base_url}/file-urls/batch",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={
            "files": [{"name": pdf_path.name}],
            "model_version": "vlm",
        },
        timeout=30,
    )
    response.raise_for_status()

    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"申请上传地址失败：{result}")

    data = result.get("data") or {}
    try:
        return data["file_urls"][0], data["batch_id"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"申请上传地址的响应格式异常：{result}") from exc


def upload_pdf(signed_url: str, pdf_path: Path) -> None:
    """将本地 PDF 上传到 MinerU 返回的临时地址。"""
    with pdf_path.open("rb") as file_obj:
        response = requests.put(
            signed_url,
            data=file_obj,
            timeout=(10, 120),
        )
    response.raise_for_status()


def wait_for_result(
    base_url: str,
    token: str,
    batch_id: str,
    timeout_seconds: int = 600,
    poll_interval: int = 3,
) -> str:
    """轮询解析状态，并返回结果 ZIP 的下载地址。"""
    poll_url = f"{base_url}/extract-results/batch/{batch_id}"
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        response = requests.get(poll_url, headers=headers, timeout=15)
        response.raise_for_status()

        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"查询解析状态失败：{payload}")

        result_items = (payload.get("data") or {}).get("extract_result") or []
        if not result_items:
            raise RuntimeError(f"查询解析状态的响应缺少 extract_result：{payload}")

        item = result_items[0]
        state = item.get("state")

        if state == "done":
            zip_url = item.get("full_zip_url")
            if not zip_url:
                raise RuntimeError(f"解析完成，但响应中没有 full_zip_url：{item}")
            return zip_url

        if state == "failed":
            raise RuntimeError(
                f"MinerU 解析失败：{item.get('err_msg', '未提供错误信息')}"
            )

        logger.info("解析中，当前状态：%s", state)
        time.sleep(poll_interval)

    raise TimeoutError(f"解析任务超过 {timeout_seconds} 秒仍未完成：{batch_id}")


def download_and_extract(
    zip_url: str,
    pdf_path: Path,
    output_dir: Path,
) -> Path:
    """下载 ZIP，解压并提取最终 Markdown 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{pdf_path.stem}_result.zip"

    response = requests.get(zip_url, timeout=(10, 120))
    response.raise_for_status()
    zip_path.write_bytes(response.content)

    extract_dir = output_dir / pdf_path.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)

    md_files = list(extract_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"解压结果中没有找到 Markdown 文件：{extract_dir}")

    source_md = next(
        (path for path in md_files if path.name.lower() == "full.md"),
        md_files[0],
    )
    target_md = output_dir / f"{pdf_path.stem}.md"
    shutil.copy2(source_md, target_md)
    return target_md


def parse_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """执行 PDF 上传、解析、下载和 Markdown 提取。"""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    base_url, token = get_config()
    logger.info("正在申请上传地址：%s", pdf_path)
    signed_url, batch_id = request_upload_url(base_url, token, pdf_path)

    logger.info("正在上传 PDF")
    upload_pdf(signed_url, pdf_path)

    logger.info("开始轮询解析任务：%s", batch_id)
    zip_url = wait_for_result(base_url, token, batch_id)

    logger.info("正在下载并解压解析结果")
    markdown_path = download_and_extract(zip_url, pdf_path, output_dir)
    logger.info("Markdown 已生成：%s", markdown_path)
    return markdown_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 MinerU 将 PDF 转换为 Markdown")
    parser.add_argument(
        "pdf",
        nargs="?",
        default="data/input/report.pdf",
        help="待解析的 PDF 路径，默认：data/input/report.pdf",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parsed",
        help="输出目录，默认：data/parsed",
    )
    args = parser.parse_args()

    parse_pdf(Path(args.pdf).resolve(), Path(args.output_dir).resolve())
```

运行：

```bash
python scripts/parse_pdf.py
```

也可以指定输入文件和输出目录：

```bash
python scripts/parse_pdf.py data/input/paper.pdf --output-dir data/parsed
```

## 运行流程

脚本的处理流程如下：

1. 从 `.env` 读取 MinerU Token。
2. 调用 `/file-urls/batch` 获取临时上传地址和 `batch_id`。
3. 使用临时地址上传本地 PDF。
4. 通过 `/extract-results/batch/{batch_id}` 轮询解析状态。
5. 下载解析结果 ZIP，解压并提取 `full.md`。
6. 将最终文件保存为与 PDF 同名的 Markdown 文件。

至此，PDF 就完成了从上传、解析、轮询、下载到 Markdown 提取的完整流程。接下来可以对生成的 Markdown 进行清洗和切分，再送入 Embedding 模型与向量数据库，构建完整的 RAG 检索链路。

> 提示：请不要把 `.env` 提交到 Git 仓库，并根据实际项目需要调整轮询超时时间和输出目录。
