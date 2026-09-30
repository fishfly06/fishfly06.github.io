---
title: "RAG 文档解析（2）：Markdown 文档中的图片处理"
date: 2026-09-30 20:23:43
tags:
  - RAG
  - Markdown
  - MinIO
  - Python
---

将 PDF 转换为 Markdown 后，正文中的图片通常仍是 `![说明](images/图片名.png)` 这样的相对路径。只把 Markdown 文本切片写入向量数据库，图片文件不会随文本一起保存；检索到相关片段时，原路径也可能已经无法访问。本文沿用上一篇的 MinerU 解析结果：用多模态模型为图片生成简短描述，将图片上传到 MinIO，再把 Markdown 中的本地引用替换为可访问的 URL。

处理顺序是：读取 Markdown、筛选实际引用的图片、生成摘要、上传图片、替换引用，最后另存为 `_new.md`。原文件保持不变。

## 准备目录和配置

上一篇的脚本会把 MinerU 返回的 ZIP 解压到 `output/report/`，其中包含 `full.md` 和 `images/`。本篇应处理解压目录中的 `full.md`，因为上一篇额外复制到 `output/report.md` 的文件与 `images/` 不在同一级目录。

```text
.
├── .env
├── process_images.py
└── output/
    └── report/
        ├── full.md
        └── images/
            ├── figure1.png
            └── figure2.jpg
```

需要 Python 3.10 或更新版本。安装依赖：

```bash
python -m pip install python-dotenv minio langchain-openai
```

在项目根目录创建 `.env`，填写自己的 MinIO 和兼容 OpenAI 接口的视觉模型配置：

```dotenv
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_SECURE=false
MINIO_ACCESS_KEY=你的访问密钥
MINIO_SECRET_KEY=你的私有密钥
MINIO_BUCKET_NAME=knowledge-base-files
MINIO_IMG_DIR=upload-images
MINIO_PUBLIC_BASE_URL=http://127.0.0.1:9000

VL_MODEL=qwen-vl-plus
VL_API_KEY=你的模型API密钥
VL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`MINIO_ENDPOINT` 是 Python 程序连接 MinIO 的地址，不带 `http://`；`MINIO_PUBLIC_BASE_URL` 是读者打开图片时使用的地址，必须包含协议，并且能从读者所在的网络访问。上面的 `127.0.0.1` 仅适用于本机验证。请提前创建**专门存放公开图片**的桶，并在 MinIO 管理端为该桶配置匿名只读权限；否则上传虽然成功，生成的 URL 仍会返回无权限。不要把私有文档放进这个公开桶，也不要把 `.env` 提交到 Git。

下面是原项目中按文档名称存放图片的 MinIO 目录示例，配置时请使用自己的桶和地址：

![MinIO 中按文档分组的图片目录](/assets/image-20260318025329233.png)

## 完整代码

将下面的代码保存为 `process_images.py`：

```python
from __future__ import annotations

import argparse
import base64
import logging
import mimetypes
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote, unquote

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from minio import Minio


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 与 MinerU 生成的 ![描述](images/文件名.png) 格式对应。
IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\((?P<path>(?:\./)?images/[^)\n]+)\)")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"请在 .env 中配置 {name}")
    return value


def find_images(md_content: str, md_path: Path) -> dict[str, tuple[Path, tuple[str, str]]]:
    """只收集 Markdown 实际引用、且位于同级 images 目录的图片。"""
    images_dir = (md_path.parent / "images").resolve()
    images = {}
    for match in IMAGE_RE.finditer(md_content):
        relative_path = unquote(match.group("path"))
        image_path = (md_path.parent / relative_path).resolve()
        if image_path.parent != images_dir or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning("跳过不支持的图片路径：%s", relative_path)
            continue
        if not image_path.is_file():
            logger.warning("图片文件不存在：%s", image_path)
            continue
        before = md_content[max(0, match.start() - 100):match.start()]
        after = md_content[match.end():match.end() + 100]
        images[match.group("path")] = (image_path, (before, after))
    return images


def apply_api_rate_limit(request_times: deque[float], max_requests: int = 10) -> None:
    """限制单次运行在任意 60 秒内的视觉模型请求数。"""
    now = time.monotonic()
    while request_times and now - request_times[0] >= 60:
        request_times.popleft()
    if len(request_times) >= max_requests:
        time.sleep(max(0, 60 - (now - request_times[0])))
        now = time.monotonic()
        while request_times and now - request_times[0] >= 60:
            request_times.popleft()
    request_times.append(now)


def summarize_image(model: ChatOpenAI, image_path: Path, context: tuple[str, str]) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0]
    if not mime_type:
        raise ValueError(f"无法识别图片格式：{image_path}")
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = model.invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                f"这张图片来自 {image_path.parent.parent.name}。"
                f"上文：{context[0]}；下文：{context[1]}。"
                "请用中文简要描述图片的可见内容，供 Markdown 图片替代文本使用。"
            )},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }},
        ],
    }])
    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError(f"模型没有返回图片摘要：{image_path}")
    return " ".join(response.content.split()).replace("[", "\\[").replace("]", "\\]")


def process_images(md_path: Path) -> Path:
    load_dotenv()
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在：{md_path}")

    md_content = md_path.read_text(encoding="utf-8")
    images = find_images(md_content, md_path)
    logger.info("找到 %d 张实际引用的受支持图片", len(images))

    if images:
        model = ChatOpenAI(
            model=required_env("VL_MODEL"),
            api_key=required_env("VL_API_KEY"),
            base_url=required_env("VL_BASE_URL"),
            temperature=0,
        )
        minio_client = Minio(
            required_env("MINIO_ENDPOINT"),
            access_key=required_env("MINIO_ACCESS_KEY"),
            secret_key=required_env("MINIO_SECRET_KEY"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )
        bucket = required_env("MINIO_BUCKET_NAME")
        if not minio_client.bucket_exists(bucket):
            raise ValueError(f"MinIO 桶不存在，请先创建并配置公开只读：{bucket}")
        public_base_url = required_env("MINIO_PUBLIC_BASE_URL").rstrip("/")
        upload_dir = os.getenv("MINIO_IMG_DIR", "upload-images").strip("/")
        document_dir = md_path.parent.name
        request_times = deque()
        replacements = {}

        for original_path, (image_path, context) in images.items():
            apply_api_rate_limit(request_times)
            summary = summarize_image(model, image_path, context)
            object_name = f"{upload_dir}/{document_dir}/{image_path.name}"
            minio_client.fput_object(
                bucket,
                object_name,
                str(image_path),
                content_type=mimetypes.guess_type(image_path.name)[0],
            )
            url = f"{public_base_url}/{bucket}/{quote(object_name, safe='/')}"
            replacements[original_path] = (summary, url)
            logger.info("已处理：%s", image_path.name)

        def replace_image(match: re.Match[str]) -> str:
            info = replacements.get(match.group("path"))
            if info is None:
                return match.group(0)
            summary, url = info
            return f"![{summary}]({url})"

        md_content = IMAGE_RE.sub(replace_image, md_content)

    output_path = md_path.with_name(f"{md_path.stem}_new.md")
    output_path.write_text(md_content, encoding="utf-8")
    logger.info("已保存：%s", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 Markdown 图片生成摘要并上传到 MinIO")
    parser.add_argument("markdown", type=Path, help="MinerU 解压目录中的 full.md")
    args = parser.parse_args()
    process_images(args.markdown.resolve())
```

运行：

```bash
python process_images.py output/report/full.md
```

处理后得到 `output/report/full_new.md`。同一张图片在 Markdown 中出现多次时，只生成一次摘要、上传一次，但会替换所有引用；`images/` 中未被引用的文件不会上传。模型调用或上传失败时，程序会报错且不会写出新的 Markdown；已上传的图片可以在排查后重新运行，重名对象会被覆盖。这里没有在运行前清空 MinIO 目录，避免误删已有对象。

> 当前示例针对 MinerU 常见的内联图片语法 `![描述](images/文件名.png)`，支持 JPG、PNG 和 WebP；GIF、BMP、带 Markdown 标题或括号等复杂路径的图片引用需要按实际输出格式扩展。最终 URL 是否能从检索端访问，还取决于 MinIO 的网络地址和桶的读取权限。
