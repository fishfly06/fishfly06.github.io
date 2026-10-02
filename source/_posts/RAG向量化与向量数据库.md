---
title: RAG向量化与向量数据库
date: 2026-10-02 11:37:22
topic: rag
tags:
---

向量化就是将人类可读的文本切片转化为机器可计算的向量表示。本章节我们采用 BGE-M3 模型，同时生成稠密向量（Dense）和稀疏向量（Sparse），为“混合检索”提供底层数据支持。

## 本章在项目中的位置

前一章已经生成 `data/chunks/report_chunks.json`。本章读取这个文件，为每个 Chunk 增加 Dense 和 Sparse 向量，再写入 Milvus。模型文件放在 `models/bge-m3/`，Embedding 配置放在 `config/embedding_config.py`，Milvus 配置放在 `config/milvus_config.py`。

```text
RAG/
├── data/
│   └── chunks/
│       └── report_chunks.json   # 本章输入
├── models/
│   └── bge-m3/                  # 本地模型
├── config/
│   ├── embedding_config.py
│   └── milvus_config.py
└── scripts/
    └── embed_and_insert.py      # 本章最终脚本（尚在完善）
```

上面的目录是本章的目标结构。本文仍在编写，下面保留的项目片段尚待迁入这套结构，包括旧模型路径、商品字段以及配置、日志和状态对象的依赖；目前不能拼接为独立脚本运行。完成时将以 `data/chunks/report_chunks.json` 为统一输入，在项目根目录运行入口。

### 本地下载BGE-M3模型

```python
from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到当前目录下的 models/bge-m3 文件夹
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache')
print(f"模型已下载到: {model_dir}")
```

创建BGE-M3实例

```python
# utils/embedding_utils.py

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from config.embedding_config import embedding_config


# BGE-M3 模型本地路径
BGE_M3_PATH=D:\ai_models\modelscope_cache\models\BAAI--bge-m3/snapshots/master
# BGE-M3 模型名称
BGE_M3=BAAI/bge-m3
# 嵌入模型运行设备
BGE_DEVICE=cuda:0
# 是否使用半精度（True/False）
BGE_FP16=True

# 模型单例对象，避免重复初始化
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    # 从环境变量加载配置
    model_name = BGE_M3_PATH
    device = BGE_DEVICE
    use_fp16 = BGE_FP16

    # 如果模型没有被提前下载，会自动下载
    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )
    return _bge_m3_ef

def generate_embeddings(texts ):
    """
    为文本生成向量嵌入
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)
    processed_sparse = []
    for i in range(len(texts)):
        sparse_indices = embeddings["sparse"].indices[embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]].tolist()
        sparse_data = embeddings["sparse"].data[embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i+1]].tolist()
        sparse_dict = {k: v for k, v in zip(sparse_indices,sparse_data)}
        processed_sparse.append(sparse_dict)

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse
    }
```

### 向量化实现思路

1.  **双路编码**: 利用 BGE-M3 的特性，一次推理同时产出语义向量（Dense，捕获语义相似度）和词汇向量（Sparse，捕获关键词匹配），兼顾语义理解和精确匹配。
2.  **批处理优化**: 针对大量切片，采用 Batch 处理模式调用模型，大幅提升 GPU/CPU 的计算利用率。

#### 批量生成向量

分批次为文本生成稠密和稀疏向量，并进行归一化处理。

```python
"""
核心逻辑：
1. 分批处理：避免一次性处理过多数据导致显存溢出（OOM）。
2. 文本构造：将 item_name 和 content 拼接，增强语义（商品名作为核心特征前置）。
3. 向量生成：调用模型批量生成 Dense（稠密）和 Sparse（稀疏）向量。
"""


# 初始化空列表，存储最终带向量的文本切片

output_data = []

# 设置批次大小（每批处理5条，可根据显存/性能调整：显存大则调大，反之调小）

batch_size = 5  # 设置批次大小，可以根据显存大小进行调整！


# 按批次遍历文本切片：range(起始, 终止, 步长) → 0,5,10... 分批处理
for i in range(0, len(chunks), batch_size):
    
    batch_texts = chunks[i:i + batch_size]

    input_texts = []

    for doc in batch_texts:

        item_name = doc["item_name"]

        content = doc["content"]

        input_texts.append(f"{item_name}\n{content}" if item_name else content)


    docs_embeddings = generate_embeddings(input_texts)

    for j, doc in enumerate(batch_texts):

        item = doc.copy()

        item["dense_vector"] = docs_embeddings["dense"][j]

     	item["sparse_vector"] = docs_embeddings["sparse"][j]

        output_data.append(item)

    
    logger.info(f"成功获取第 {i + 1}-{min(i + len(batch_texts), len(chunks))} 项的嵌入。")
```

向量入库是数据加载流程的终点，负责将处理好的结构化数据（切片内容、元数据、向量）持久化存储到向量数据库中，构建可供即时查询的索引。

### 实现思路

1.  **幂等性设计**: 在插入新数据前，根据 `item_name` 或文件 ID 清理旧数据，防止重复导入导致的数据污染。
2.  **Schema 适配**: 严格按照 Milvus 集合的 Schema 定义（主键、Dense字段、Sparse字段、JSON元数据字段）组织数据，确保插入成功率。
3.  **混合索引构建**: 确保存入的数据能够支持 Milvus 的 Hybrid Search（Dense + Sparse 加权），最大化检索效果。

```python
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from config.milvus_config import milvus_config
from tool.logger import logger

milvus_uri = milvus_config.milvus_url

_milvus_client = None


def validate_private_schema(client, collection_name):
    """
    确认向量集合包含知识库、文档及导入版本字段
    :param client: 数据库或存储客户端
    :param collection_name: 待检查的 Milvus 集合名
    """
    fields = {f["name"]: f for f in client.describe_collection(collection_name=collection_name)["fields"]}
    if not {"kb_id", "document_id", "import_task"} <= fields.keys():
        raise RuntimeError("向量集合缺少私有知识库字段，请使用新的 PRIVATE_* 集合名称。")


def cleanup_import_vectors(task, *, keep_current=False):
    """
    在当前知识库和文档范围内清理指定导入版本的向量
    :param task: 含知识库、文档和任务 ID 的记录
    :param keep_current: True 删除旧版本；False 仅删除当前任务产生的向量
    """
    from utils.knowledge_access import kb_filter
    import json
    client = get_milvus_client()
    expr = kb_filter(task["kb_id"], document_id=task["document_id"])
    expr += f' and import_task {"!=" if keep_current else "=="} {json.dumps(task["task_id"])}'
    for name in (milvus_config.chunks_collection, milvus_config.item_name_collection):
        if client.has_collection(name):
            validate_private_schema(client, name)
            client.delete(collection_name=name, filter=expr)

def get_milvus_client():
    """
    延迟创建并复用 Milvus 客户端
    :return: 带请求超时设置的 Milvus 客户端
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    _milvus_client = MilvusClient(milvus_uri, timeout=8)
    return _milvus_client

def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用：
        转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数：
        value: 需要转义的原始字符串
    返回：
        str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value

```

#### 步骤 1: 检查输入

**功能**: 验证 `chunks` 是否存在，并提取 `dense_vector` 维度。

```python
# 校验1：chunks非空
chunks_json_data = state.get("chunks")

if not chunks:
    raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

if not isinstance(chunks, list):
    raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

# 校验2：切片包含dense_vector字段
first_chunk = chunks[0]
if 'dense_vector' not in first_chunk:
    raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

# 校验3：切片包含 sparse_vector 字段
if 'sparse_vector' not in first_chunk:
    raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

# 提取向量维度
vector_dimension = len(first_chunk['dense_vector'])
    
```

#### 步骤 2: 准备集合 

**功能**: 获取 Milvus 客户端，如果集合不存在则创建。

```python
# 1. 获取milvus客户端对象
milvus_client = get_milvus_client()
if not milvus_client:
    logger.error("Milvus 连接失败")
    raise MilvusError("Milvus 连接失败")
# 2. 集合不存在则创建
collections_name = milvus_config.chunks_collection
if not milvus_client.has_collection(collections_name):
    _create_chunks_collection(collections_name, milvus_client, vector_dimension)


def _create_chunks_collection(collections_name, milvus_client, vector_dimension):

    # 1. 创建schem
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    # 2. 创建列
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
    schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

    # 3. 创建索引
    index_params = milvus_client.prepare_index_params()
    # 稠密向量索引：AUTOINDEX自动选最优索引类型+余弦相似度（语义检索常用）
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+内积（IP），适配稀疏向量检索
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
    )

    # 创建集合
    milvus_client.create_collection(
        collection_name=collections_name,
        schema=schema,
        index_params=index_params
     
```

**IVF_FLAT 和 AUTOINDEX** 

1. IVF_FLAT - 手动指定索引类型

特点：
✅ 明确控制：你知道用的是什么索引
✅ 可 tuning：可以调整 nlist 等参数优化性能
❌ 需要经验：要自己判断适合什么索引
❌ 固定不变：数据量变化后可能不是最优

2. AUTOINDEX - 自动选择最优索引

特点：
✅ 智能选择：Milvus 根据数据量、维度自动选最优
✅ 自适应：数据量变化时自动升级索引策略
✅ 省心：不需要懂索引原理也能用好
❌ 黑盒：你不知道具体用的是什么
❌ 不可控：无法手动调优

3. Milvus 的 AUTOINDEX 如何选择？

- Milvus 会根据以下因素自动选择：

```python
# 伪代码展示 AUTOINDEX 的决策逻辑
if 数据量 < 10 万:
    使用 FLAT 索引  # 精确搜索，速度也够快
elif 数据量 < 1000 万:
    使用 IVF_FLAT  # 近似搜索，精度高速度快
elif 数据量 < 1 亿:
    使用 IVF_PQ  # 压缩存储，节省内存
else:
    使用 HNSW  # 超大规模最优
```

4. 哪个更好？

- 推荐 AUTOINDEX 的场景 ✅
  - 快速原型开发：先跑通业务，再优化
  - 数据量不确定：不知道未来会有多少数据
  - 团队无专家：没有人专门研究向量索引
  - 中小规模：数据量 < 1000 万

- 推荐 IVF_FLAT 的场景 ✅
  - 生产环境优化：已经知道数据特征
  - 性能敏感：需要极致优化检索速度
  - 有专业团队：有人能 tuning 参数
  - 特殊需求：需要精确控制内存/速度比

#### 步骤 3: 清理旧数据 

**功能**: 根据 `item_name` 删除已存在的切片，确保幂等性。

```python
"""
幂等清理
基于每个片段的file_title进行旧数据的清理
"""
# 1. 获取查询条件
file_title = chunks_json_data[0].get("file_title")

# 2. 执行幂等清理
_clear_chunks_by_file_title(client, file_title)

def _clear_chunks_by_file_title(client, file_title):

    try:
        file_title = escape_milvus_string(file_title)
        client.delete(
            collection_name=milvus_config.chunks_collection,
            filter=f"file_title=='{file_title}'")
    except Exception as e:
        logger.error(f"Milvus 数据删除失败: {str(e)}")
        raise MilvusError(f"Milvus 数据删除失败: {str(e)}")
```

#### 步骤4：批量插入切片数据到Milvus+主键回填

```python
"""
核心逻辑：
    1. 批量插入数据：提升入库效率，减少Milvus连接次数
    2. 回填chunk_id：将Milvus生成的自增主键回填到切片，供下游业务使用
"""

# 1. 预处理数据：移除手动chunk_id，避免与Milvus自增主键冲突
data_to_insert = []
for item in chunks_json_data:
    item_copy = item.copy()

    # 补充 part 字段
    if "part" not in item_copy:
        item_copy["part"] = 0

    # 添加到待插入列表
    data_to_insert.append(item_copy)

# 2. 执行批量插入
insert_result = client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
insert_count = insert_result.get('insert_count', 0)
     
# 3. 主键回填：将Milvus生成的chunk_id回填到原始切片
inserted_ids = insert_result.get('ids', [])
if inserted_ids:
    for idx, item in enumerate(chunks_json_data):
        item['chunk_id'] = str(inserted_i
```
