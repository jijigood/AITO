# Task D4 — RAGHit Dataclass 与 `retrieve()` 接口桩

**粒度**: 极小（仅新增 1 个 dataclass + 1 个方法签名 + 1 处重命名）  
**优先级**: 🔴 最高——Owner A 需要此接口才能对接  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: 无  
**后续依赖**: D1a、D1b、D2a、D2b、D3a、D3b 全部依赖本任务

---

## 背景

Owner A 的 `build_sft_v3.py` 需要调用 `rag.retrieve(sf, k=6)` 并消费返回的对象。当前文件中：
- 没有统一的返回类型，消费方只能解包 `case["_similarity"]` 等私有字段
- 现有 `retrieve()` 签名是 `retrieve(query_feature_vector, family_name, top_k)`，与 Owner A 期望的签名不匹配
- `main_v3.py` 仍在调用 `retrieve_and_format(query_feature_vector, family_name, top_k)`，因此 **D4 不能破坏旧路径**

本任务**只做接口定义**，`retrieve()` 实现先返回空列表，让 Owner A 可以立刻对接而不等待后续实现。旧的向量检索路径保留在 `_retrieve_raw()` / `retrieve_and_format()` 中，直到后续任务完成再逐步迁移。

---

## 需要修改的内容

### 变更 1：在文件顶部新增 `RAGHit` dataclass

位置：在 `import` 区块之后、`class StructuredCaseRAG` 定义之前，插入以下代码：

```python
from dataclasses import dataclass

@dataclass
class RAGHit:
    scenario_id: str
    features: Dict[str, float]   # 归一化特征向量字典，供 prompt 渲染
    gt_c_code: str               # 正确答案 C-code，如 "C6"
    gt_action_type: str          # lift_tilt / insufficient_data / add_neighbor / ...
    similarity: float            # 余弦相似度 [0, 1]
    prompt_snippet: str          # 两行可读摘要，供 _build_f1_prompt 拼入 prompt
    is_confusion_pair: bool = False  # D3b 注入时标记为 True
```

确保 `Dict` 已在顶部 `from typing import ...` 中导入。

### 变更 2：将现有 `retrieve()` 重命名为 `_retrieve_raw()`

在 `StructuredCaseRAG` 类中，将现有方法：

```python
def retrieve(self, query_feature_vector, family_name="F1", top_k=6):
```

改名为：

```python
def _retrieve_raw(self, query_feature_vector, family_name="F1", top_k=6):
```

方法体不做任何修改，只改名。

### 变更 3：更新 `retrieve_and_format()` 内部调用

`retrieve_and_format()` 方法内部原先调用 `self.retrieve(...)`，改为调用 `self._retrieve_raw(...)`，其余内容不变。

> 这一步是为了保证 `main_v3.py` 在 D4 合入后仍然可运行；D4 **不是** `main_v3.py` 迁移任务。

### 变更 4：新增公开的 `retrieve()` 接口桩

在 `StructuredCaseRAG` 类中新增方法（**实现先返回 `[]`**）：

```python
def retrieve(
    self,
    sf,
    k: int = 6,
    stratify_by_action: bool = True,
    include_confusion_pairs: bool = True,
    family_name: str = "F1",
) -> List["RAGHit"]:
    """
    接口已冻结。实现由 D2b 任务填充。
    当前返回空列表，使 Owner A 可以立即对接签名。
    """
    return []
```

⚠️ `retrieve()` 的参数名称和顺序从此刻起**不得再改变**。  
⚠️ `retrieve_and_format()` 的参数名称和顺序在本任务中也**不得改变**。

---

## 验收检查（无需 pytest，手动验证）

```bash
# 1. 导入不报错
python -c "from rag_retriever_v2 import StructuredCaseRAG, RAGHit; print('OK')"

# 2. RAGHit 字段正确
python -c "
from rag_retriever_v2 import RAGHit
h = RAGHit(
    scenario_id='s1', features={}, gt_c_code='C1',
    gt_action_type='lift_tilt', similarity=0.9, prompt_snippet='test'
)
assert h.is_confusion_pair == False
print('RAGHit OK')
"

# 3. retrieve_and_format() 仍可调用（不报 AttributeError）
python -c "
from rag_retriever_v2 import StructuredCaseRAG
r = StructuredCaseRAG.__new__(StructuredCaseRAG)
# 确认 _retrieve_raw 存在且 retrieve_and_format 不报错
assert hasattr(r, '_retrieve_raw'), '_retrieve_raw missing'
print('rename OK')
"

# 4. 新 retrieve() 返回空列表
python -c "
from rag_retriever_v2 import StructuredCaseRAG
r = StructuredCaseRAG.__new__(StructuredCaseRAG)
result = r.retrieve(None, k=6)
assert result == []
print('stub OK')
"
```

---

## 禁止事项

- 不得修改 `main_v3.py`（它调用 `retrieve_and_format()`，需要继续工作）
- 不得修改 `_retrieve_raw()` 的方法体（只改名，不改逻辑）
- 新 `retrieve()` 此阶段只能返回 `[]`，**不要**实现任何检索逻辑（留给 D2b）
- `RAGHit` 字段不得增减（后续任务依赖此结构）
- 不得修改 `retrieve_and_format()` 的函数签名
