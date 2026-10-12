# Task D3a — 混淆对索引构建：`_build_confusion_index` + `build_index()` 集成

**粒度**: 小（1 个纯计算方法 + 缓存读写挂接）  
**优先级**: 🟠 高  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: D1b 完成（`feature_matrix`、`_action_indices`、缓存目录已存在）  
**后续依赖**: D3b

---

## 背景

`lift_tilt` 和 `insufficient_data` 是 F1 里最容易混淆的一组。D3a 的目标不是改 `retrieve()`，而是提前把“一个样本最像哪几个对立类别样本”算好并缓存下来，让 D3b 只做轻量拼接。

---

## 需要修改的内容

### 变更 1：在 `__init__` 中初始化 `_confusion_index`

```python
self._confusion_index: Dict[str, Dict[str, List[str]]] = {}
```

### 变更 2：新增 `_build_confusion_index`

```python
def _build_confusion_index(self) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a seed -> opposite-class-neighbor mapping for the two most confusing
    action buckets: lift_tilt and insufficient_data.

    Returns:
        {
            "lift_tilt": {seed_sid: [opp_sid1, opp_sid2, ...]},
            "insufficient_data": {seed_sid: [opp_sid1, opp_sid2, ...]},
        }
    """
    result = {"lift_tilt": {}, "insufficient_data": {}}

    lt_idxs = self._action_indices.get("lift_tilt", [])
    id_idxs = self._action_indices.get("insufficient_data", [])
    if not lt_idxs or not id_idxs:
        return result

    lt_matrix = self.feature_matrix[lt_idxs].astype(np.float32)
    id_matrix = self.feature_matrix[id_idxs].astype(np.float32)

    lt_norms = np.linalg.norm(lt_matrix, axis=1, keepdims=True) + 1e-8
    id_norms = np.linalg.norm(id_matrix, axis=1, keepdims=True) + 1e-8
    sims = (lt_matrix / lt_norms) @ (id_matrix / id_norms).T

    for row_i, row in enumerate(sims):
        top2 = np.argsort(-row)[:2]
        seed_sid = self.cases[lt_idxs[row_i]]["scenario_id"]
        result["lift_tilt"][seed_sid] = [self.cases[id_idxs[j]]["scenario_id"] for j in top2]

    for row_i, row in enumerate(sims.T):
        top2 = np.argsort(-row)[:2]
        seed_sid = self.cases[id_idxs[row_i]]["scenario_id"]
        result["insufficient_data"][seed_sid] = [self.cases[lt_idxs[j]]["scenario_id"] for j in top2]

    return result
```

### 变更 3：在缓存命中分支中加载 `confusion_index.pkl`

```python
confuse_path = os.path.join(self.CACHE_DIR, "confusion_index.pkl")
if os.path.exists(confuse_path):
    with open(confuse_path, "rb") as f:
        self._confusion_index = pickle.load(f)
else:
    self._confusion_index = {}
```

### 变更 4：在重建分支末尾写入 `confusion_index.pkl`

注意顺序：

1. 先写 `cases.pkl` / `matrix.npy`
2. 再写 `confusion_index.pkl`
3. 再写可选 FAISS
4. **最后**写 `data_hash.txt`

示意：

```python
self._confusion_index = self._build_confusion_index()
confuse_path = os.path.join(self.CACHE_DIR, "confusion_index.pkl")
with open(confuse_path, "wb") as f:
    pickle.dump(self._confusion_index, f)
```

---

## 验收检查

优先使用合成索引：

```python
import numpy as np
from rag_retriever_v2 import StructuredCaseRAG

rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
rag.cases = [
    {"scenario_id": "lt0"},
    {"scenario_id": "lt1"},
    {"scenario_id": "id0"},
    {"scenario_id": "id1"},
]
rag._action_indices = {"lift_tilt": [0, 1], "insufficient_data": [2, 3]}
rag.feature_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [1.0, 0.0, 0.1],
    [0.0, 1.0, 0.0],
], dtype=np.float32)

ci = rag._build_confusion_index()
assert "lift_tilt" in ci and "insufficient_data" in ci
assert "lt0" in ci["lift_tilt"]
assert "id0" in ci["insufficient_data"]
```

---

## 禁止事项

- 不得在 `retrieve()` 中现算 confusion index
- 不得修改 `main_v3.py`
