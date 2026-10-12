# Task D5a — `skills/data_tools.py`：单例管理 + `duplicate_score` + `outlier_score`

**粒度**: 小（单例框架 + 2 个函数）  
**优先级**: 🟠 高  
**负责文件**: `skills/data_tools.py`  
**前置依赖**: D1b 完成（`build_index()` 可稳定加载缓存，`_action_indices` 已存在）  
**后续依赖**: D5b

---

## 背景

`skills/data_tools.py` 目前还是 stub。D5a 先把**可独立使用**的两类统计分数做好：

- `duplicate_score`：和其他样本有多像
- `outlier_score`：在自己的动作类里有多离群

这两项都只依赖 RAG 索引，不依赖 D2/D3 的新检索主流程，因此可以作为单独分支推进。

---

## 需要修改的内容

### 变更 1：替换头部 import 与单例框架

目标：

- 懒加载 `StructuredCaseRAG`
- 允许测试注入 fake rag
- 允许批量函数复用已加载单例，避免反复建索引

建议结构：

```python
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_rag_singleton = None
_DEFAULT_TRAIN_JSON = os.path.join(
    os.path.dirname(__file__), "..", "data", "Phase_1", "train.json"
)

def _get_rag(train_json_path: Optional[str] = None, _rag=None):
    if _rag is not None:
        return _rag
    global _rag_singleton
    if _rag_singleton is not None:
        return _rag_singleton
    from rag_retriever_v2 import StructuredCaseRAG
    path = train_json_path or os.environ.get("RAG_TRAIN_JSON", _DEFAULT_TRAIN_JSON)
    rag = StructuredCaseRAG(train_json_path=path)
    rag.build_index()
    _rag_singleton = rag
    return _rag_singleton

def _reset_rag_singleton():
    global _rag_singleton
    _rag_singleton = None

def _inject_rag_singleton(rag_instance):
    global _rag_singleton
    _rag_singleton = rag_instance

def _sid_to_global_idx(r, scenario_id: str) -> Optional[int]:
    for idx, case in enumerate(r.cases):
        if case["scenario_id"] == scenario_id:
            return idx
    return None
```

### 变更 2：实现 `duplicate_score`

```python
def duplicate_score(scenario_id: str, cache: Optional[Dict] = None, _rag=None) -> float:
    """
    Max cosine similarity to any *other* sample in the same family.
    """
    if cache is not None and scenario_id in cache:
        return float(cache[scenario_id])

    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    family = r.cases[global_idx].get("family", "")
    family_idxs = r.family_indices.get(family, [])
    if len(family_idxs) < 2:
        return 0.0

    query_vec = r.feature_matrix[global_idx].astype(np.float32)
    sub = r.feature_matrix[family_idxs].astype(np.float32)

    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
    sims = ((sub / norms) @ q_norm.reshape(-1, 1)).flatten()

    try:
        local_self = family_idxs.index(global_idx)
        sims[local_self] = -1.0
    except ValueError:
        pass

    return float(np.max(sims))
```

### 变更 3：实现 `outlier_score`

```python
def outlier_score(scenario_id: str, _rag=None) -> float:
    """
    Euclidean distance to the centroid of the same action_type bucket.
    """
    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    action_type = r.cases[global_idx].get("action_type", "unknown")
    action_idxs = r._action_indices.get(action_type, [])
    if not action_idxs:
        return 0.0

    centroid = r.feature_matrix[action_idxs].mean(axis=0)
    vec = r.feature_matrix[global_idx]
    return float(np.linalg.norm(vec - centroid))
```

---

## 验收检查

使用注入 fake rag：

```python
import numpy as np
from skills.data_tools import (
    _inject_rag_singleton,
    _reset_rag_singleton,
    duplicate_score,
    outlier_score,
)

class FakeRAG:
    cases = [
        {"scenario_id": "s0", "family": "F1", "action_type": "lift_tilt"},
        {"scenario_id": "s1", "family": "F1", "action_type": "lift_tilt"},
        {"scenario_id": "s2", "family": "F1", "action_type": "lift_tilt"},
        {"scenario_id": "s3", "family": "F1", "action_type": "check_server"},
        {"scenario_id": "s4", "family": "F1", "action_type": "check_server"},
    ]
    family_indices = {"F1": [0, 1, 2, 3, 4]}
    _action_indices = {"lift_tilt": [0, 1, 2], "check_server": [3, 4]}
    feature_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.9],
    ], dtype=np.float32)

_reset_rag_singleton()
_inject_rag_singleton(FakeRAG())

assert duplicate_score("s1") > 0.9
assert duplicate_score("missing") == 0.0
assert outlier_score("s2") > 0.0

_reset_rag_singleton()
```

---

## 禁止事项

- 对外公开签名仍然必须是 `duplicate_score(scenario_id, cache=None)` / `outlier_score(scenario_id)`
- 允许内部偷偷加 `_rag=None` 优化参数，但不要把它写进用户文档
- 不得在函数内部发 HTTP 请求
