# Task D3b — 混淆对检索集成：`_get_confusion_pairs` + `retrieve()` 触发条件

**粒度**: 小（1 个辅助方法 + `retrieve()` 末尾替换 `pass`）  
**优先级**: 🟠 高  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: D2b + D3a 完成  
**后续依赖**: 无

---

## 背景

D2b 已经让 `retrieve(sf, ...)` 返回正常 top-k；D3a 已经把 `lift_tilt ↔ insufficient_data` 的对立样本关系缓存成 `_confusion_index`。  
本任务是在**真正模糊**的场景里，把这两个类别的边界样本追加到结果末尾，且不挤占正常的 k 个命中。

---

## 触发原则

- 首选 `sf.top_issue()` 判断是否属于混淆触发区
- 允许的触发值：`coverage_hole`、`backhaul_server`
- 如果传入的 `sf` 没有 `top_issue()` 方法，再退回到向量维度 `query_vec[8:13]` 的 helper

这样做是为了兼容当前仓库状态：13 维向量本身并不显式编码 `backhaul_server`，但真实 `ScenarioFeatures.top_issue()` 会返回它。

---

## 需要修改的内容

### 变更 1：模块级常量

```python
_CONFUSION_TRIGGER_ISSUES = frozenset({"coverage_hole", "backhaul_server"})

_ISSUE_DIM_NAMES = [
    "late_handover",
    "missing_neighbor",
    "interference",
    "coverage_hole",
    "inter_freq_load",
]

def _get_top_issue_from_query_vec(query_vec: np.ndarray) -> str:
    issue_scores = query_vec[8:13]
    if len(issue_scores) < 5:
        return "unknown"
    return _ISSUE_DIM_NAMES[int(np.argmax(issue_scores))]
```

### 变更 2：新增 `_get_confusion_pairs`

```python
def _get_confusion_pairs(self, query_vec: np.ndarray, family_idxs: List[int]) -> List[RAGHit]:
    """
    Return up to 2 confusion exemplars, appended after the normal top-k hits.
    Each returned hit represents a seed case from one confusing class; its
    prompt_snippet can be enriched with one opposite-class sid from
    _confusion_index for boundary context.
    """
    family_set = set(family_idxs)
    out: List[RAGHit] = []

    for ctype in ("lift_tilt", "insufficient_data"):
        bucket_idxs = [gi for gi in self._action_indices.get(ctype, []) if gi in family_set]
        if not bucket_idxs:
            continue
        sims = self._cosine_sim_to_subset(query_vec, bucket_idxs)
        best_local = int(np.argmax(sims))
        gi = bucket_idxs[best_local]
        case = dict(self.cases[gi])
        hit = self._case_to_raghit(case, float(sims[best_local]), is_confusion=True)

        seed_sid = case["scenario_id"]
        paired = self._confusion_index.get(ctype, {}).get(seed_sid, [])
        if paired:
            hit.prompt_snippet = f"{hit.prompt_snippet} contrast={paired[0]}"[:120]
        out.append(hit)

    return out
```

### 变更 3：替换 `retrieve()` 中的占位 `pass`

```python
if include_confusion_pairs:
    if hasattr(sf, "top_issue") and callable(sf.top_issue):
        top_issue = sf.top_issue()[0]
    else:
        top_issue = _get_top_issue_from_query_vec(query_vec)
    if top_issue in _CONFUSION_TRIGGER_ISSUES:
        result.extend(self._get_confusion_pairs(query_vec, family_idxs))
```

---

## 验收检查

优先使用合成对象测试：

```python
import numpy as np
from rag_retriever_v2 import StructuredCaseRAG

class MockSF:
    def top_issue(self):
        return ("coverage_hole", 0.9)

rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
rag.feature_matrix = np.eye(13, dtype=np.float32)
rag.family_indices = {"F1": [0, 1]}
rag._action_indices = {"lift_tilt": [0], "insufficient_data": [1]}
rag._confusion_index = {"lift_tilt": {"s0": ["s1"]}, "insufficient_data": {"s1": ["s0"]}}
rag.cases = [
    {"scenario_id": "s0", "family": "F1", "action_type": "lift_tilt", "answer": "C0", "feature_vector": rag.feature_matrix[0]},
    {"scenario_id": "s1", "family": "F1", "action_type": "insufficient_data", "answer": "C1", "feature_vector": rag.feature_matrix[1]},
]
rag._cosine_sim_to_subset = StructuredCaseRAG._cosine_sim_to_subset.__get__(rag, StructuredCaseRAG)
rag._case_to_raghit = StructuredCaseRAG._case_to_raghit.__get__(rag, StructuredCaseRAG)

pairs = rag._get_confusion_pairs(np.ones(13, dtype=np.float32), [0, 1])
assert 1 <= len(pairs) <= 2
assert all(h.is_confusion_pair for h in pairs)
```

---

## 禁止事项

- confusion pairs 不得挤占正常 `k` 个命中
- 不得直接重算对立类最近邻并绕过 `_confusion_index`
- 不得修改 `main_v3.py`
