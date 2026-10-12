# Task D2b — 分层检索主流程：`_flat_retrieve` / `_stratified_retrieve` / `retrieve()`

**粒度**: 中（3 个方法，共同组成检索主流程）  
**优先级**: 🔴 高——这是 `retrieve(sf, ...)` 空桩的首次真正实现  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: D2a 完成  
**后续依赖**: D3b（在 `retrieve()` 末尾追加混淆对）

---

## 背景

当前状态：

- 旧路径：`_retrieve_raw(query_feature_vector, family_name, top_k)` 仍服务 `main_v3.py`
- 新路径：`retrieve(sf, k=6, ...)` 已存在，但暂时返回 `[]`

本任务只负责把**新路径**做起来，不迁移 `main_v3.py`，也不改 `retrieve_and_format()`。

---

## 模块级常量

在 `class StructuredCaseRAG` 定义之前新增：

```python
_F1_ACTION_BUCKETS = [
    "lift_tilt",
    "insufficient_data",
    "add_neighbor",
    "pdcch_2sym",
    "check_server",
    "decrease_a3",
]
```

---

## 需要新增/修改的方法

### 方法 1：`_flat_retrieve`

```python
def _flat_retrieve(self, query_vec: np.ndarray, family_idxs: List[int], k: int) -> List[dict]:
    """Plain top-k cosine retrieval within the given family bucket."""
    sims = self._cosine_sim_to_subset(query_vec, family_idxs)
    top_local = np.argsort(-sims)[:k]
    out = []
    for li in top_local:
        gi = family_idxs[li]
        case = dict(self.cases[gi])
        case["_similarity"] = float(sims[li])
        out.append(case)
    return out
```

### 方法 2：`_stratified_retrieve`

```python
def _stratified_retrieve(self, query_vec: np.ndarray, family_idxs: List[int], k: int) -> List[dict]:
    """
    Try to draw at least one example from each major F1 action bucket, then
    backfill by overall similarity.
    """
    family_set = set(family_idxs)
    quota = max(1, k // max(len(_F1_ACTION_BUCKETS), 1))

    collected: List[dict] = []
    used: set[int] = set()

    for bucket in _F1_ACTION_BUCKETS:
        bucket_idxs = [gi for gi in self._action_indices.get(bucket, []) if gi in family_set]
        if not bucket_idxs:
            continue
        sims = self._cosine_sim_to_subset(query_vec, bucket_idxs)
        top_local = np.argsort(-sims)[:quota]
        for li in top_local:
            gi = bucket_idxs[li]
            if gi in used:
                continue
            case = dict(self.cases[gi])
            case["_similarity"] = float(sims[li])
            collected.append(case)
            used.add(gi)

    remaining = k - len(collected)
    if remaining > 0:
        all_sims = self._cosine_sim_to_subset(query_vec, family_idxs)
        order = np.argsort(-all_sims)
        for li in order:
            gi = family_idxs[li]
            if gi in used:
                continue
            case = dict(self.cases[gi])
            case["_similarity"] = float(all_sims[li])
            collected.append(case)
            used.add(gi)
            remaining -= 1
            if remaining == 0:
                break

    return collected
```

### 方法 3：实现新的 `retrieve()`

```python
def retrieve(
    self,
    sf,
    k: int = 6,
    stratify_by_action: bool = True,
    include_confusion_pairs: bool = True,
    family_name: str = "F1",
) -> List[RAGHit]:
    if self.feature_matrix is None:
        raise RuntimeError("Index not built. Call build_index() first.")

    query_vec = self._sf_to_feature_vec(sf)
    family_idxs = self.family_indices.get(family_name, list(range(len(self.cases))))

    if stratify_by_action:
        hits = self._stratified_retrieve(query_vec, family_idxs, k)
    else:
        hits = self._flat_retrieve(query_vec, family_idxs, k)

    hits = self._deduplicate(hits, query_vec, threshold=0.98)
    hits.sort(key=lambda h: -h["_similarity"])
    hits = hits[:k]

    result = [self._case_to_raghit(h, h["_similarity"]) for h in hits]

    # D3b 之后这里才会真正追加 confusion pairs。
    # 在 D2b 阶段，include_confusion_pairs 仅保留参数，不做任何事。
    if include_confusion_pairs:
        pass

    return result
```

---

## 验收检查

优先使用合成索引测试；真实 `train.json` 只做 smoke check。

```python
import numpy as np
from rag_retriever_v2 import RAGHit, StructuredCaseRAG

class MockSF:
    class throughput:
        worst_throughput_mbps = 50.0
        throughput_drop_magnitude = 30.0
    class mobility:
        serving_rsrp_dbm = -90.0
        serving_sinr_db = 5.0
        neighbor_advantage_db = 3.0
        a3_margin_db = 0.0
        visible_neighbor_pcis = [1, 2, 3]
    class inter_freq:
        max_dl_prb_util = 80.0
    score_late_handover = 0.1
    score_missing_neighbor = 0.05
    score_interference = 0.2
    score_coverage_hole = 0.7
    score_inter_freq_load = 0.15

rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
rag.feature_matrix = np.eye(13, dtype=np.float32)
rag.family_indices = {"F1": [0, 1, 2, 3, 4, 5]}
rag._action_indices = {
    "lift_tilt": [0],
    "insufficient_data": [1],
    "add_neighbor": [2],
    "pdcch_2sym": [3],
    "check_server": [4],
    "decrease_a3": [5],
}
rag.cases = [
    {"scenario_id": f"s{i}", "family": "F1", "action_type": action, "answer": f"C{i}", "feature_vector": rag.feature_matrix[i]}
    for i, action in enumerate(["lift_tilt", "insufficient_data", "add_neighbor", "pdcch_2sym", "check_server", "decrease_a3"])
]

hits = rag.retrieve(MockSF(), k=6, include_confusion_pairs=False)
assert len(hits) <= 6
assert all(isinstance(h, RAGHit) for h in hits)
assert len({h.gt_action_type for h in hits}) >= 2
```

---

## 禁止事项

- 不得修改 `_retrieve_raw()` 或 `retrieve_and_format()`
- 不得修改 `retrieve()` 参数签名
- 不得在 D2b 阶段实现 confusion pair 逻辑
