# Task D2a — 检索辅助方法：`_cosine_sim_to_subset` / `_sf_to_feature_vec` / `_deduplicate` / `_case_to_raghit`

**粒度**: 小（4 个辅助方法，可独立测试）  
**优先级**: 🔴 高——D2b 直接依赖  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: D1b 完成（`feature_matrix`、`_action_indices`、缓存逻辑已可用）  
**后续依赖**: D2b

---

## 背景

D4/D1a/D1b 已经把以下基础打好：

- `RAGHit` dataclass 已存在
- `retrieve(sf, ...)` 公开接口已冻结，但目前仍是空桩
- `_retrieve_raw()` 和 `retrieve_and_format()` 继续服务 `main_v3.py`
- `build_index()` 已能生成 `feature_matrix` 和 `_action_indices`

本任务只补齐 D2b 所需的私有辅助方法，**不触碰主流程消费方**。

---

## 需要新增的 4 个方法

### 方法 1：`_cosine_sim_to_subset`

```python
def _cosine_sim_to_subset(self, query: np.ndarray, subset_idxs: List[int]) -> np.ndarray:
    """Return cosine similarities between query and feature_matrix[subset_idxs]."""
    if not subset_idxs:
        return np.array([], dtype=np.float32)
    sub = self.feature_matrix[subset_idxs].astype(np.float32)
    q = query.reshape(1, -1).astype(np.float32)
    norms_sub = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
    norm_q = float(np.linalg.norm(q)) + 1e-8
    return ((sub / norms_sub) @ (q / norm_q).T).flatten()
```

### 方法 2：`_sf_to_feature_vec`

```python
@staticmethod
def _sf_to_feature_vec(sf) -> np.ndarray:
    """
    Convert ScenarioFeatures-like input to the same 13-dim float32 vector used
    by main_v3._sf_to_feature_vec().
    """
    tp = sf.throughput
    mob = sf.mobility
    ifreq = sf.inter_freq
    vec = np.array([
        (tp.worst_throughput_mbps or 0.0) / 500.0,
        (tp.throughput_drop_magnitude or 0.0) / 500.0,
        ((mob.serving_rsrp_dbm or -140.0) + 140.0) / 60.0,
        ((mob.serving_sinr_db or -20.0) + 20.0) / 40.0,
        (mob.neighbor_advantage_db or 0.0) / 20.0,
        ((mob.a3_margin_db or -10.0) + 10.0) / 20.0,
        (ifreq.max_dl_prb_util or 0.0) / 100.0,
        float(len(getattr(mob, "visible_neighbor_pcis", []) or [])) / 5.0,
        sf.score_late_handover,
        sf.score_missing_neighbor,
        sf.score_interference,
        sf.score_coverage_hole,
        sf.score_inter_freq_load,
    ], dtype=np.float32)
    return np.clip(vec, 0.0, 1.0)
```

> 必须与 `main_v3.py::_sf_to_feature_vec()` 保持一致。  
> 这里用的是当前仓库真实字段名：`throughput_drop_magnitude`、`score_interference`。

### 方法 3：`_deduplicate`

```python
@staticmethod
def _deduplicate(hits: List[dict], query_vec: np.ndarray, threshold: float = 0.98) -> List[dict]:
    """
    Drop hits whose feature_vector is too close to query_vec.
    This is mainly used to remove self-hits / near-duplicates.
    """
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    kept = []
    for hit in hits:
        vec = hit["feature_vector"].astype(np.float32)
        v_norm = vec / (np.linalg.norm(vec) + 1e-8)
        if float(np.dot(q_norm, v_norm)) < threshold:
            kept.append(hit)
    return kept
```

### 方法 4：`_case_to_raghit`

```python
def _case_to_raghit(self, case: dict, sim: float, is_confusion: bool = False) -> "RAGHit":
    """
    Convert internal case dict → public RAGHit.
    Note: in the current codebase RAGHit.features is a compact dict, not a full
    ScenarioFeatures object.
    """
    vec = case["feature_vector"]
    features = {
        "worst_tp": float(vec[0]),
        "serving_rsrp": float(vec[2]),
        "serving_sinr": float(vec[3]),
        "neighbor_advantage": float(vec[4]),
        "max_prb_util": float(vec[6]),
        "score_late_handover": float(vec[8]),
        "score_missing_neighbor": float(vec[9]),
        "score_coverage_hole": float(vec[11]),
    }
    snippet = (
        f"[{case.get('family', '?')}] {case.get('action_type', '?')} "
        f"sim={sim:.3f} ans={case.get('answer', '?')}"
    )[:120]
    return RAGHit(
        scenario_id=case["scenario_id"],
        features=features,
        gt_c_code=case.get("answer", ""),
        gt_action_type=case.get("action_type", "unknown"),
        similarity=sim,
        prompt_snippet=snippet,
        is_confusion_pair=is_confusion,
    )
```

---

## 验收检查

优先写成 `tests/test_rag_retriever.py` 的合成测试，不依赖 `train.json`。

```python
import numpy as np
from rag_retriever_v2 import RAGHit, StructuredCaseRAG

rag = StructuredCaseRAG.__new__(StructuredCaseRAG)
rag.feature_matrix = np.eye(13, dtype=np.float32)

query = np.ones(13, dtype=np.float32)
sims = rag._cosine_sim_to_subset(query, [0, 1, 2])
assert sims.shape == (3,)
assert len(rag._cosine_sim_to_subset(query, [])) == 0

hits = [
    {"feature_vector": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32), "_similarity": 0.99},
    {"feature_vector": np.array([0,1,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32), "_similarity": 0.30},
]
kept = StructuredCaseRAG._deduplicate(hits, hits[0]["feature_vector"], threshold=0.98)
assert len(kept) == 1

case = {
    "scenario_id": "s1",
    "family": "F1",
    "action_type": "lift_tilt",
    "answer": "C3",
    "feature_vector": np.zeros(13, dtype=np.float32),
}
hit = rag._case_to_raghit(case, 0.85)
assert isinstance(hit, RAGHit)
assert hit.similarity == 0.85
assert hit.is_confusion_pair is False
assert len(hit.prompt_snippet) <= 120
```

---

## 禁止事项

- 不得修改 `retrieve()` 的公开参数签名
- 不得修改 `_retrieve_raw()` 或 `retrieve_and_format()`
- 不得修改 `main_v3.py`
