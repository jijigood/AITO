# Task D5b — `skills/data_tools.py`：`label_confidence` + `ambiguity_score` + `score_all_scenarios`

**粒度**: 小（3 个函数）  
**优先级**: 🟠 高  
**负责文件**: `skills/data_tools.py`  
**前置依赖**: D5a 完成  
**后续依赖**: Owner A 的 `curate_sft.py`

---

## 背景

D5a 已经给了我们：

- 可复用的 `_get_rag()` 单例
- `duplicate_score`
- `outlier_score`

本任务补全剩余三项。为保持任务自洽，**这三项只依赖当前 RAG 索引已有信息**，不依赖候选生成器，也不依赖服务端工具。

---

## 设计原则

### `label_confidence`

目标是“这个标签像不像一个可靠样本”，因此只使用三类信号：

1. `gt` 是否真的出现在 options 里；不在则直接 `0.0`
2. 该样本与**同动作类**样本是否相似
3. 该样本是否被**其他动作类**强烈竞争

### `ambiguity_score`

目标是“这个样本在动作类边界上有多模糊”，因此定义成：

- 同动作类最近邻支持越强，越不模糊
- 异动作类最近邻越接近，越模糊

这样就不需要发明一套和真实 pipeline 脱节的“维度 → 动作”硬编码规则。

---

## 需要修改的内容

### 变更 1：新增一个私有相似度 helper（可选但推荐）

```python
def _max_cosine_to_indices(vec: np.ndarray, matrix: np.ndarray, indices: List[int]) -> float:
    if not indices:
        return 0.0
    sub = matrix[indices].astype(np.float32)
    v = vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
    return float(np.max(sub @ v.reshape(-1, 1)))
```

### 变更 2：实现 `label_confidence`

```python
def label_confidence(scenario_id: str, _rag=None) -> float:
    """
    0.0 = clearly suspicious / mislabeled
    1.0 = highly trustworthy
    """
    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    case = r.cases[global_idx]
    vec = r.feature_matrix[global_idx]

    answer_str = case.get("answer", "")
    gt_c_code = answer_str.split("|")[0].strip() if answer_str else ""
    options = case.get("options", {})
    if not gt_c_code or gt_c_code not in options:
        return 0.0

    action_type = case.get("action_type", "unknown")
    same_action = [gi for gi in r._action_indices.get(action_type, []) if gi != global_idx]
    if same_action:
        same_support = _max_cosine_to_indices(vec, r.feature_matrix, same_action)
    else:
        same_support = 0.5

    family = case.get("family", "")
    rival_idxs = [
        gi for gi in r.family_indices.get(family, [])
        if gi != global_idx and r.cases[gi].get("action_type") != action_type
    ]
    rival_support = _max_cosine_to_indices(vec, r.feature_matrix, rival_idxs) if rival_idxs else 0.0

    margin = max(0.0, same_support - rival_support)
    return float(np.clip(0.4 + 0.4 * same_support + 0.2 * margin, 0.0, 1.0))
```

### 变更 3：实现 `ambiguity_score`

```python
def ambiguity_score(scenario_id: str, _rag=None) -> float:
    """
    0.0 = very clear class boundary
    1.0 = strong cross-class competition
    """
    r = _get_rag(_rag=_rag)
    global_idx = _sid_to_global_idx(r, scenario_id)
    if global_idx is None:
        return 0.0

    case = r.cases[global_idx]
    vec = r.feature_matrix[global_idx]
    action_type = case.get("action_type", "unknown")
    family = case.get("family", "")

    same_action = [gi for gi in r._action_indices.get(action_type, []) if gi != global_idx]
    rival_idxs = [
        gi for gi in r.family_indices.get(family, [])
        if gi != global_idx and r.cases[gi].get("action_type") != action_type
    ]

    same_support = _max_cosine_to_indices(vec, r.feature_matrix, same_action) if same_action else 0.0
    rival_support = _max_cosine_to_indices(vec, r.feature_matrix, rival_idxs) if rival_idxs else 0.0

    if same_support <= 1e-8:
        return float(np.clip(rival_support, 0.0, 1.0))
    return float(np.clip(rival_support / max(same_support, 1e-8), 0.0, 1.0))
```

### 变更 4：实现 `score_all_scenarios`

```python
def score_all_scenarios(scenario_ids: List[str]) -> Dict[str, Dict[str, float]]:
    r = _get_rag()

    # duplicate_score cache
    dup_cache: Dict[str, float] = {}
    for family, fidxs in r.family_indices.items():
        if len(fidxs) < 2:
            for gi in fidxs:
                dup_cache[r.cases[gi]["scenario_id"]] = 0.0
            continue
        sub = r.feature_matrix[fidxs].astype(np.float32)
        sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
        sim_matrix = sub @ sub.T
        np.fill_diagonal(sim_matrix, -1.0)
        max_sims = sim_matrix.max(axis=1)
        for li, gi in enumerate(fidxs):
            dup_cache[r.cases[gi]["scenario_id"]] = float(max_sims[li])

    try:
        from tqdm import tqdm
        iterator = tqdm(scenario_ids, desc="Scoring") if sys.stdout.isatty() else scenario_ids
    except ImportError:
        iterator = scenario_ids

    out: Dict[str, Dict[str, float]] = {}
    for sid in iterator:
        out[sid] = {
            "duplicate": duplicate_score(sid, cache=dup_cache, _rag=r),
            "outlier": outlier_score(sid, _rag=r),
            "confidence": label_confidence(sid, _rag=r),
            "ambiguity": ambiguity_score(sid, _rag=r),
        }
    return out
```

---

## 验收检查

使用注入 fake rag：

```python
import numpy as np
from skills.data_tools import (
    _inject_rag_singleton,
    _reset_rag_singleton,
    label_confidence,
    ambiguity_score,
    score_all_scenarios,
)

class FakeRAG:
    cases = [
        {
            "scenario_id": "s0",
            "family": "F1",
            "action_type": "lift_tilt",
            "answer": "C3",
            "options": {"C3": "Lift the tilt by 2 degrees"},
        },
        {
            "scenario_id": "s1",
            "family": "F1",
            "action_type": "check_server",
            "answer": "C5",
            "options": {"C5": "Check test server"},
        },
        {
            "scenario_id": "s2",
            "family": "F1",
            "action_type": "add_neighbor",
            "answer": "C6",
            "options": {},
        },
    ]
    family_indices = {"F1": [0, 1, 2]}
    _action_indices = {"lift_tilt": [0], "check_server": [1], "add_neighbor": [2]}
    feature_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

_reset_rag_singleton()
_inject_rag_singleton(FakeRAG())

assert label_confidence("s2") == 0.0
assert 0.0 <= label_confidence("s0") <= 1.0
assert 0.0 <= ambiguity_score("s0") <= 1.0

result = score_all_scenarios(["s0", "s1", "s2"])
assert set(result.keys()) == {"s0", "s1", "s2"}
for scores in result.values():
    assert set(scores.keys()) == {"duplicate", "outlier", "confidence", "ambiguity"}

_reset_rag_singleton()
```

---

## 禁止事项

- 不得依赖候选生成器或 HTTP 工具
- 不得修改公开函数名
- `_rag` 仍视为内部优化参数，不写入对外文档
