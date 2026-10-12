# Task D6 — 特征向量扩展至 16 维（工具证据维度）

**优先级**: 🟡 低  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: D2a 完成 + Owner C 的 C4 落地  
**协调要求**: 与 Owner A 的下一轮重训窗口对齐

---

## 背景

当前 13 维向量只编码本地 RF / KPI 线索，不含工具证据。  
D6 的目标是在**不改公开检索签名**的前提下，把工具证据压进向量尾部三维。

为保证与当前代码库一致，本任务以**当前 `skills/tool_caller.py` 的字段语义**为准：

- `mainlobe_flags: Dict[int, bool]`
  - 当前 `bool=True` 表示“UE 在该小区主瓣外”
- `tilt_angles: Dict[int, float]`
- `overlap_ratio: Optional[float]`

如果 Owner C 的 C4 最终把这些字段扩展成更丰富的结构，D6 只允许做**向后兼容读取**，不能破坏当前骨架。

---

## 新增三维定义

| 维度索引 | 名称 | 当前语义 | 归一化 |
|---|---|---|---|
| 13 | `outside_mainlobe_ratio` | `mainlobe_flags` 中值为 `True` 的比例 | `outside_count / max(n,1)` |
| 14 | `tilt_angle_mean` | `tilt_angles` 平均值 | `mean / 20.0`，再裁剪到 `[0,1]` |
| 15 | `overlap_ratio_scalar` | `overlap_ratio` 标量；若未来变成 dict，则取最大值 | 已在 `[0,1]` 或再裁剪 |

若 `sf` 没有 `tool_evidence`，或 `tool_evidence is None`，则这三维都为 `0.0`。

---

## 实现要求

### 6.1 修改 `FEATURE_DIM`

```python
FEATURE_DIM = 16
```

### 6.2 修改 `_extract_feature_vector`

训练数据没有工具证据，因此在原 13 维末尾拼接 `[0.0, 0.0, 0.0]`。

### 6.3 修改 `_sf_to_feature_vec`

在 D2a 的 13 维桥接基础上追加三维：

```python
@staticmethod
def _sf_to_feature_vec(sf) -> np.ndarray:
    # 先保留原 13 维逻辑
    base = ...

    tool_ev = getattr(sf, "tool_evidence", None)
    outside_ratio = 0.0
    tilt_mean = 0.0
    overlap_scalar = 0.0

    if tool_ev is not None:
        ml_flags = getattr(tool_ev, "mainlobe_flags", {}) or {}
        tilt_angles = getattr(tool_ev, "tilt_angles", {}) or {}
        raw_overlap = getattr(tool_ev, "overlap_ratio", None)

        n = max(len(ml_flags), 1)
        outside_ratio = sum(1 for v in ml_flags.values() if bool(v)) / n

        if tilt_angles:
            tilt_mean = min(sum(tilt_angles.values()) / len(tilt_angles) / 20.0, 1.0)

        if isinstance(raw_overlap, dict):
            overlap_scalar = float(max(raw_overlap.values(), default=0.0))
        elif isinstance(raw_overlap, (int, float)) and raw_overlap is not None:
            overlap_scalar = float(raw_overlap)

    tool_vec = np.array([outside_ratio, tilt_mean, overlap_scalar], dtype=np.float32)
    return np.clip(np.concatenate([base, tool_vec]), 0.0, 1.0)
```

### 6.4 保持 D1b 的缓存维度校验

D1b 已经要求在缓存加载时检查 `feature_matrix.shape[1] != FEATURE_DIM` 就自动重建。  
D6 **不要再发明第二套重建机制**；只要确认这条路径在 16 维时仍然成立。

---

## 验收测试

```python
def test_extract_feature_vector_shape_16():
    vec = StructuredCaseRAG._extract_feature_vector({"data": {}})
    assert vec.shape == (16,)

def test_sf_to_feature_vec_tool_dims_nonzero():
    class MockToolEvidence:
        mainlobe_flags = {1: True, 2: False}  # outside ratio = 0.5
        tilt_angles = {1: 10.0}               # 10 / 20 = 0.5
        overlap_ratio = 0.7

    sf = make_mock_sf()
    sf.tool_evidence = MockToolEvidence()
    vec = StructuredCaseRAG._sf_to_feature_vec(sf)
    assert vec.shape == (16,)
    assert abs(vec[13] - 0.5) < 1e-4
    assert abs(vec[14] - 0.5) < 1e-4
    assert abs(vec[15] - 0.7) < 1e-4

def test_sf_to_feature_vec_none_tool_evidence():
    sf = make_mock_sf()
    sf.tool_evidence = None
    vec = StructuredCaseRAG._sf_to_feature_vec(sf)
    assert vec[13] == 0.0
    assert vec[14] == 0.0
    assert vec[15] == 0.0
```

---

## 禁止事项

- 不得修改 `retrieve()` 的公开签名
- 不得绕过 D1b 的缓存维度校验
- 不得假设 `mainlobe_flags` 的布尔语义是“in mainlobe”；当前代码里它表示“outside mainlobe”
