# Task D1b — SHA-256 缓存校验 + FAISS 索引集成

**粒度**: 小（修改 `build_index()` 入口逻辑 + 新增 2 个私有方法）  
**优先级**: 🟠 中——提升开发效率（改文件后自动重建），FAISS 是可选优化  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: D1a 完成（`_action_indices` 已存在，缓存 pkl 结构已更新）  
**后续依赖**: D2a（需要 `self.feature_matrix` 可从缓存加载）

---

## 背景

当前 `build_index()` 没有任何缓存校验机制，每次运行都全量重建（2000 条 × feature extraction ≈ 数十秒）。本任务添加：
1. SHA-256 哈希校验：`train.json` 未变则跳过重建
2. 可选 FAISS 索引：`faiss-cpu` 可用时构建 `IndexFlatIP`，不可用时静默降级到 NumPy

优化要求：
- **缓存正确性优先于性能**。只要缓存结构缺字段、矩阵维度不匹配、或 hash 缺失，都必须重建。
- 这一步先把“维度校验”做进来，后续 D6 从 13 维扩到 16 维时就不需要再额外补一套旧缓存防御逻辑。

---

## 需要修改的内容

### 变更 1：在文件顶部添加 FAISS 可选导入

在现有 `import numpy as np` 下方添加：

```python
try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss = None
    _FAISS_AVAILABLE = False
```

### 变更 2：新增 `_sha256_file` 静态方法

在 `StructuredCaseRAG` 类内部添加：

```python
@staticmethod
def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

### 变更 3：新增 `_build_faiss_index` 私有方法

```python
def _build_faiss_index(self, matrix: np.ndarray):
    """构建 FAISS IndexFlatIP（内积 = 余弦，矩阵已 L2 归一化）。
    faiss-cpu 不可用时返回 None。"""
    if not _FAISS_AVAILABLE:
        return None
    norm_matrix = matrix.copy().astype(np.float32)
    _faiss.normalize_L2(norm_matrix)
    index = _faiss.IndexFlatIP(norm_matrix.shape[1])
    index.add(norm_matrix)
    return index
```

### 变更 4：修改 `build_index()` 入口——缓存命中检查

在 `build_index()` 方法的**最开头**（`force_rebuild=False` 参数下），替换或包裹原有的"是否加载缓存"逻辑：

```python
def build_index(self, force_rebuild: bool = False):
    os.makedirs(self.CACHE_DIR, exist_ok=True)

    cases_path  = os.path.join(self.CACHE_DIR, "rag_v2_cases.pkl")
    matrix_path = os.path.join(self.CACHE_DIR, "rag_v2_matrix.npy")
    hash_path   = os.path.join(self.CACHE_DIR, "data_hash.txt")
    faiss_path  = os.path.join(self.CACHE_DIR, "rag_v2_faiss.index")

    current_hash = self._sha256_file(self.train_json_path)
    cached_hash  = ""
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            cached_hash = f.read().strip()

    cache_valid = (
        not force_rebuild
        and current_hash == cached_hash
        and os.path.exists(cases_path)
        and os.path.exists(matrix_path)
    )

    if cache_valid:
        print("[RAG V2] Loading cached index...")
        with open(cases_path, "rb") as f:
            data = pickle.load(f)
        self.cases           = data["cases"]
        self.family_indices  = data["family_indices"]
        self._action_indices = data.get("action_indices", {})
        self.feature_matrix  = np.load(matrix_path)
        if self.feature_matrix.ndim != 2 or self.feature_matrix.shape[1] != self.FEATURE_DIM:
            print(
                f\"[RAG V2] Feature dim mismatch "
                f"({self.feature_matrix.shape} vs *x{self.FEATURE_DIM}), rebuilding...\"
            )
            return self.build_index(force_rebuild=True)
        # 尝试加载 FAISS（可选）
        if _FAISS_AVAILABLE and os.path.exists(faiss_path):
            self._faiss_index = _faiss.read_index(faiss_path)
        else:
            self._faiss_index = None
        return

    print("[RAG V2] Building index from scratch...")
    # ... 原有的构建逻辑不变 ...
```

### 变更 5：在 `build_index()` 构建完成后写入缓存

在原有的 `pickle.dump(...)` 和 `np.save(...)` 之后，追加（**FAISS 写入和 hash 写入必须最后，防止部分重建**）：

```python
# 保存 FAISS 索引（可选）
self._faiss_index = self._build_faiss_index(self.feature_matrix)
if self._faiss_index is not None:
    _faiss.write_index(self._faiss_index, faiss_path)

# 最后写入 hash（表示本次构建完整）
with open(hash_path, "w") as f:
    f.write(current_hash)
```

### 变更 6：在 `__init__` 中初始化 `_faiss_index`

```python
self._faiss_index = None
```

---

## 最终缓存目录布局

```
.rag_cache_v2/
  rag_v2_cases.pkl       ← cases + family_indices + action_indices
  rag_v2_matrix.npy      ← (N, FEATURE_DIM) float32
  rag_v2_faiss.index     ← FAISS IndexFlatIP（可选）
  confusion_index.pkl    ← D3a 产出（本任务不创建，只预留）
  data_hash.txt          ← SHA-256，最后写入（原子性标志）
```

---

## 验收检查

```bash
python -c "
from rag_retriever_v2 import StructuredCaseRAG
import time

r = StructuredCaseRAG('data/Phase_1/train.json')

# 第一次：应打印 'Building'
r.build_index(force_rebuild=True)
assert r.feature_matrix is not None

# 第二次：应打印 'Loading cached'（速度 < 5s）
t0 = time.perf_counter()
r2 = StructuredCaseRAG('data/Phase_1/train.json')
r2.build_index()
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'cache load too slow: {elapsed:.1f}s'
print(f'cache load in {elapsed:.2f}s — D1b OK')
"
```

---

## 禁止事项

- `data_hash.txt` **必须最后写入**（防止构建中途失败留下无效 hash）
- 不得在 FAISS 不可用时抛出任何异常（静默降级）
- 不得修改 `main_v3.py`
- `FEATURE_DIM` 保持 13（D6 任务负责改为 16）
- 缓存命中分支不得“将错就错”加载坏矩阵；形状不对必须触发重建
