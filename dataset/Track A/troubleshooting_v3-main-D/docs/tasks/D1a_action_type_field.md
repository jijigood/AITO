# Task D1a — `build_index()` 中提取 `action_type` 字段

**粒度**: 小（仅修改 `build_index()` 循环内部 + 复用现有解析逻辑 + 1 个实例属性）  
**优先级**: 🔴 高——D2a、D2b 的分层检索依赖此字段  
**负责文件**: `rag_retriever_v2.py`  
**前置依赖**: 无（建议在 D4 之后串行合入，避免同文件冲突）  
**后续依赖**: D1b、D2a（都需要 `_action_indices` 已存在）

---

## 背景

当前 `build_index()` 在构建 case dict 时不提取答案对应的动作类型，导致后续无法按 action_type 分桶检索。本任务**只做一件事**：在 `build_index()` 的 case 构建循环中新增 `action_type` 字段，并维护 `_action_indices` 分桶字典。

优化要求：**不要再复制一份动作正则表**。仓库里已有 `skills/scenario_parser.py::parse_option_label()` 作为 canonical parser，本任务直接复用它，避免后续标签规则漂移。

---

## 需要修改的内容

### 变更 1：在文件顶部引入现有解析器

在 `import` 区块中新增：

```python
from skills.scenario_parser import parse_option_label
```

> `scenario_parser.py` 不依赖 `rag_retriever_v2.py`，这里不存在循环导入问题。  
> 本任务的目标是复用唯一真源，而不是把规则复制到第二个文件里。

### 变更 2：在 `__init__` 中初始化 `_action_indices`

在 `StructuredCaseRAG.__init__` 中，在 `self.family_indices` 初始化的下方添加：

```python
self._action_indices: Dict[str, List[int]] = {}
```

### 变更 3：在 `build_index()` 的 case 构建循环中填充 `action_type`

找到 `build_index()` 中构建单个 case dict 的循环（将已有的 `case = {...}` 赋值语句之后），添加：

```python
# 提取 action_type
task_block = raw.get("task", {})
options_list = task_block.get("options", [])
options_map = {opt["id"]: opt["label"] for opt in options_list}
answer_str = raw.get("answer", "")
first_c_code = answer_str.split("|")[0].strip() if answer_str else ""
gt_label = options_map.get(first_c_code, "")
parsed = parse_option_label(gt_label) if gt_label else {}
action_type = parsed.get("action_type", "unknown")

case["action_type"] = action_type

global_idx = len(self.cases)  # 当前 case 将被 append 的索引
self._action_indices.setdefault(action_type, []).append(global_idx)
```

⚠️ `global_idx` 的值必须与 case 实际被 `self.cases.append(case)` 后的索引一致——请确认此处使用 `len(self.cases)` 是在 `append` **之前**（即 `0-based` 预先记录），或调整为 `append` 之后 `len(self.cases) - 1`，以保证索引正确。

### 变更 4：`_action_indices` 随缓存一起保存

在 `build_index()` 写入 `rag_v2_cases.pkl` 的 `pickle.dump(...)` 调用中，将 dump 对象改为包含三个字段：

```python
pickle.dump({
    "cases":          self.cases,
    "family_indices": self.family_indices,
    "action_indices": self._action_indices,   # 新增
}, f)
```

在 `build_index()` 的缓存加载分支中，同步读取：

```python
data = pickle.load(f)
self.cases          = data["cases"]
self.family_indices = data["family_indices"]
self._action_indices = data.get("action_indices", {})  # 向后兼容旧缓存
```

---

## 验收检查

```bash
python -c "
import json, os
from rag_retriever_v2 import StructuredCaseRAG

# 用真实 train.json（或少量数据）构建索引
r = StructuredCaseRAG('data/Phase_1/train.json')
r.build_index(force_rebuild=True)

# 检查 _action_indices 不为空
assert r._action_indices, '_action_indices is empty'

# 检查所有 case 都有 action_type 字段
assert all('action_type' in c for c in r.cases), 'some cases missing action_type'

# 检查 lift_tilt 存在
assert 'lift_tilt' in r._action_indices, 'lift_tilt bucket missing'

# 检查 unknown 比例 < 10%
total = len(r.cases)
unknown = len(r._action_indices.get('unknown', []))
assert unknown / total < 0.10, f'too many unknown: {unknown}/{total}'

print('D1a OK')
"
```

---

## 禁止事项

- 不得修改 case dict 中任何已有字段
- 不得复制第二份 `_ACTION_PATTERNS` 到 `rag_retriever_v2.py`
- 不得修改 `main_v3.py`
- `_action_indices` 不得暴露为公开属性（带下划线前缀）
