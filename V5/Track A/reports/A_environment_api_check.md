# A Environment / API Check

检查时间：2026-05-15

## 结论

A 侧环境前置检查通过，可以进入 B patch review / Phase1 smoke gate 阶段。

当前不要直接启动 500 题 cloud rerun。全量 rerun 的 go/no-go 仍然取决于 B 交付的三层 guard patch 和 Phase1 smoke 结果。

## 已确认

### 1. V5 主流水线文件存在

工作目录：

```text
/home/huangxiaolin/AITO/V5/Track A
```

关键文件已在 V5 内自包含：

```text
phase2_run.py
rag_retriever_v2.py
skills/candidate_generator.py
data/Phase_1/train.json
data/Phase_2/test.json
```

### 2. Python 语法 / 导入级检查通过

检查命令：

```bash
/home/huangxiaolin/AITO/bin/python -m py_compile phase2_run.py rag_retriever_v2.py scripts/B_phase1_knn_calibration.py scripts/C_f2_pair_repair.py skills/*.py
```

结果：通过。

### 3. `phase2_run.py` 可正常启动

检查命令：

```bash
/home/huangxiaolin/AITO/bin/python phase2_run.py --help
```

结果：通过。

确认参数包括：

```text
--server_url
--token
--test_json
--out
--diagnostics_jsonl
--llm_policy
--agentic_policy
--resume
--max_samples
--scenario_ids
```

### 4. Track A API / token 可用

Health endpoint：

```text
https://120.46.145.77/no/health
```

结果：

```text
HTTP 200
body: {"status":"ok"}
elapsed: 1.33s
```

说明：cloud API 服务可访问，当前 token 可用。

### 5. `phase2_run.py` 真实 API smoke 通过

检查方式：用 `--max_samples 1` 跑 1 个 Phase2 scenario，不启用 LLM，不消耗提交次数。

输出文件：

```text
reports/A_phase2_api_smoke_result.csv
reports/A_phase2_api_smoke_diagnostics.jsonl
logs/A_phase2_api_smoke.log
```

结果：

```text
returncode: 0
answered: 1/1
scenario_id: 40573780-92ac-4436-97c7-efca33b2a839
family: F1
answer: C7
elapsed: 10s
api_call_count: 5
api_errors: 0
```

说明：V5 主流水线可以实际调用 Track A API 并生成 diagnostics。

### 6. Fallback CSV 格式通过

以下文件均通过 `scripts/check_600_format.py`：

```text
submissions/submission_600_v5_07_adj_only.csv
submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
submissions/submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv
```

共同结果：

```text
READY
rows: 600
Track A answered: 500
Track B non-empty: 0
```

## A 接下来怎么做

### 立即可做

- 等 B 交付 `candidate_generator.py` / `phase2_run.py` / post-output guard / smoke script patch；
- review B patch 是否覆盖：
  - F1 / F1_5BS 禁止 final `adjust_azimuth`；
  - F2 exactly 2 codes；
  - F2 只能是 `{adjust_azimuth, increase_power}` 或 `{decrease_power, press_down_tilt}`；
  - F4M exactly 2 个 `decrease_interfreq`；
  - post-output guard 最后重放 2 个 audited LoRA overrides；
- 跑 B 的 Phase1 smoke script。

### Go 条件

只有全部满足时，A 才启动 full cloud rerun：

```text
F1 single-answer top1 accuracy >= 0.80
F2 legal-pair top1 accuracy >= 0.50
F4 top1 accuracy >= 0.95
invalid F2 outputs = 0
cloud API health = OK
```

### No-Go 条件

任一关键 smoke gate 失败，或 patch 到 Day 1 PM 仍不稳定：

```text
Sub1 = submissions/submission_600_v5_07_adj_only.csv
Sub2 = submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
```

如果 patched rerun 已提交但 public score 低于 0.130：

```text
Sub2 = submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
```
