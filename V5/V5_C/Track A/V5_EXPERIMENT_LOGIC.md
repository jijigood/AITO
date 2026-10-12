# V5 实验逻辑说明


这份文档解释 V5 测试包为什么这样设计，以及它和 RAG、prompt、LoRA、规则之间的关系。当前 V5 已经是自包含的 patched rerun 冲刺包，主线请先读 `RERUN_SPRINT_PLAN.md`、`BCD_QUICK_START.md` 和 `TEAM_SPLIT.md`。

重要约束：全队只剩 2 次正式提交机会。V5 里既有完整主流水线，也有 fallback CSV；这些 CSV 是安全网和候选池，不是让大家逐个提交的测试队列。

## 0. 当前策略更新：0.20 目标下优先 patched rerun

如果目标只是 0.14-0.16，V5-only 方案更稳。

但如果目标是 public score 0.20，V5-only 的改动面不够，应该把第 1 次正式提交押在 patched rerun：

```text
Sub1 = patched rerun
     + candidate_generator guards
     + phase2_run prompt/cardinality constraints
     + post-output guard
     + 2 audited LoRA overrides
     + C's F2 calibration patches if ready
```

V5 的角色改为安全网：

```text
B patch / smoke / cloud API 失败：
  Sub1 = v5_07
  Sub2 = v5_10

patched rerun Sub1 < 0.130：
  Sub2 = v5_11
```

V5 现在包含完整主流水线，B/C/D 直接在这个目录工作：

```text
/home/huangxiaolin/AITO/V5/Track A
```

关键 patch 目标：

```text
phase2_run.py
skills/candidate_generator.py
skills/feature_engine.py
skills/scenario_router.py
skills/api_data_gatherer.py
rag_retriever_v2.py
data/Phase_1/train.json
data/Phase_2/test.json
```

## 1. 赛题到底在考什么

Track A 是 5G 无线网络路测故障诊断任务。

每个测试场景会给：

- 一个故障描述；
- 一批可通过 Track A API / tools 获取的网络数据；
- 24 个候选操作，编号为 `C1` 到 `C24`；
- 一个题型标签，比如 single-answer 或 multiple-answer；
- 需要我们输出正确的 `C-code` 答案。

答案形式可能是：

```text
C7
C3|C11
C2|C7|C14|C20
```

平台打分本质上看我们选的答案集合和真实答案集合的重合度。对于多选题，不是必须完全一样才有分，但漏选、错选都会降低 IOU。

Phase2 提交还有一个很重要的格式要求：

```text
ID,Track A,Track B
```

并且必须保留 600 行 ID 模板：

- 前 500 行是 Track A；
- 后 100 行是 Track B；
- 即使 Track B 不做，也必须保留 ID 和空的 `Track B` 列。

之前 500 行 raw result 会报 Track B missing IDs，所以 V5 里的所有候选提交都固定为 600 行。

## 2. 我们现在的系统是怎么工作的

当前 V5 Track A 实现可以理解成一条诊断流水线：

```text
Phase2 test.json
-> 对每个 scenario 调 API 获取数据
-> parse_scenario 解析场景、选项、小区和 C-code
-> scenario_router 判断题型族
-> feature_engine 提取无线特征和问题分数
-> candidate_generator 生成候选答案排名
-> 可选：RAG / prompt / LoRA / LLM verifier 做仲裁
-> 输出 submission CSV
```

核心模块大致是：

- `phase2_run.py`：Phase2 主入口，负责整条流水线。
- `skills/api_data_gatherer.py`：调用 Track A API / tools 拿数据。
- `skills/scenario_parser.py`：解析场景、选项和 `C1-C24` 的 action type。
- `skills/scenario_router.py`：判断 F1/F2/F3/F4 等题型族。
- `skills/feature_engine.py`：从 RF/KPI/MR/config 中提取特征分数。
- `skills/candidate_generator.py`：把特征分数转换成候选 C-code。
- `rag_retriever_v2.py`：结构化 RAG / KNN 检索。
- `skills/agentic_overlay.py`：高风险样本上的 LLM 仲裁。

这些文件现在都已经放在 V5 包里。patched rerun、Phase1 smoke、C 的 F2 校准和 V5 fallback 都应在 V5 目录下完成。

系统里的 issue score 包括：

```text
late_handover
missing_neighbor
coverage_hole
overshoot_overlap
pdcch_issue
backhaul_server
insufficient_data
inter_freq_load
interference
```

这些 score 最后会映射到具体动作，比如：

```text
missing_neighbor -> add_neighbor
late_handover -> decrease_a3
coverage_hole -> lift_tilt
pdcch_issue -> pdcch_2sym
backhaul_server -> check_server
```

注意：`C1-C24` 的含义不是固定的。每道题里 `C7` 可能是 `lift_tilt`，另一道题里 `C7` 可能是 `decrease_a3`。所以所有分析都必须先解析每题的 option label，不能按 C-code 位置硬判断。

## 3. RAG、prompt、LoRA、规则分别是什么角色

它们不是互相排斥的路线，而是同一套诊断系统里的不同工具。

### 规则

规则负责把可解释的无线特征变成候选答案。

例如：

```text
如果 SINR 很差、邻区很强、A3 margin 异常
-> 更可能是 handover / neighbor 问题
-> 候选 add_neighbor 或 decrease_a3
```

规则的优点是可解释、稳定、便宜。缺点是边界 case 容易错。

### RAG / KNN

RAG 或 KNN 的作用是找相似历史样本，帮助判断当前 case 更像哪一类。

例如当前样本规则判成 `adjust_azimuth`，但相似训练样本多数是 `add_neighbor`，那它就可以给出一个改写 vote。

V5 里的 `knn_f1_votes.csv` 就是这种思想的产物：不是让模型重新自由回答，而是对 F1 样本做结构化近邻投票。

### Prompt / LLM

Prompt 的作用是让 LLM 在少数高风险边界上做仲裁。

它不应该泛泛地重判所有题，因为：

- 调用成本高；
- 输出不稳定；
- 很难归因；
- 赛末阶段时间不够。

更合理的用法是只让它回答非常具体的问题：

```text
这 20 个 pdcch_2sym case 里，哪些更像 check_server？
这 30 个 lift_tilt case 里，哪些其实是 insufficient_data？
```

### LoRA

LoRA 是 fine-tuned verifier。理想情况下，它可以学习 Phase1 的边界，把 deterministic top1 改成更合理的答案。

但目前从 0.1302 这轮结果看，LoRA gate 本身贡献有限。真正带来 public score 提升的，是 F1 分布被大范围重排。

所以 LoRA 仍然有用，但它必须证明：

```text
能稳定改对几十个 F1 case，而不是只触发 2 个 override。
```

## 4. 为什么 0.1 分数低

0.1 不是格式错，而是答案质量低。

因为 Track B 为空时，总分大约等于 Track A 分数除以 2。

```text
public score 0.1000
约等于 Track A accuracy 0.2000
```

这意味着 Track A 500 题大约只有 100 题量级是对的。

后来 0.1302 这轮说明我们确实修对了一部分：

```text
public score 0.130199999
约等于 Track A accuracy 0.2604
相对 0.1000 净增约 30 题
```

这次改动了 154 个 Track A 答案，净赚约 30 题。说明方向是有效的，但效率还不够高。

## 5. 当前最大问题集中在哪里

从 0.1302 文件看，最大可解释问题首先集中在 F1 single-answer；从 0.20 冲刺角度看，还必须同时修 rerun contract 和 F2 multi-answer 结构。

Phase1 训练集中，F1 单答案的六类大致均衡：

```text
check_server        239
decrease_a3         231
lift_tilt           254
insufficient_data   252
pdcch_2sym          240
add_neighbor        240
```

缩放到当前 Phase2 的 269 个 F1/F1_5BS single-answer，大致期望每类 43-47 个。

但当前 0.1302 文件里，F1/F1_5BS 分布是：

```text
decrease_a3         54
insufficient_data   45
adjust_azimuth      60
lift_tilt           71
pdcch_2sym          26
add_neighbor        13
check_server         0
```

这暴露出几个异常：

- `adjust_azimuth` 在 F1 single-answer 中明显异常；
- `lift_tilt` 偏多；
- `add_neighbor` 偏少；
- `pdcch_2sym` 偏少；
- `check_server` 完全缺失；
- 当前分布远离 Phase1 的 F1 标签结构。

所以问题不是“某个 prompt 写得不够好”，而是 F1 判断结构整体偏了。

另一个已经确认的结构问题是 F2：

```text
Phase1 F2 全部是 2-code legal pair；
当前 Phase2 有 10 行 F2 出现 cardinality 或 action-type 违规；
F2 只能在 {adjust_azimuth, increase_power} 和 {decrease_power, press_down_tilt} 两类 pair 中选择。
```

这也是为什么 patched rerun 必须修三层 contract：

```text
candidate filter
prompt/cardinality constraint
post-output guard
```

## 6. V5 的两种运行模式

V5 现在有两种模式：

```text
主攻模式：patched rerun，需要 Track A API Key，可能调用 LLM / prompt arbitration。
fallback 模式：离线 CSV postprocess，不需要 API Key，不需要 LLM。
```

主攻模式使用完整主流水线：

```text
phase2_run.py
skills/candidate_generator.py
skills/feature_engine.py
skills/api_data_gatherer.py
rag_retriever_v2.py
```

fallback 模式的目的不是重新跑整条流水线，而是做离线、可复现、可归因的提交实验。

fallback 使用的输入已经固定：

- 当前 0.1302 的 600 行提交；
- 已经跑出来的 Phase2 diagnostics；
- 已经生成的 KNN votes；
- Phase2 `test.json` 用来解析选项 action type。

fallback 做的事情只是：

```text
读取现有 CSV / diagnostics / votes
-> 根据明确规则改写一部分 F1 答案
-> 输出新的 600 行 submission CSV
```

所以不需要：

- 重新请求 Track A API；
- 重新拉 user plane / KPI / MR；
- 调 Qwen；
- 启 vLLM；
- 跑 LoRA verifier。

这有几个好处：

- 快；
- 稳定；
- 队友不需要密钥；
- 每个提交可以清楚解释改了哪些题；
- public score 涨跌可以归因到具体假设。

什么时候需要 API Key 或 LLM？

- 当前 0.20 主线 patched rerun 需要 Track A API Key。
- 如果要在 V5 中重新跑 `phase2_run.py` 全量获取数据，需要 Track A API Key。
- 如果要做新的 prompt arbitration，需要 LLM / vLLM。
- 如果要重新训练或调用 LoRA verifier，需要 LoRA 服务。
- 如果只是分析 V5 fallback 包、生成离线 CSV，不需要。

## 7. V5 fallback 文件到底在验证什么

V5 的主线是 patched rerun；下面这些 submission CSV 是 fallback 矩阵，不是最终答案，也不是逐个上传的提交列表。0.20 目标下，它们主要在 patched rerun 失败时保护提交下限。

注意：早期的 `knn_conf50/hardblock_adjust` 仍然混入了多个 transition，所以现在更推荐用 `adj_only` 做第 1 次正式提交。它的归因最干净。

### `submission_600_v5_07_adj_only.csv`

主假设：

```text
F1/F1_5BS single-answer 中当前为 adjust_azimuth 的答案，是否应该全部替换成更安全的非 adjust 动作。
```

动作：

- 只触碰当前 0.1302 文件中 F1/F1_5BS single-answer 且 action 为 `adjust_azimuth` 的 60 行；
- `KNN share >= 0.50` 且 vote 是安全 F1 action 时使用 KNN vote；
- 其他 adjust 行使用候选集中最安全的非 adjust action；
- 不改 `lift_tilt`、`check_server`、`pdcch_2sym` 等其他边界。

主要转移：

```text
adjust_azimuth -> add_neighbor   36
adjust_azimuth -> decrease_a3    24
```

它的 F1/F1_5BS 分布：

```text
decrease_a3         78
lift_tilt           71
add_neighbor        49
insufficient_data   45
pdcch_2sym          26
adjust_azimuth       0
check_server         0
```

这是 V5 fallback 中最适合第 1 次正式提交的候选，因为它只验证一个强结构性假设：`adjust_azimuth` 在 Phase1 F1 single-answer 标签中没有出现，而 Phase2 当前文件却有 60 个。

### `submission_600_v5_02_hardblock_adjust.csv`

主假设：

```text
F1/F1_5BS single-answer 中不应该出现 adjust_azimuth。
```

动作：

- 硬拦 `adjust_azimuth`；
- 优先用 KNN/conf50 或候选集中更安全的 F1 action 替代；
- 同时降低过量 `lift_tilt`；
- 补充 `add_neighbor` 和少量 `pdcch_2sym/check_server`。

它的 F1/F1_5BS 分布：

```text
decrease_a3         67
insufficient_data   64
add_neighbor        57
pdcch_2sym          30
lift_tilt           47
check_server         4
adjust_azimuth       0
```

它不再是首推版本，因为它同时改了 adjust 和非 adjust KNN flip。它适合当激进备选，或者当 `adj_only` 涨分后用于拆分哪些额外 transition 还能加入最终版。

### `submission_600_v5_01_knn_conf50.csv`

主假设：

```text
KNN vote share >= 0.50 的修复有稳定正收益。
```

动作：

- 只采用中等置信的 KNN vote；
- 保留少量 `adjust_azimuth`；
- 改动数比 hardblock 少，但不等于风险更低，因为它仍然混合了强证据和弱证据。

它适合用来判断 KNN 方向是否可靠，但不是最干净的首提交。原因是它同时包含：

- `adjust_azimuth -> add_neighbor/decrease_a3`；
- `lift_tilt -> insufficient_data`；
- 少量 `decrease_a3/insufficient_data` 的其他 KNN flip。

如果它涨分，很难判断到底是谁贡献；如果它跌分，也很难判断污染来自哪里。所以在只剩 2 次提交机会时，它应作为备选，不作为默认。

### `submission_600_v5_03_knn_all.csv`

主假设：

```text
完整 KNN/RAG 分布修复比当前 deterministic 分布更好。
```

动作：

- 使用所有 KNN votes；
- 更激进；
- 如果涨分，说明 KNN 方向可以放大；
- 如果跌分，说明低置信 vote 污染大。

### `submission_600_v5_06_knn_conf50_plus_check.csv`

主假设：

```text
check_server 是否有微弱可用信号。
```

注意：这个版本不是为了把 `check_server` 补到 40 个。它只比 conf50 多放开少量 check_server votes，用来探测是否有收益。

### `submission_600_v5_08_adj_only_tau70_reference.csv`

这个版本只按 candidate score ratio >= 0.70 改 6 个 `adjust_azimuth`，太保守，不适合第 1 次正式提交。它的作用是说明：如果完全依赖候选分数比例，改动范围会小到无法检验结构性假设。

### `submission_600_v5_09_adj_only_plus_calibrated_knn60.csv`

这个版本不是第 1 次提交候选，而是第 2 次整合参考。它在 `adj_only` 的 60 个改动基础上，只加入 Phase1 holdout 初步支持的 KNN60 transition：

```text
decrease_a3 -> add_neighbor       4
lift_tilt -> insufficient_data    4
```

总改动 68 行。只有当第 1 次 `adj_only` 涨分时，才考虑用它作为最终版起点。

### `submission_600_v5_10_adj_only_plus_f2_pair_fix.csv`

这个版本是当前更推荐的第 2 次默认候选：`v5_07_adj_only` 加上 F2 的 10 行结构修复。

F2 修复依据：

```text
Phase1 F2 全部是 cardinality=2；
Phase1 F2 只出现两类合法 pair：
{adjust_azimuth, increase_power}
{decrease_power, press_down_tilt}
```

当前 Phase2 有 10 行 F2 违反这个结构：8 行 cardinality=4，2 行虽然 cardinality=2 但 action 是 F1 泄漏。`scripts/C_f2_pair_repair.py` 已经把它们替换成候选列表中最高分的合法 F2 pair，7 个 azimuth pair、3 个 tilt pair。

它不放进第 1 次提交，是为了让第 1 次只测 F1 adjust 结构性假设；但它应该优先进入第 2 次。

### `submission_600_v5_11_baseline_plus_f2_pair_fix.csv`

这个版本只在 0.1302 baseline 上加入 F2 10 行修复。如果第 1 次 `v5_07` 持平或跌分，这是第 2 次最安全的 fallback。

### `submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv`

这个版本是 `v5_09 + F2 修复`，总计相对 baseline 改 78 行。只有当第 1 次 `v5_07` 涨分，并且 B 判断 KNN60 增量也值得带入时，才考虑它。

### Phase1 KNN holdout 校准

`conf50` 的另一个问题是 `share >= 0.50` 还没有被校准。现在新增：

```bash
/home/huangxiaolin/AITO/bin/python scripts/B_phase1_knn_calibration.py
```

输出：

```text
reports/B_phase1_knn_calibration.csv
reports/B_phase1_knn_calibration_details.jsonl
```

它在 Phase1 train 上切出 holdout，用已知 GT 估计每个 `share_bucket + transition` 的 `vote_precision` 和 `old_precision`。第 2 次最终提交只应该吸收 holdout 上精度真正更高的 transition。

当前已跑出的关键结果：

```text
0.50-0.59 lift_tilt -> insufficient_data：vote 0.4286，old 0.5714
0.60-0.69 lift_tilt -> insufficient_data：vote 0.7333，old 0.2667
0.70-0.79 lift_tilt -> insufficient_data：vote 0.8571，old 0.1429
>=0.60 decrease_a3 -> add_neighbor：样本少，但 vote 1.0，old 0.0
```

这说明 `conf50` 的 0.50 桶并不安全。更合理的二次整合是 `adj_only + KNN60 中 holdout 支持的 transition`，也就是 `v5_09`。

## 8. 为什么 C 从 check_server 转向 F2

Phase1 分布确实显示 F1 中应该有不少 `check_server`。

但当前 diagnostics 里，虽然 `check_server` 候选出现了 146 次，高置信 Phase1-like server case 暂时是 0。

也就是说：

```text
check_server 数量缺失是真的；
但现有证据还不足以知道该把哪些题改成 check_server。
```

如果为了配平直接硬改 40 个，很可能会把真实的 `pdcch_2sym` 或 `insufficient_data` 改坏。

所以 `check_server` 不再适合作为 C 的主任务。C 更应该看 F2/F3/F4/F4M 这些 V5 F1 修复没有触碰的区域。

目前多答案 sanity check 的结论：

```text
F3：cardinality=4，模式和 Phase1 一致，暂时干净。
F4：single-answer decrease_interfreq，和 Phase1 一致，暂时干净。
F4M：cardinality=2，两个 decrease_interfreq，无 Phase1 可校准，暂时保守不动。
F2：145 行里有 10 行结构违规，已经生成可复现修复。
```

所以对 `check_server` 的策略应该是：

- 暂时不提交；
- 不再作为 C 的主线；
- 只有出现明确 server/backhaul 证据时才重新打开。

## 9. 四个人为什么按得分假设分工

原来的分工是按技术分：

```text
一个人做 RAG
一个人做 prompt
一个人做 LoRA
一个人做规则
```

这个分工在前期适合探索系统能力。

但赛末几天，更重要的是每个人都能产出一个可提交、可归因的 CSV。否则大家都在改各自模块，最后分数涨跌无法归因。

所以现在按 0.20 冲刺目标分工：

```text
A：总控、提交、记录 public score
B：rerun contract patch + Phase1 smoke + post-sub1 attribution
C：F2 pair calibration
D：spot-check / audit
```

这不是否定 RAG/prompt/LoRA，而是把它们绑定到具体问题上。

例如：

- RAG / 规则负责人不是泛泛做检索系统，而是先修 rerun contract，再判断 KNN60 是否值得进入第 2 次提交；
- prompt / 规则负责人不是泛泛优化 prompt，而是收紧 F2/F1_5BS/F4M 的输出 contract；
- LoRA 负责人不是只看 override 触发率，而是回答：能否稳定改对几十个 F1 case？
- C/D 不是各自独立提交，而是给 A 的 Sub1/Sub2 集成提供证据和审计。

## 10. 每次测试要交付什么

每个人交付时，不要只说“我改了 RAG”或“我优化了 prompt”。

B 的 rerun contract patch 必须交付：

1. 修改了哪些文件；
2. F1/F1_5BS、F2、F4M guard 如何实现；
3. Phase1 smoke 结果；
4. invalid F2 outputs 是否为 0；
5. 是否建议 A 启动 cloud rerun。

C / D 的补丁或审计必须交付：

1. 如果有 CSV，必须是 600 行 submission CSV；
2. 改动了多少行；
3. 主要动作转移是什么；
4. F1/F2/F4M 分布或结构怎么变；
5. 为什么这些改动有依据；
6. 是否建议进入 Sub1 或 Sub2。

在两次提交约束下，B/C/D 不各自正式提交；他们的结果是给 A 的最终集成决策。只有 A 选择的文件才正式上传。

推荐记录到：

```text
SUBMISSION_BOARD.csv
```

## 11. 拿到 public score 后怎么优化

我们要用 public score 做闭环，但现在只有一次中途反馈机会。

0.20 主线流程：

```text
修 rerun contract
-> Phase1 smoke gate
-> patched cloud rerun
-> Sub1
-> 拿到 public score
-> B attribution
-> Sub2 修补或回退
```

举例 1：

```text
base:          0.1302
patched_rerun: 0.1700
```

结论：

- rerun 有明显收益；
- 但距离 0.20 仍有空间；
- Sub2 应该叠加 C 的 F2 patch 和 B 认可的 KNN60 / LLM boundary。

第 2 次提交策略：

- 从 Sub1 出发继续修；
- 不回退 V5；
- 只加 attribution 支持的补丁。

举例 2：

```text
base:          0.1302
patched_rerun: 0.1280
```

结论：

- rerun 方向失败或 API/LLM 非确定性污染太大；
- 不继续在 rerun 上修；
- 直接用 V5 safety net。

第 2 次提交策略：

- `submission_600_v5_11_baseline_plus_f2_pair_fix.csv`；
- 或 A/B 明确认为 V5 F1 结构仍值得保留时用 `v5_10`。

举例 3：

```text
patched_rerun: 0.1450
```

结论：

- rerun 有一点收益但不够；
- Sub2 只能做 targeted manual fixes；
- 不要再大规模引入低置信 KNN 或 check_server。

## 12. 赛末阶段的原则

### 原则一：不要混合太多假设

如果一个提交同时改：

- adjust；
- lift；
- check_server；
- add_neighbor；
- pdcch；
- F2；
- prompt；
- LoRA；

那么涨了也不知道是谁贡献，跌了也不知道是谁污染。

赛末更应该小步快跑。

但现在只剩两次提交，所以“小步快跑”主要发生在线下：先生成多个离线候选和报告，正式提交只选最有把握的 1-2 个。

### 原则二：API rerun 必须先过 smoke gate

全量重跑 API 会引入非确定性：

- 服务端数据可能漂移；
- tool 返回可能不稳定；
- LLM 输出可能变；
- 很难复现某个提交为什么涨分。

所以只有在 0.20 目标下才值得赌 patched rerun，并且必须先通过 Phase1 smoke gate。若 smoke / API / token 任一失败，就回到 V5 offline fallback。

### 原则三：方法必须服务于具体边界

RAG、prompt、LoRA 都不是目的。目的永远是：

```text
这个 ID 的答案应该从什么 C-code 改成什么 C-code？
这样改 public score 是否上涨？
```

### 原则四：不要为了分布配平硬改

Phase1 分布能提供方向，但不能当作硬约束。

例如 `check_server` 当前为 0 很异常，但不能因此直接改 40 个。必须找到真实证据。

## 13. V5 包里的文件怎么用

推荐先看：

```text
RERUN_SPRINT_PLAN.md
BCD_QUICK_START.md
TEAM_SPLIT.md
README_V5.md
SUBMISSION_BOARD.csv
reports/repair_report.md
```

主流水线文件：

```text
phase2_run.py
skills/candidate_generator.py
skills/feature_engine.py
skills/scenario_router.py
skills/api_data_gatherer.py
rag_retriever_v2.py
data/Phase_1/train.json
data/Phase_2/test.json
```

V5 fallback 候选提交：

```text
submissions/submission_600_v5_07_adj_only.csv
submissions/submission_600_v5_10_adj_only_plus_f2_pair_fix.csv
submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
submissions/submission_600_v5_12_adj_only_plus_knn60_plus_f2_pair_fix.csv
submissions/submission_600_v5_09_adj_only_plus_calibrated_knn60.csv
submissions/submission_600_v5_02_hardblock_adjust.csv
submissions/submission_600_v5_01_knn_conf50.csv
submissions/submission_600_v5_03_knn_all.csv
```

0.20 主线第 1 次正式提交：

```text
submissions/submission_600_v5_20_patched_rerun_sub1.csv
```

这个文件当前是计划名，需要 A 在 V5 中跑 patched `phase2_run.py` 后生成。

V5 fallback 第 1 次正式提交：

```text
submissions/submission_600_v5_07_adj_only.csv
```

patched rerun 低于 0.130 的 Sub2 fallback：

```text
submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv
```

格式检查：

```bash
/home/huangxiaolin/AITO/bin/python scripts/check_600_format.py submissions/submission_600_v5_07_adj_only.csv
```

离线复现：

```bash
/home/huangxiaolin/AITO/bin/python scripts/f1_distribution_repair.py
/home/huangxiaolin/AITO/bin/python scripts/C_f2_pair_repair.py
```

Phase1 离线校准示例：

```bash
/home/huangxiaolin/AITO/bin/python scripts/B_phase1_knn_calibration.py
```

patched rerun 的实际命令由 B/A 在修改 `phase2_run.py` 和 guard 后确认，运行目录必须是：

```text
/home/huangxiaolin/AITO/V5/Track A
```

## 14. 最终目标

V5 包当前的最终目标不是证明某个模块好，而是在 0.20 目标下组织一次高上限 rerun，同时保留一个干净 fallback。

当前最可能的主线是：

```text
patched rerun
-> F1/F1_5BS guard
-> F2 legal-pair guard
-> F4M decrease_interfreq guard
-> 2 audited LoRA overrides
-> C's F2 calibration if ready
-> Sub2 attribution repair or V5 fallback
```

等 public score 反馈回来后，再把有效规则合并回主系统。
