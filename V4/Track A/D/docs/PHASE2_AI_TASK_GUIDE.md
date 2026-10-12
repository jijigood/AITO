# Phase 2 AI Task Guide

本文件用于给后续参与开发的 AI 一个统一入口，避免每次任务都从零理解项目边界。

## 目标

- 让 AI 在接任务前先读对文档，再读对代码。
- 让不同 Owner 的任务边界保持稳定，减少越界修改。
- 统一调试环境、依赖安装位置和交付方式。

## AI 开始任务前的必读顺序

1. 先读 `docs/PHASE2_TEAM_DIVISION.md`
2. 如果更适合中文，再读 `docs/PHASE2_TEAM_DIVISION_CN.MD`
3. 再读当前任务对应的 Owner skill 文档
4. 最后再读当前 Owner 的主代码文件和调用链

如果用户没有明确指定 Owner，先根据 `PHASE2_TEAM_DIVISION` 判断任务属于 A、B、C、D 哪一方，再进入对应 skill 文档。

## 当前 skill 文档位置

- Owner D:
  `docs/owner_d/OWNER_D_SKILL.md`

## 当前任务包位置

- Owner D:
  `docs/owner_d/README.md`

如果未来补充 A/B/C 的 skill 文档，优先沿用同样命名方式：

- `docs/owner_a/OWNER_A_SKILL.md`
- `docs/owner_b/OWNER_B_SKILL.md`
- `docs/owner_c/OWNER_C_SKILL.md`

如果某个 Owner 的 skill 文档还不存在，先读 `PHASE2_TEAM_DIVISION` 和相关代码，再按最小修改原则执行任务。

## 环境与依赖规则

- 所有调试、测试、脚本运行、数据构建、索引构建，都必须在 conda 环境 `tele` 中进行。
- 推荐命令前缀：

```bash
source /home/jett/miniconda3/etc/profile.d/conda.sh && conda activate tele && <command>
```

- 需要安装新依赖时，必须安装到 `tele` 环境。
- 不要把新依赖装到 `base`、系统 Python 或其它虚拟环境。
- 安装完新依赖后，要在任务结果中说明：
  - 安装了什么
  - 为什么需要
  - 是通过 `pip` 还是 `conda` 装到 `tele`

## Owner 级任务入口

### A: Integration + Fine-tuning + Submission

先读：

- `docs/PHASE2_TEAM_DIVISION.md`
- `phase2_run.py`
- `build_sft_v3.py`
- `curate_sft.py`
- `eval_ckpt.py`
- `phase2_audit.py`

关注点：

- 端到端是否可运行
- B/C/D 产物是否被正确接入
- probe、audit、submission 是否稳定

### B: Skills + Tools

先读：

- `docs/PHASE2_TEAM_DIVISION.md`
- `skills/feature_engine.py`
- `skills/candidate_generator.py`
- `skills/tool_caller.py`
- `skills/agentic_overlay.py`

关注点：

- branch trigger
- tool menu
- evidence schema
- root cause / action 到候选项的映射稳定性

### C: Prompt + CoT + Agentic Loop

先读：

- `docs/PHASE2_TEAM_DIVISION.md`
- `skills/agentic_overlay.py`
- 与 prompt 相关的文档或模板

关注点：

- JSON 输出稳定性
- tool-use 决策是否受控
- D 提供的 trajectory snippets 如何注入

### D: Trajectory RAG + Memory

先读：

- `docs/PHASE2_TEAM_DIVISION.md`
- `docs/owner_d/OWNER_D_SKILL.md`
- `docs/owner_d/README.md`
- `rag_retriever_v2.py`
- `phase2_run.py`
- `build_sft_v3.py`
- `eval_ckpt.py`

关注点：

- 基于现有 `rag_retriever_v2.py` 演进
- retrieval 是否 deterministic
- prompt 文本是否泄露训练答案
- 是否兼容 A/C 的调用方式

## 统一执行规则

- 先确认任务属于哪个 Owner，再动代码。
- 优先修改该 Owner 的主责任文件，避免无边界扩散。
- 如果改动影响其它 Owner 的输入输出契约，必须在结果里明确说明。
- 没有必要时，不要新建平行实现。
- 优先兼容现有调用链，而不是只新增一个“理论上更好”的接口。

## 调试与验证

- 调试命令默认在 `tele` 环境中执行。
- 任何测试、脚本运行、索引构建都应尽量附带最小验证结果。
- 如果因为环境、数据或权限原因无法完成验证，必须明确说明未验证项。

## 文档与交付

- 任务完成后，至少说明：
  - 改了什么
  - 为什么改
  - 验证了什么
  - 哪些地方还有风险或未验证
- 如果新增依赖或新增产物路径，必须写清楚。
- 如果任务明确属于某个 Owner，结果描述中应写明本次按哪个 Owner 的职责执行。
