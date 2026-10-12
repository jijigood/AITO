# Agentic Overlay Test Report

Date: 2026-05-07

## Model / Network Guard

- Model endpoint: `http://127.0.0.1:8001/v1`
- Served model name: `qwen`
- Verified backing model path: `/home/huangxiaolin/AITO/model`
- Verified config: `model_type=qwen3_5_moe`, `architectures=Qwen3_5MoeForConditionalGeneration`
- Verified model README: `Qwen3.5-35B-A3B`
- Proxy guard used for live tests: unset `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, lowercase variants; set `NO_PROXY=120.46.145.77,124.71.227.61,127.0.0.1,localhost`

## High-Risk Probe

- IDs: `results/phase2/agentic_overlay_tests/high_risk_ids.txt`
- Output: `results/phase2/agentic_overlay_tests/high_risk_agentic_probe.csv`
- Diagnostics: `results/phase2/agentic_overlay_tests/high_risk_agentic_probe.diagnostics.jsonl`
- Audit: `results/phase2/agentic_overlay_tests/high_risk_agentic_probe.audit.json`
- Rows: 28 / 28 answered
- Audit: submit-ready partial, 0 hard errors, 0 warnings, 0 API errors
- Mean latency: 12.3 seconds/scenario
- Agentic path: 28 completed, 0 JSON/LLM failures
- Branches:
  - `lift_tilt_vs_insufficient_data`: 22
  - `add_neighbor_vs_decrease_a3`: 6
- Agentic answer changes vs deterministic: 0

## Representative Probe

- IDs: `results/phase2/probe_selection/probe_ids.txt`
- Output: `results/phase2/agentic_overlay_tests/probe_agentic_representative.csv`
- Diagnostics: `results/phase2/agentic_overlay_tests/probe_agentic_representative.diagnostics.jsonl`
- Audit: `results/phase2/agentic_overlay_tests/probe_agentic_representative.audit.json`
- Rows: 18 / 18 answered
- Families covered: F1, F1_5BS, F2, F3, F4, F4M
- Audit: submit-ready partial, 0 hard errors, 0 warnings, 0 API errors
- Mean latency: 1.6 seconds/scenario
- Agentic path: 1 completed, 17 skipped, 0 JSON/LLM failures
- Agentic answer changes vs deterministic: 0

## Interpretation

The controlled overlay is engineering-stable: it runs without formatting,
tool-call, proxy, or model-routing failures.  It currently behaves as a
confirmation layer rather than a correction layer, because all tested agentic
answers agreed with the deterministic backbone.

Next improvement should focus on making the overlay useful, not merely safe:

- Add trajectory RAG for hard branches.
- Make the prompt explicitly compare deterministic top candidate against
  plausible alternatives.
- Add a separate `F1_5BS` branch; current high-risk rules do not trigger on
  most F1_5BS rows.
- Let Person D diff future probes by `deterministic_answer`, `agentic_answer`,
  branch, diagnosis action, and tool evidence.
