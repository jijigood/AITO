# Owner D Deliverables

任务文档保留在 `docs/owner_d/`，正式交付物统一放在 `artifacts/owner_d/`。

## Data

- `artifacts/owner_d/data/trajectory_examples.jsonl`
- `artifacts/owner_d/data/trajectory_probe_memory.jsonl`
- `artifacts/owner_d/data/trajectory_memory_combined.jsonl`

## Index

- `artifacts/owner_d/index/trajectory_index.faiss`

## Docs

- `artifacts/owner_d/docs/trajectory_schema.md`
- `artifacts/owner_d/docs/trajectory_schema.json`
- `artifacts/owner_d/docs/trajectory_prompt_snippets.md`
- `artifacts/owner_d/docs/trajectory_coverage_report.md`
- `artifacts/owner_d/docs/task06_integration_validation.md`

## Notes

- The code still accepts legacy root-level artifact paths for backward compatibility.
- Smoke tests must use an explicit `--out` path outside the default artifacts path when a sample file is desired.
