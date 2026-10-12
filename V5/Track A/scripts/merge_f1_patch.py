"""
Merge 60-scenario patched results into v5_11 baseline to produce v5_20.
Usage:
  python3 scripts/merge_f1_patch.py \
    --base submissions/submission_600_v5_11_baseline_plus_f2_pair_fix.csv \
    --patch results/phase2/f1_adjaz_patch/result_60_patch.csv \
    --ids   data/f1_adjust_azimuth_60_ids.txt \
    --out   submissions/submission_600_v5_20_f1_guard_patch.csv
"""

import argparse
import csv
import sys

ADJUST_AZIMUTH = "C18"
LORA_OVERRIDES = {
    "6060dca7": "C22",  # must stay C22
    "1784b64c": "C20",  # must stay C20
}
F2_SINGLES = {
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
    "C11","C12","C13","C14","C15","C16","C17","C18","C19",
    "C20","C21","C22","C23","C24",
}
F1_SAFE = {"C1","C7","C9","C11","C12","C16","C22"}  # add_neighbor,decrease_a3,lift_tilt,insufficient_data,check_server,pdcch_2sym,press_down_tilt

def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    return headers, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.ids) as f:
        patch_ids = {line.strip() for line in f if line.strip()}
    print(f"Patch IDs loaded: {len(patch_ids)}")

    base_headers, base_rows = load_csv(args.base)
    _, patch_rows = load_csv(args.patch)

    patch_map = {}
    for r in patch_rows:
        sid = r["ID"]
        ans = r.get("Track A", "").strip()
        if sid and ans:
            patch_map[sid] = ans
    print(f"Patch answers loaded: {len(patch_map)}")

    changed = 0
    skipped_lora = 0
    still_adjaz = 0
    out_rows = []

    for r in base_rows:
        sid = r["ID"]
        base_ans = r.get("Track A", "").strip()
        new_r = dict(r)

        if sid in patch_ids:
            if sid in LORA_OVERRIDES:
                print(f"  SKIP LoRA override: {sid[:8]} keeps {LORA_OVERRIDES[sid]}")
                skipped_lora += 1
            elif sid in patch_map:
                new_ans = patch_map[sid]
                if new_ans == ADJUST_AZIMUTH:
                    print(f"  WARN still adjust_azimuth: {sid[:8]} {new_ans}")
                    still_adjaz += 1
                elif new_ans != base_ans:
                    print(f"  CHANGE: {sid[:8]} {base_ans} -> {new_ans}")
                    new_r["Track A"] = new_ans
                    changed += 1
                else:
                    print(f"  SAME:   {sid[:8]} {base_ans} (no change)")
            else:
                print(f"  MISSING in patch: {sid[:8]} (keeping base {base_ans})")

        out_rows.append(new_r)

    print(f"\nSummary: {changed} changed, {skipped_lora} LoRA-protected, {still_adjaz} still C18")
    print(f"Total rows: {len(out_rows)}")

    # Smoke checks
    track_a_rows = [r for r in out_rows if r.get("Track A", "").strip()]
    remaining_adjaz = sum(1 for r in track_a_rows if r.get("Track A","").strip() == ADJUST_AZIMUTH)
    print(f"Track A rows: {len(track_a_rows)}  (expected 500)")
    print(f"Remaining C18 (adjust_azimuth): {remaining_adjaz}")

    for sid_prefix, expected in LORA_OVERRIDES.items():
        match = [r for r in out_rows if r["ID"].startswith(sid_prefix)]
        if match:
            actual = match[0].get("Track A","").strip()
            status = "OK" if actual == expected else "FAIL"
            print(f"LoRA override {sid_prefix}: {actual} (expected {expected}) [{status}]")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_headers)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote: {args.out}")
    if remaining_adjaz > 0:
        print("WARNING: Some C18 remain — review before submitting")
    if len(track_a_rows) != 500:
        print("ERROR: Track A row count != 500")
        sys.exit(1)

if __name__ == "__main__":
    main()
