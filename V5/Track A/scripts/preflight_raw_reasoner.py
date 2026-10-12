"""
Preflight: measure base-Qwen raw-evidence reasoner latency/feasibility.
STRICT NO-LEAKAGE: prompt contains ONLY task text + options + raw API tables.
No deterministic answer, no candidates, no reasons, no feature scores,
no top_issue, no RAG, no Phase-1 subtypes.
"""
import json, time, sys
import httpx

PHASE1 = "/home/huangxiaolin/AITO/V5/Track A/data/Phase_1/train.json"
URL = "http://127.0.0.1:8001/v1/chat/completions"

def fam(s):
    tag = s.get("tag", ""); single = "single" in tag.lower()
    ctx = s.get("context", {}); wi = ctx.get("wireless_network_information", {}) if isinstance(ctx, dict) else {}
    try: n = int(str(wi.get("num_base_stations", "4")).strip())
    except: n = 4
    return ({4: "F1", 5: "F1_5BS", 13: "F4"} if single else {4: "F2", 5: "F3", 13: "F4M"}).get(n, "?")

def build_prompt(s):
    task = s.get("task", {})
    desc = task.get("description", "")
    opts = task.get("options", [])
    opt_txt = "\n".join(f"{o.get('id')}: {o.get('label')}" for o in opts)
    d = s.get("data", {})
    ev = ""
    for k in ["user_plane_data", "network_configuration_data",
              "signaling_plane_data", "traffic_data", "mr_data"]:
        v = d.get(k)
        if v:
            ev += f"\n## {k}\n{v}\n"
    is_multi = "multiple" in s.get("tag", "").lower()
    fmt = ("Select 2-4 options. Final line MUST be exactly \\boxed{C3|C7} style."
           if is_multi else
           "Select exactly ONE option. Final line MUST be exactly \\boxed{C7} style.")
    user = (
        f"You are a 5G RAN troubleshooting expert. Diagnose ONLY from the raw "
        f"measurement evidence below. Reason step by step, then answer.\n\n"
        f"## Task\n{desc}\n\n## Options\n{opt_txt}\n\n## Raw Evidence{ev}\n\n{fmt}"
    )
    return user, is_multi

def extract_boxed(t):
    import re
    m = re.findall(r"\\boxed\{([^}]*)\}", t or "")
    return m[-1].strip() if m else ""

def iou(p, g):
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    ps, gs = set(p.upper().split("|")), set(g.upper().split("|"))
    return len(ps & gs) / len(ps | gs)

def main():
    data = json.load(open(PHASE1))
    picks = {}
    for s in data:
        f = fam(s)
        if f in ("F1", "F2", "F3", "F4") and f not in picks:
            picks.setdefault(f, []).append(s)
        if f == "F1" and len(picks.get("F1", [])) < 2:
            picks["F1"].append(s)
    sample = (picks.get("F1", [])[:2] + picks.get("F2", [])[:1]
              + picks.get("F3", [])[:1] + picks.get("F4", [])[:1])

    client = httpx.Client(trust_env=False, timeout=400.0)
    results = []
    for s in sample:
        f = fam(s); gt = str(s.get("answer", ""))
        prompt, is_multi = build_prompt(s)
        t0 = time.time()
        try:
            r = client.post(URL, json={
                "model": "qwen",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048, "temperature": 0,
            })
            dt = time.time() - t0
            j = r.json()
            msg = j["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            ans = extract_boxed(content) or extract_boxed(reasoning)
            usage = j.get("usage", {})
            sc = iou(ans, gt)
            rec = {"fam": f, "sid": s["scenario_id"][:8], "latency_s": round(dt, 1),
                   "prompt_tok": usage.get("prompt_tokens"),
                   "completion_tok": usage.get("completion_tokens"),
                   "finish": j["choices"][0].get("finish_reason"),
                   "ans": ans, "gt": gt, "iou": round(sc, 2),
                   "parsed": bool(ans)}
        except Exception as e:
            rec = {"fam": f, "sid": s["scenario_id"][:8],
                   "latency_s": round(time.time() - t0, 1), "error": str(e)[:120]}
        results.append(rec)
        print(json.dumps(rec), flush=True)

    ok = [r for r in results if "latency_s" in r and not r.get("error")]
    if ok:
        avg = sum(r["latency_s"] for r in ok) / len(ok)
        mx = max(r["latency_s"] for r in ok)
        parsed = sum(1 for r in ok if r.get("parsed"))
        print(f"\n=== PREFLIGHT SUMMARY ===")
        print(f"avg_latency={avg:.1f}s  max={mx:.1f}s  parseable={parsed}/{len(ok)}")
        for n in (750,):  # 500 Phase-2 + ~250 harness
            print(f"extrapolated {n} scenarios @avg: {n*avg/3600:.1f}h  @max: {n*mx/3600:.1f}h")
    json.dump(results, open(
        "/home/huangxiaolin/AITO/V5/Track A/results/preflight_raw_reasoner.json", "w"), indent=2)

if __name__ == "__main__":
    main()
