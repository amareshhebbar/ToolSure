import glob
import json
import os
import shutil
import subprocess
import sys
import time

from .common import BASELINES, banner, read_json, resolve_model, results_path, wpath, write_json, write_jsonl
from .data import AST_CATS, chat_prompt, eval_sets, subset
from .grade import grade


def metrics(rows):
    n = len(rows)
    att = [r for r in rows if r["attempted"]]
    return {
        "n": n,
        "acc": sum(r["correct"] for r in rows) / max(1, n),
        "call_rate": len(att) / max(1, n),
        "n_attempted": len(att),
        "n_parse_ok": sum(r["parse_ok"] for r in att),
        "parse_rate": sum(r["parse_ok"] for r in att) / len(att) if att else 1.0,
        "n_calls": sum(r["n_calls"] for r in rows),
        "n_halluc": sum(r["halluc"] for r in rows),
        "halluc_rate": sum(r["halluc"] for r in rows) / max(1, sum(r["n_calls"] for r in rows)),
        "tokens": sum(r["ntok"] for r in rows),
        "mean_tokens": sum(r["ntok"] for r in rows) / max(1, n),
        "skipped_too_long": sum(r.get("skipped", False) for r in rows),
    }


def summarize(b):
    s = {}
    ast = [b[c]["acc"] for c in AST_CATS if c in b]
    if ast:
        s["ast_mean"] = sum(ast) / len(ast)
    if "irrelevance" in b:
        s["irrelevance"] = b["irrelevance"]["acc"]
    if "xlam" in b:
        s["xlam"] = b["xlam"]["acc"]
    parts = [s[k] for k in ("ast_mean", "irrelevance", "xlam") if k in s]
    s["overall"] = sum(parts) / len(parts)
    att = sum(m["n_attempted"] for m in b.values())
    s["parse_rate"] = sum(m["n_parse_ok"] for m in b.values()) / att if att else 1.0
    calls = sum(m["n_calls"] for m in b.values())
    s["halluc_rate"] = sum(m["n_halluc"] for m in b.values()) / max(1, calls)
    s["mean_tokens"] = sum(m["tokens"] for m in b.values()) / max(1, sum(m["n"] for m in b.values()))
    return s


def run_suite(cfg, eng, tag, variant, mode):
    e = cfg["eval"]
    mt = e["max_tokens"]
    sets = eval_sets(cfg, mode)
    out = {"tag": tag, "model": eng.model, "variant": variant, "mode": mode, "benchmarks": {}}
    dump, total_tok, total_time = [], 0, 0.0
    for bname, items in sets.items():
        prompts = [chat_prompt(eng.tok, ex, strict=variant == "strict") for ex in items]
        keep = [i for i, p in enumerate(prompts) if eng.fits(p, mt)]
        banner(f"[{tag} | {variant} | {mode}] {bname}: {len(keep)} of {len(items)} prompts")
        kp, kx = [prompts[i] for i in keep], [items[i] for i in keep]
        fallback = 0
        if variant == "constrained":
            res, dt, fallback = eng.run_constrained(kp, kx, mt)
        else:
            res, dt = eng.run(kp, eng.greedy(mt))
        got = dict(zip(keep, res))
        rows = []
        for i, ex in enumerate(items):
            if i in got:
                text, ntok, fr = got[i]
                g = grade(ex, text)
                g.update({"ntok": ntok, "finish": fr})
            else:
                text = ""
                g = {"attempted": False, "parse_ok": True, "n_calls": 0, "halluc": 0, "correct": False, "ntok": 0, "skipped": True}
            rows.append(g)
            if sum(d["bench"] == bname for d in dump) < 12:
                dump.append({"bench": bname, "id": ex["id"], "gold": ex["gold"], "correct": g["correct"], "text": text})
        m = metrics(rows)
        m["wall_s"] = dt
        m["constraint_fallback"] = fallback
        out["benchmarks"][bname] = m
        total_tok += m["tokens"]
        total_time += dt
        print(json.dumps({k: m[k] for k in ("n", "acc", "call_rate", "parse_rate", "halluc_rate", "mean_tokens")}), flush=True)
    out["summary"] = summarize(out["benchmarks"])
    out["throughput_tok_s"] = total_tok / max(1e-9, total_time)
    queries = sum(m["n"] for m in out["benchmarks"].values())
    out["cost_per_1k_queries_usd"] = total_time / max(1, queries) * 1000 / 3600 * cfg["gpu_cost_per_hour"]
    if mode == "full" and e.get("latency_n"):
        lat = subset(sets["simple"], e["latency_n"], cfg["seed"] + 1)
        lp = [chat_prompt(eng.tok, ex, strict=variant == "strict") for ex in lat]
        banner(f"[{tag} | {variant}] latency at batch size 1 on {len(lp)} prompts")
        out["latency"] = eng.latency(lp, lat, mt, constrained=variant == "constrained")
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(results_path(cfg, tag, mode), out)
    write_jsonl(wpath(cfg, "results", f"{tag}_{mode}_samples.jsonl"), dump)
    banner(f"[{tag}] summary {json.dumps(out['summary'])}")
    return out


def run_baseline(cfg):
    from .engine import Engine

    eng = Engine(cfg, cfg["base_model"])
    run_suite(cfg, eng, "base_prompted", "prompted", "quick")
    for v in BASELINES:
        tag = f"base_{v}"
        if read_json(results_path(cfg, tag, "full")):
            print(f"skip {tag}, already evaluated", flush=True)
            continue
        run_suite(cfg, eng, tag, v, "full")


def run_reference(cfg):
    ref = cfg["eval"].get("reference_model")
    if not ref:
        print("no reference_model in config, skipping", flush=True)
        return
    from .engine import Engine

    eng = Engine(cfg, ref)
    run_suite(cfg, eng, "reference", "prompted", "full")


def variant_for(name):
    return "constrained" if name == "constrained" else "prompted"


def quick_eval(cfg, model_path, tag, variant="prompted"):
    from .engine import Engine

    eng = Engine(cfg, model_path)
    return run_suite(cfg, eng, tag, variant, "quick")


def run_model_eval(cfg, name):
    from .engine import Engine

    eng = Engine(cfg, resolve_model(cfg, name))
    return run_suite(cfg, eng, name, variant_for(name), "full")


def run_mmlu(cfg, name):
    if not cfg["eval"].get("mmlu"):
        print("mmlu disabled in config", flush=True)
        return
    tag = "base" if name == "base" else name
    if name == "constrained":
        info = read_json(wpath(cfg, "models", "constrained", "pick.json"), {})
        src = wpath(cfg, "results", f"{info.get('source')}_mmlu.json")
        if src.exists():
            shutil.copy(src, wpath(cfg, "results", "constrained_mmlu.json"))
            print(f"constrained uses the same weights as {info.get('source')}, copied its MMLU result", flush=True)
            return
    path = resolve_model(cfg, name)
    out_dir = wpath(cfg, "results", f"mmlu_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    margs = (
        f"pretrained={path},dtype=bfloat16,gpu_memory_utilization={cfg['gen']['gpu_memory_utilization']},"
        f"max_model_len={cfg['eval']['mmlu_max_model_len']},seed={cfg['seed']}"
    )
    lm_eval_bin = os.path.join(os.path.dirname(sys.executable), "lm_eval")
    cmd = [lm_eval_bin, "--model", "vllm", "--model_args", margs, "--tasks", "mmlu", "--batch_size", "auto", "--output_path", str(out_dir)]
    banner("running " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True), key=os.path.getmtime)
    if not files:
        raise RuntimeError("lm_eval produced no results file")
    res = json.loads(open(files[-1]).read())["results"]["mmlu"]
    acc = res.get("acc,none", res.get("acc"))
    write_json(wpath(cfg, "results", f"{tag}_mmlu.json"), {"tag": tag, "mmlu_acc": acc, "raw": res})
    banner(f"[{tag}] MMLU acc {acc:.4f}")
