import json
import shutil

from huggingface_hub import HfApi, snapshot_download

from .common import banner, read_json, repo_id, results_path, wpath, write_json


def verify_model(cfg, name):
    from transformers import AutoConfig, AutoTokenizer

    from .evaluate import quick_eval, variant_for
    from .hub import login

    login()
    repo = repo_id(cfg, name)
    banner(f"verifying {repo} straight from the hub")
    files = HfApi().list_repo_files(repo)
    checks = {
        "config.json": "config.json" in files,
        "tokenizer": "tokenizer.json" in files or "tokenizer_config.json" in files,
        "safetensors": any(f.endswith(".safetensors") and "/" not in f for f in files),
        "README.md": "README.md" in files,
    }
    if name == "constrained":
        checks["toolsure_constrained.py"] = "toolsure_constrained.py" in files
    for k, v in checks.items():
        print(f"  {k}: {'OK' if v else 'MISSING'}", flush=True)
    if not all(checks.values()):
        raise SystemExit(f"verify {name}: repo is missing files")
    local = wpath(cfg, "verify", name)
    if local.exists():
        shutil.rmtree(local)
    try:
        snapshot_download(repo, local_dir=str(local), ignore_patterns=["adapter/*", "training/*", "assets/*", "eval/*"], force_download=True)
        conf = AutoConfig.from_pretrained(str(local))
        tok = AutoTokenizer.from_pretrained(str(local))
        print(f"  model_type {conf.model_type}, eos {tok.eos_token}, vocab {len(tok)}", flush=True)
        res = quick_eval(cfg, str(local), name, variant_for(name))
    finally:
        shutil.rmtree(local, ignore_errors=True)
    base = read_json(results_path(cfg, "base_prompted", "quick"))
    mine = res["summary"]
    gates = {"parse_rate": mine["parse_rate"] >= cfg["eval"]["gate_parse_rate"]}
    if base:
        gates["overall"] = mine["overall"] >= base["summary"]["overall"] - cfg["eval"]["gate_overall_drop"]
    report = {"repo": repo, "quick": mine, "base_quick": base["summary"] if base else None, "gates": gates, "passed": all(gates.values())}
    write_json(wpath(cfg, "results", f"{name}_verify.json"), report)
    banner(f"verify {name}\n" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(f"verify {name} FAILED, see {wpath(cfg, 'results', name + '_verify.json')}")
    print(f"verify {name} PASSED", flush=True)
