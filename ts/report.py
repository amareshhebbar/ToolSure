from huggingface_hub import HfApi

from .card import results_table
from .common import BASELINES, MODELS, banner, dataset_repo, read_json, repo_id, results_path, wpath, write_json


def collect(cfg):
    out = {}
    for tag in [f"base_{v}" for v in BASELINES] + ["reference"] + MODELS:
        r = read_json(results_path(cfg, tag, "full"))
        if r:
            out[tag] = r
    return out


def label(tag):
    return tag.replace("base_", "base ") if tag.startswith("base_") or tag == "reference" else tag.upper()


def plots(cfg, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tags = list(res)
    x = np.arange(len(tags))
    keys = [("ast_mean", "BFCL AST mean"), ("irrelevance", "Irrelevance"), ("xlam", "xLAM held out")]
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(tags)), 4.8))
    for i, (k, name) in enumerate(keys):
        ax.bar(x + (i - 1) * w, [res[t]["summary"].get(k, 0) * 100 for t in tags], w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([label(t) for t in tags], rotation=30, ha="right")
    ax.set_ylabel("Accuracy %")
    ax.set_title("ToolSure: function calling accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "scores.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(tags)), 4.8))
    ax.bar(x - 0.2, [(1 - res[t]["summary"]["parse_rate"]) * 100 for t in tags], 0.4, label="Broken call rate (JSON did not parse)")
    ax.bar(x + 0.2, [res[t]["summary"]["halluc_rate"] * 100 for t in tags], 0.4, label="Hallucinated function rate")
    ax.set_xticks(x)
    ax.set_xticklabels([label(t) for t in tags], rotation=30, ha="right")
    ax.set_ylabel("% (lower is better)")
    ax.set_title("ToolSure: call reliability")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "reliability.png"), dpi=140)
    plt.close(fig)


def run_report(cfg):
    from .hub import login, publish

    login()
    res = collect(cfg)
    if not res:
        raise RuntimeError("no full results to report")
    plots(cfg, res)
    summary = {t: {**r["summary"], "cost_per_1k_queries_usd": r.get("cost_per_1k_queries_usd"), "latency": r.get("latency")} for t, r in res.items()}
    write_json(wpath(cfg, "report", "summary.json"), summary)
    wpath(cfg, "report", "table.html").write_text(results_table(cfg))
    api = HfApi()
    for f in ["summary.json", "table.html", "scores.png", "reliability.png"]:
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f"report/{f}", repo_id=dataset_repo(cfg), repo_type="dataset", commit_message=f"Add report {f}")
    for m in MODELS:
        if read_json(results_path(cfg, m, "full")):
            publish(cfg, m)
    banner("report done\n" + "\n".join(f"{k}: overall {v['overall']:.3f} ast {v.get('ast_mean', 0):.3f} irrelevance {v.get('irrelevance', 0):.3f} parse {v['parse_rate']:.3f}" for k, v in summary.items()))
    for m in MODELS:
        print(f"https://huggingface.co/{repo_id(cfg, m)}", flush=True)
