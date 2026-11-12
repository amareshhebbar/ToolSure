import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from ts.common import MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath, write_json

COLS = [("simple", "Simple"), ("multiple", "Multiple"), ("parallel", "Parallel"), ("parallel_multiple", "Parallel multiple")]


def pct(x):
    return "" if x is None else f"{x * 100:.1f}%"


def row(r):
    s, b = r["summary"], r["benchmarks"]
    out = {k: s.get(k) for k in ("ast_mean", "irrelevance", "xlam", "overall", "parse_rate", "halluc_rate")}
    out.update({c: b.get(c, {}).get("acc") for c, _ in COLS})
    out["latency_p50_s"] = r.get("latency", {}).get("p50_s")
    return out


def build(cfg):
    base = read_json(results_path(cfg, "base_prompted", "full"))
    if not base:
        raise SystemExit("base_prompted results missing, run the baseline step first")
    rows = {"base_prompted": row(base)}
    for tag in ["base_strict", "base_constrained", "reference"] + MODELS:
        r = read_json(results_path(cfg, tag, "full"))
        if r:
            rows[tag] = row(r)
    trained = [m for m in MODELS if m in rows]
    if not trained:
        raise SystemExit("no model results yet, run the pipeline first")
    best = max(trained, key=lambda m: rows[m]["overall"])
    return rows, best


def label(cfg, tag):
    names = {"base_prompted": f"{cfg['base_label']} prompted", "base_strict": f"{cfg['base_label']} with a strict system prompt", "base_constrained": f"{cfg['base_label']} with constrained decoding", "reference": f"{(cfg['eval'].get('reference_model') or 'reference').split('/')[-1].replace('-', ' ')} prompted (bigger model)"}
    return names.get(tag, f"ToolSure {tag.upper()}")


def sentence(cfg, rows, best):
    b, v = rows["base_prompted"], rows[best]
    s = (
        f"ToolSure {best.upper()} raises {cfg['base_label']} tool call accuracy on the BFCL benchmark from {pct(b['ast_mean'])} to {pct(v['ast_mean'])}, "
        f"correct \"no tool needed\" answers from {pct(b['irrelevance'])} to {pct(v['irrelevance'])}, and held out xLAM accuracy from {pct(b['xlam'])} to {pct(v['xlam'])}"
    )
    if b["halluc_rate"] and b["halluc_rate"] > 0:
        cut = (1 - v["halluc_rate"] / b["halluc_rate"]) * 100
        s += f", while cutting invented function names by {cut:.0f}%"
    ref = rows.get("reference")
    if ref:
        word = "beating" if v["overall"] > ref["overall"] else "against"
        s += f" ({word} {label(cfg, 'reference').split(' prompted')[0]} at {pct(ref['overall'])} overall versus {pct(v['overall'])})"
    return s + "."


def markdown(cfg, rows, best):
    coll = get_meta(cfg, "collection_slug")
    head = "| Model | " + " | ".join(n for _, n in COLS) + " | BFCL mean | No tool needed | xLAM held out | Overall | Valid JSON | Invented functions |"
    lines = [
        f"# ToolSure results ({cfg['base_label']})",
        "",
        f"**Headline:** {sentence(cfg, rows, best)}",
        "",
        "BFCL is the Berkeley Function Calling Leaderboard (non live categories), scored with a reimplementation of its AST checker: right function, required parameters present, no unexpected parameters, values matching the allowed answers, parallel calls in any order. "
        "\"No tool needed\" is the BFCL irrelevance set, where the right answer is to call nothing. xLAM held out is exact call accuracy on xLAM examples never used in training. Every row uses greedy decoding and the same checker.",
        "",
        head,
        "|" + "---|" * (len(COLS) + 7),
    ]
    order = ["base_prompted", "base_strict", "base_constrained", "reference"] + sorted([m for m in MODELS if m in rows], key=lambda m: -rows[m]["overall"])
    for tag in [t for t in order if t in rows]:
        v = rows[tag]
        name = label(cfg, tag)
        if tag in MODELS:
            name = f"[{name}](https://huggingface.co/{repo_id(cfg, tag)})"
        if tag == best:
            name = f"**{name}**"
        cells = [name] + [pct(v[c]) for c, _ in COLS] + [pct(v["ast_mean"]), pct(v["irrelevance"]), pct(v["xlam"]), pct(v["overall"]), pct(v["parse_rate"]), pct(v["halluc_rate"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "The headline model is the trained model with the highest overall score (mean of BFCL, no tool needed and xLAM).",
        "",
        "## Check it yourself",
        "",
        f"* Every model card has the full table and training settings: [collection](https://huggingface.co/collections/{coll})" if coll else "* Every model card has the full table and training settings.",
        f"* Raw evaluation files and sample outputs: [{best.upper()} eval](https://huggingface.co/{repo_id(cfg, best)}/tree/main/eval).",
        f"* Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}).",
        "* Code to reproduce every number: [github.com/amareshhebbar/ToolSure](https://github.com/amareshhebbar/ToolSure).",
        "",
        f"Generated {time.strftime('%Y-%m-%d')} from the pipeline's own result files.",
        "",
    ]
    return "\n".join(lines)


def main():
    from huggingface_hub import HfApi

    from ts.common import load_cfg
    from ts.hub import login

    cfg = load_cfg()
    rows, best = build(cfg)
    for tag, v in rows.items():
        print(f"{tag:18s} BFCL {pct(v['ast_mean']):>7s}  no tool {pct(v['irrelevance']):>7s}  xLAM {pct(v['xlam']):>7s}  overall {pct(v['overall']):>7s}  valid JSON {pct(v['parse_rate']):>7s}  invented {pct(v['halluc_rate']):>7s}")
    md = markdown(cfg, rows, best)
    wpath(cfg, "report", "HEADLINE.md").write_text(md)
    write_json(wpath(cfg, "report", "headline.json"), {"best": best, "sentence": sentence(cfg, rows, best), "rows": rows})
    print("\n" + sentence(cfg, rows, best) + "\n")
    if "--local" in sys.argv:
        print(f"written to {wpath(cfg, 'report', 'HEADLINE.md')}, not uploaded")
        return
    login()
    api = HfApi()
    ds = dataset_repo(cfg)
    for f in ["HEADLINE.md", "headline.json"]:
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f"report/{f}", repo_id=ds, repo_type="dataset", commit_message=f"Add {f}")
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f, repo_id=repo_id(cfg, best), commit_message=f"Add {f}")
    print("proof links:")
    print(f"  https://huggingface.co/{repo_id(cfg, best)}/blob/main/HEADLINE.md")
    print(f"  https://huggingface.co/datasets/{ds}/blob/main/report/HEADLINE.md")


if __name__ == "__main__":
    main()
