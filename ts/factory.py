import copy
import json
import random
from collections import Counter
from pathlib import Path

from .common import banner, dataset_repo, read_jsonl, wpath, write_json, write_jsonl
from .data import chat_prompt, format_calls, glaive, target_text, xlam_split
from .grade import grade

TEXT_REPLY = "Sure, I can help with that. Let me look that up for you."


def sample_value(schema, rng):
    t = schema.get("type")
    if schema.get("enum"):
        return rng.choice(schema["enum"])
    return {"integer": 5, "number": 1.5, "boolean": True, "array": ["item"], "object": {"key": "value"}}.get(t, "example")


def corrupt(ex, rng):
    tools = ex["tools"]
    if not ex["calls"]:
        t = rng.choice(tools)
        props = t["parameters"].get("properties", {})
        args = {p: sample_value(s, rng) for p, s in list(props.items())[:2]}
        return format_calls([{"name": t["name"], "arguments": args}]), "call_when_not_needed"
    by = {t["name"]: t for t in tools}
    kinds = ["wrong_function", "hallucinated_function", "missing_argument", "wrong_type", "extra_argument", "broken_json", "text_instead_of_call"]
    rng.shuffle(kinds)
    for kind in kinds:
        calls = copy.deepcopy(ex["calls"])
        i = rng.randrange(len(calls))
        c = calls[i]
        if kind == "wrong_function":
            others = [n for n in by if n != c["name"]]
            if not others:
                continue
            c["name"] = rng.choice(others)
        elif kind == "hallucinated_function":
            c["name"] = rng.choice(["get_", "fetch_", "search_"]) + c["name"] + rng.choice(["_info", "_v2", "_data"])
        elif kind == "missing_argument":
            req = [k for k in by[c["name"]]["parameters"].get("required", []) if k in c["arguments"]]
            if not req:
                continue
            del c["arguments"][rng.choice(req)]
        elif kind == "wrong_type":
            keys = [k for k, v in c["arguments"].items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
            if not keys:
                continue
            k = rng.choice(keys)
            v = c["arguments"][k]
            c["arguments"][k] = str(v) if isinstance(v, (int, float)) else [v]
        elif kind == "extra_argument":
            props = by[c["name"]]["parameters"].get("properties", {})
            name = next(n for n in ["verbose", "user_id", "max_results", "extra_options"] if n not in props and n not in c["arguments"])
            c["arguments"][name] = rng.choice([True, "auto", 10])
        elif kind == "broken_json":
            text = format_calls(calls)
            j = text.rfind("}\n</tool_call>")
            return (text[:j] + "\n</tool_call>" if rng.random() < 0.5 else text.replace('"', "'")), kind
        elif kind == "text_instead_of_call":
            return TEXT_REPLY, kind
        text = format_calls(calls)
        if not grade(ex, text)["correct"]:
            return text, kind
    return TEXT_REPLY, "text_instead_of_call"


def build_pool(cfg, tok):
    d = cfg["data"]
    rng = random.Random(cfg["seed"])
    _, xtrain = xlam_split(cfg)
    xtrain = list(xtrain)
    rng.shuffle(xtrain)
    gcall, gno = glaive(cfg)
    pool = xtrain[: d["xlam_train_n"]] + gcall[: d["glaive_call_n"]] + gno[: d["glaive_nocall_n"]]
    rng.shuffle(pool)
    kept = []
    for ex in pool:
        p = chat_prompt(tok, ex)
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        if n <= d["max_prompt_tokens"]:
            ex["prompt"] = p
            ex["prompt_tokens"] = n
            kept.append(ex)
    print(f"pool: {len(kept)} of {len(pool)} examples within {d['max_prompt_tokens']} prompt tokens", flush=True)
    return kept


def build_sets(cfg):
    from transformers import AutoTokenizer

    d = cfg["data"]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    eos = tok.eos_token
    pool = build_pool(cfg, tok)
    sft = [
        {
            "id": ex["id"],
            "source": ex["source"],
            "category": ex["category"],
            "prompt": ex["prompt"],
            "completion": target_text(ex) + eos,
            "tools_json": json.dumps(ex["tools"]),
            "gold_json": json.dumps(ex["gold"]),
        }
        for ex in pool
    ]
    rng = random.Random(cfg["seed"] + 7)
    pair_src = rng.sample(pool, min(d["pairs_n"], len(pool)))
    onpolicy = {}
    base_acc = None
    if d.get("onpolicy_negatives"):
        from .engine import Engine

        eng = Engine(cfg, cfg["base_model"])
        res, _ = eng.run([ex["prompt"] for ex in pair_src], eng.sample(512, d["onpolicy_temperature"]))
        ok = 0
        for ex, (text, ntok, fr) in zip(pair_src, res):
            g = grade(ex, text)
            ok += g["correct"]
            if not g["correct"] and text.strip():
                onpolicy[ex["id"]] = text
        base_acc = ok / max(1, len(pair_src))
        print(f"base model on pair prompts: acc {base_acc:.3f}, {len(onpolicy)} on policy negatives", flush=True)
    pairs = []
    for ex in pair_src:
        if ex["id"] in onpolicy:
            rej, kind = onpolicy[ex["id"]], "onpolicy"
        else:
            rej, kind = corrupt(ex, rng)
        pairs.append(
            {
                "id": ex["id"],
                "source": ex["source"],
                "category": ex["category"],
                "prompt": ex["prompt"],
                "chosen": target_text(ex) + eos,
                "rejected": rej + eos,
                "negative_type": kind,
            }
        )
    prompts = [
        {
            "id": ex["id"],
            "source": ex["source"],
            "category": ex["category"],
            "prompt": ex["prompt"],
            "prompt_tokens": ex["prompt_tokens"],
            "gold_json": json.dumps(ex["gold"]),
            "tool_names_json": json.dumps([t["name"] for t in ex["tools"]]),
        }
        for ex in pool
    ]
    xe, _ = xlam_split(cfg)
    ev = [{"id": ex["id"], "messages_json": json.dumps(ex["messages"]), "tools_json": json.dumps(ex["tools"]), "gold_json": json.dumps(ex["gold"])} for ex in xe]
    stats = {
        "pool": len(pool),
        "sft_rows": len(sft),
        "pair_rows": len(pairs),
        "prompt_rows": len(prompts),
        "eval_xlam_rows": len(ev),
        "sources": dict(Counter(ex["category"] for ex in pool)),
        "negative_types": dict(Counter(p["negative_type"] for p in pairs)),
        "base_acc_on_pair_prompts": base_acc,
        "mean_prompt_tokens": sum(ex["prompt_tokens"] for ex in pool) / max(1, len(pool)),
    }
    for name, rows in [("sft", sft), ("pairs", pairs), ("prompts", prompts), ("eval_xlam", ev)]:
        write_jsonl(wpath(cfg, "data", f"{name}.jsonl"), rows)
    write_json(wpath(cfg, "data", "stats.json"), stats)
    banner("factory stats\n" + "\n".join(f"{k}: {v}" for k, v in stats.items()))
    return stats


def dataset_card_body(cfg, stats):
    src = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats["sources"].items())
    neg = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats["negative_types"].items())
    return f"""
# ToolSure Calls ({cfg['base_label']})

Function calling data used to train every model in the ToolSure collection, rendered in the native `{cfg['base_model']}` tool calling chat template.

## Configs

<table>
<tr><th>Config</th><th>Rows</th><th>What it holds</th></tr>
<tr><td>sft</td><td>{stats['sft_rows']}</td><td>Prompt with tools and the correct tool calls, or a plain text answer when no tool fits</td></tr>
<tr><td>pairs</td><td>{stats['pair_rows']}</td><td>Chosen is the correct answer; rejected is a real mistake made by the base model when it made one, otherwise a synthetic broken call</td></tr>
<tr><td>prompts</td><td>{stats['prompt_rows']}</td><td>Prompts with gold calls for GRPO with a verifiable reward</td></tr>
<tr><td>eval_xlam</td><td>{stats['eval_xlam_rows']}</td><td>Held out xLAM examples, never used for training</td></tr>
</table>

## Sources

<table>{src}</table>

## Negative types in pairs

<table>{neg}</table>

Sources: `Salesforce/xlam-function-calling-60k` (CC BY 4.0) and `glaiveai/glaive-function-calling-v2` (Apache 2.0). Tool schemas were converted to JSON Schema. Glaive conversations where tools were offered but the assistant answered in text are kept as examples of correctly not calling a tool.
"""


def push_dataset(cfg, stats):
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    from .hub import ensure_collection, login

    login()
    api = HfApi()
    repo = dataset_repo(cfg)
    api.create_repo(repo, repo_type="dataset", private=cfg["private"], exist_ok=True)
    keys = {"sft": "sft_rows", "pairs": "pair_rows", "prompts": "prompt_rows", "eval_xlam": "eval_xlam_rows"}
    for name in keys:
        rows = read_jsonl(wpath(cfg, "data", f"{name}.jsonl"))
        banner(f"pushing {name}: {len(rows)} rows to {repo}")
        Dataset.from_list(rows).push_to_hub(repo, config_name=name, private=cfg["private"], commit_message=f"Add {name} config")
    readme = Path(hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)).read_text()
    front = ""
    if readme.startswith("---"):
        front = readme[: readme.find("\n---", 3) + 4]
    api.upload_file(path_or_fileobj=(front + "\n" + dataset_card_body(cfg, stats)).encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset", commit_message="Update dataset card")
    api.upload_file(path_or_fileobj=str(wpath(cfg, "data", "stats.json")), path_in_repo="stats.json", repo_id=repo, repo_type="dataset", commit_message="Add stats")
    slug = ensure_collection(cfg)
    api.add_collection_item(slug, repo, "dataset", note="Training data shared by every model in this collection", exists_ok=True)
    banner("verifying dataset from the hub")
    for name, key in keys.items():
        ds = load_dataset(repo, name, split="train", download_mode="force_redownload")
        if len(ds) != stats[key]:
            raise RuntimeError(f"{name}: hub has {len(ds)} rows, expected {stats[key]}")
        print(f"{name}: {len(ds)} rows OK", flush=True)


def run_factory(cfg):
    stats = build_sets(cfg)
    if stats["sft_rows"] < 100 or stats["pair_rows"] < 50:
        raise RuntimeError(f"too little data: {stats}")
    push_dataset(cfg, stats)
