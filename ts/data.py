import json
import random
import re

from .common import read_jsonl, wpath, write_jsonl

XLAM_REPO = "Salesforce/xlam-function-calling-60k"
GLAIVE_REPO = "glaiveai/glaive-function-calling-v2"
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_CATS = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
AST_CATS = ["simple", "multiple", "parallel", "parallel_multiple"]

STRICT = (
    "You are a function calling assistant. When one of the provided functions can answer the request, "
    "respond only with tool calls that use the exact function and parameter names from the schema, with values "
    "of the correct types, and include every required parameter. When the request needs several calls, emit one "
    "tool call per call. When no provided function fits the request, answer in plain text and do not call any function."
)

TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "double": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "set": "array",
}


def norm_schema(s):
    if not isinstance(s, dict):
        return {}
    out = {}
    t = s.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if isinstance(t, str) and TYPE_MAP.get(t.lower()):
        out["type"] = TYPE_MAP[t.lower()]
    if isinstance(s.get("description"), str):
        out["description"] = s["description"]
    if isinstance(s.get("enum"), list) and s["enum"]:
        out["enum"] = s["enum"]
    props = s.get("properties")
    if isinstance(props, dict) and props:
        out["type"] = "object"
        out["properties"] = {k: norm_schema(v) for k, v in props.items()}
        req = [r for r in (s.get("required") or []) if r in out["properties"]]
        if req:
            out["required"] = req
    if out.get("type") == "array" and isinstance(s.get("items"), dict):
        out["items"] = norm_schema(s["items"])
    return out


def norm_params(p):
    s = norm_schema(p if isinstance(p, dict) else {})
    s["type"] = "object"
    s.setdefault("properties", {})
    return s


def strict_schema(s):
    out = {}
    for k in ("type", "enum"):
        if k in s:
            out[k] = s[k]
    if "properties" in s:
        out["properties"] = {k: strict_schema(v) for k, v in s["properties"].items()}
        if s.get("required"):
            out["required"] = list(s["required"])
        out["additionalProperties"] = False
    if "items" in s:
        out["items"] = strict_schema(s["items"])
    return out


def norm_tool(t):
    return {"name": str(t["name"]), "description": str(t.get("description") or ""), "parameters": norm_params(t.get("parameters"))}


def nname(n):
    return str(n).replace(".", "_").lower()


def xlam_type(t):
    t = str(t or "").strip()
    optional = "optional" in t.lower()
    base = re.sub(r",?\s*optional\s*$", "", t, flags=re.I).strip()
    m = re.match(r"^optional\[(.*)\]$", base, flags=re.I)
    if m:
        base, optional = m.group(1).strip(), True
    low = base.lower()
    if re.match(r"^(list|tuple|set)\b", low):
        sch = {"type": "array"}
        inner = re.match(r"^\w+\[(.*)\]$", base)
        if inner:
            it = xlam_type(inner.group(1))[0]
            if it.get("type"):
                sch["items"] = {"type": it["type"]}
        return sch, optional
    if low.startswith("dict"):
        return {"type": "object"}, optional
    mt = TYPE_MAP.get(low)
    return ({"type": mt} if mt else {}), optional


def xlam_tool(t):
    props, req, defaults = {}, [], {}
    params = t.get("parameters") or {}
    if isinstance(params, dict) and params.get("type") == "object" and "properties" in params:
        return norm_tool(t), {}
    for name, spec in params.items():
        spec = spec if isinstance(spec, dict) else {}
        sch, opt = xlam_type(spec.get("type"))
        if spec.get("description"):
            sch["description"] = str(spec["description"])
        props[name] = sch
        if "default" in spec and spec["default"] not in (None, ""):
            defaults[name] = spec["default"]
        if not opt and "default" not in spec:
            req.append(name)
    p = {"type": "object", "properties": props}
    if req:
        p["required"] = req
    return {"name": str(t["name"]), "description": str(t.get("description") or ""), "parameters": p}, defaults


def gold_from_calls(calls, tools, defaults=None):
    by = {t["name"]: t for t in tools}
    gold = []
    for c in calls:
        g = {k: [v] for k, v in c["arguments"].items()}
        d = (defaults or {}).get(c["name"], {})
        for p in by[c["name"]]["parameters"].get("properties", {}):
            if p not in g:
                g[p] = [""] + ([d[p]] if p in d else [])
        gold.append({c["name"]: g})
    return gold


def example(id_, source, category, tools, messages, gold, calls=None, answer_text=None):
    return {
        "id": id_,
        "source": source,
        "category": category,
        "tools": tools,
        "messages": messages,
        "gold": gold,
        "calls": calls,
        "answer_text": answer_text,
    }


def _xlam_all():
    from datasets import load_dataset

    ds = load_dataset(XLAM_REPO, split="train")
    out = []
    for i, r in enumerate(ds):
        try:
            raw_tools = json.loads(r["tools"])
            answers = json.loads(r["answers"])
            tools, defaults = [], {}
            for t in raw_tools:
                tool, d = xlam_tool(t)
                tools.append(tool)
                defaults[tool["name"]] = d
            names = {t["name"] for t in tools}
            calls = []
            for a in answers:
                args = a.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if a.get("name") not in names or not isinstance(args, dict):
                    raise ValueError("bad answer")
                calls.append({"name": a["name"], "arguments": args})
        except Exception:
            continue
        if not tools:
            continue
        msgs = [{"role": "user", "content": r["query"]}]
        out.append(example(f"xlam_{r.get('id', i)}", "xlam", "xlam", tools, msgs, gold_from_calls(calls, tools, defaults), calls))
    return out


def xlam_split(cfg):
    ev, tr = wpath(cfg, "data", "xlam_eval.jsonl"), wpath(cfg, "data", "xlam_train.jsonl")
    if ev.exists() and tr.exists():
        return read_jsonl(ev), read_jsonl(tr)
    items = _xlam_all()
    rng = random.Random(cfg["seed"])
    rng.shuffle(items)
    n = cfg["data"]["xlam_eval_n"]
    write_jsonl(ev, items[:n])
    write_jsonl(tr, items[n:])
    return items[:n], items[n:]


GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")


def glaive_tools(system):
    dec = json.JSONDecoder()
    tools = []
    i = system.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(system, i)
        except Exception:
            i = system.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "name" in obj:
            tools.append(norm_tool(obj))
        i = system.find("{", end)
    return tools


def glaive_turns(chat):
    turns, role = [], None
    for p in GLAIVE_SPLIT.split(chat):
        if p in ("USER:", "ASSISTANT:", "FUNCTION RESPONSE:"):
            role = p[:-1]
            continue
        if role:
            turns.append((role, p.replace("<|endoftext|>", "").strip()))
    return turns


def glaive_call(text):
    body = text[len("<functioncall>") :].strip()
    m = re.match(r'^\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}$', body, re.S)
    if m:
        name, args = m.group(1), json.loads(m.group(2))
    else:
        obj = json.loads(body)
        name, args = obj["name"], obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("arguments not a dict")
    return {"name": str(name), "arguments": args}


def glaive(cfg, limit=None):
    from datasets import load_dataset

    ds = load_dataset(GLAIVE_REPO, split="train")
    calls, nocalls = [], []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        try:
            tools = glaive_tools(r.get("system") or "")
            if not tools:
                continue
            turns = glaive_turns(r.get("chat") or "")
            if len(turns) < 2 or turns[0][0] != "USER" or turns[1][0] != "ASSISTANT" or not turns[0][1]:
                continue
            msgs = [{"role": "user", "content": turns[0][1]}]
            reply = turns[1][1]
            if reply.startswith("<functioncall>"):
                c = glaive_call(reply)
                if c["name"] not in {t["name"] for t in tools}:
                    continue
                calls.append(example(f"glaive_{i}", "glaive", "glaive_call", tools, msgs, gold_from_calls([c], tools), [c]))
            elif reply and "<functioncall>" not in reply:
                nocalls.append(example(f"glaive_{i}", "glaive", "glaive_nocall", tools, msgs, [], [], reply))
        except Exception:
            continue
    rng = random.Random(cfg["seed"])
    rng.shuffle(calls)
    rng.shuffle(nocalls)
    return calls, nocalls


def bfcl_files():
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(BFCL_REPO, repo_type="dataset")
    pat = re.compile(r"^BFCL_v(\d+)_(simple_python|simple|multiple|parallel_multiple|parallel|irrelevance)\.json$")
    pick = {}
    for f in files:
        m = pat.match(f)
        if not m:
            continue
        v, cat = int(m.group(1)), "simple" if m.group(2).startswith("simple") else m.group(2)
        if cat not in pick or v > pick[cat][0]:
            pick[cat] = (v, f)
    return {c: f for c, (v, f) in pick.items()}, set(files)


def _read_any(path):
    text = open(path).read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _bfcl_messages(q):
    if isinstance(q, str):
        return [{"role": "user", "content": q}]
    if q and isinstance(q[0], list):
        q = q[0]
    return [{"role": m["role"], "content": m["content"]} for m in q]


def bfcl(cfg):
    cache = wpath(cfg, "data", "bfcl.jsonl")
    if cache.exists():
        return read_jsonl(cache)
    from huggingface_hub import hf_hub_download

    picked, files = bfcl_files()
    missing = [c for c in BFCL_CATS if c not in picked]
    if missing:
        raise RuntimeError(f"BFCL categories not found in {BFCL_REPO}: {missing}")
    out = []
    for cat in BFCL_CATS:
        f = picked[cat]
        rows = _read_any(hf_hub_download(BFCL_REPO, f, repo_type="dataset"))
        answers = {}
        if cat != "irrelevance":
            ans_file = f"possible_answer/{f}"
            if ans_file not in files:
                raise RuntimeError(f"missing {ans_file} in {BFCL_REPO}")
            for a in _read_any(hf_hub_download(BFCL_REPO, ans_file, repo_type="dataset")):
                gt = a["ground_truth"]
                answers[a["id"]] = gt if isinstance(gt, list) else [gt]
        for r in rows:
            fns = r["function"] if isinstance(r["function"], list) else [r["function"]]
            tools = [norm_tool(fn) for fn in fns]
            gold = [] if cat == "irrelevance" else answers.get(r["id"])
            if gold is None:
                continue
            out.append(example(f"bfcl_{r['id']}", "bfcl", cat, tools, _bfcl_messages(r["question"]), gold))
    write_jsonl(cache, out)
    return out


def subset(items, n, seed):
    if not n or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in idx]


def eval_sets(cfg, mode):
    allb = bfcl(cfg)
    xe, _ = xlam_split(cfg)
    sets = {c: [e for e in allb if e["category"] == c] for c in BFCL_CATS}
    sets["xlam"] = xe
    if mode == "quick":
        sets = {k: subset(v, cfg["eval"]["quick_n"], cfg["seed"]) for k, v in sets.items()}
    elif cfg["eval"].get("full_n"):
        sets = {k: subset(v, cfg["eval"]["full_n"], cfg["seed"]) for k, v in sets.items()}
    return sets


def tools_for_template(tools):
    return [{"type": "function", "function": t} for t in tools]


def chat_prompt(tok, ex, strict=False):
    msgs = [dict(m) for m in ex["messages"]]
    if strict:
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = STRICT + "\n\n" + msgs[0]["content"]
        else:
            msgs = [{"role": "system", "content": STRICT}] + msgs
    return tok.apply_chat_template(msgs, tools=tools_for_template(ex["tools"]), tokenize=False, add_generation_prompt=True)


def format_call(c):
    return '<tool_call>\n{"name": "' + c["name"] + '", "arguments": ' + json.dumps(c["arguments"], ensure_ascii=False) + "}\n</tool_call>"


def format_calls(calls):
    return "\n".join(format_call(c) for c in calls)


def target_text(ex):
    return format_calls(ex["calls"]) if ex["calls"] else (ex["answer_text"] or "")
