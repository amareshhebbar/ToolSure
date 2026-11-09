from pathlib import Path

from .common import BASELINES, MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath

METHOD = {
    "sft": "LoRA supervised finetuning in bf16 on correct tool calls, plus examples where the right answer is plain text because no offered tool fits.",
    "qlora": "QLoRA: the same supervised finetuning, but the frozen base weights are loaded in 4 bit NF4 with double quantization while the LoRA adapter trains in bf16. The adapter was then merged into the bf16 base.",
    "dora": "DoRA: weight decomposed LoRA. Each adapted weight is split into a magnitude vector and a direction, and LoRA only updates the direction, which usually tracks full finetuning more closely at the same rank.",
    "orpo": "ORPO from the base model in one stage: supervised loss on the correct answer plus an odds ratio penalty against a broken answer. Broken answers are real mistakes the base model made on that prompt when it made one, otherwise synthetic corruptions (wrong function, invented function, missing argument, wrong type, extra argument, broken JSON, text instead of a call, or a call when none was needed).",
    "grpo": "Online GRPO with a verifiable reward and no reward model. Each completion is parsed and checked against the gold calls: broken JSON scores minus 0.5, a valid call earns 0.2 plus up to 0.3 for picking the right functions plus 0.5 for exactly right arguments, correctly not calling scores 1.0, and invented function names are penalized. Uses the Dr. GRPO loss.",
    "constrained": "The same weights as the best trained ToolSure model, served with grammar constrained decoding. Whenever the model opens a tool call, decoding is forced to one of the offered function names and to arguments that match that function's JSON schema, so calls always parse. The model can still answer in plain text when no tool fits.",
}

BASE_LABEL = {"prompted": "prompted", "strict": "strict system prompt", "constrained": "constrained decoding"}
COLS = [
    ("Simple", "simple"),
    ("Multiple", "multiple"),
    ("Parallel", "parallel"),
    ("Parallel multiple", "parallel_multiple"),
]


def _rows(cfg):
    rows = []
    base_mmlu = read_json(wpath(cfg, "results", "base_mmlu.json"), {}).get("mmlu_acc")
    for v in BASELINES:
        r = read_json(results_path(cfg, f"base_{v}", "full"))
        if r:
            rows.append((f"base_{v}", f"{cfg['base_label']} ({BASE_LABEL[v]})", r, base_mmlu))
    r = read_json(results_path(cfg, "reference", "full"))
    if r:
        rows.append(("reference", f"{cfg['eval']['reference_model'].split('/')[-1]} (prompted, reference)", r, None))
    for m in MODELS:
        r = read_json(results_path(cfg, m, "full"))
        if r:
            rows.append((m, f"ToolSure {m.upper()}", r, read_json(wpath(cfg, "results", f"{m}_mmlu.json"), {}).get("mmlu_acc")))
    return rows


def _pct(x):
    return "" if x is None else f"{x * 100:.1f}"


def _num(x, fmt="{:.0f}"):
    return "" if x is None else fmt.format(x)


def results_table(cfg, highlight=None):
    rows = _rows(cfg)
    if not rows:
        return "<p>Evaluation results are added after the full evaluation step.</p>"
    head = (
        "<tr><th>Model</th>"
        + "".join(f"<th>BFCL {c}</th>" for c, _ in COLS)
        + "<th>BFCL AST mean</th><th>Irrelevance</th><th>xLAM held out</th><th>JSON parse rate</th>"
        "<th>Hallucinated function rate</th><th>MMLU</th><th>Latency p50 (s)</th><th>Cost per 1k queries (USD)</th></tr>"
    )
    body = []
    for tag, label, r, mmlu in rows:
        b, s = r["benchmarks"], r["summary"]
        cells = [label] + [_pct(b.get(k, {}).get("acc")) for _, k in COLS]
        cells += [
            _pct(s.get("ast_mean")),
            _pct(s.get("irrelevance")),
            _pct(s.get("xlam")),
            _pct(s.get("parse_rate")),
            _pct(s.get("halluc_rate")),
            _pct(mmlu),
            _num(r.get("latency", {}).get("p50_s"), "{:.2f}"),
            _num(r.get("cost_per_1k_queries_usd"), "{:.4f}"),
        ]
        if tag == highlight:
            cells = [f"<b>{c}</b>" for c in cells]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>\n" + head + "\n" + "\n".join(body) + "\n</table>"


def hyperparams(cfg, name):
    if name == "constrained":
        info = read_json(wpath(cfg, "models", "constrained", "pick.json"), {})
        return "<table>\n" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in info.items()) + "\n</table>"
    t = dict(cfg["train"][name])
    t.update({f"lora_{k}": v for k, v in cfg["lora"].items()})
    if name == "grpo":
        t["loss_type"] = "dr_grpo"
        t["scale_rewards"] = False
    if name == "qlora":
        t["quantization"] = "nf4, double quant, bf16 compute"
    if name == "dora":
        t["use_dora"] = True
    return "<table>\n" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in t.items()) + "\n</table>"


def lineage(cfg, name):
    if name == "constrained":
        info = read_json(wpath(cfg, "models", "constrained", "pick.json"), {})
        return [info.get("source_repo") or cfg["base_model"]], None
    init = cfg["train"][name].get("init", "base")
    return [cfg["base_model"] if init == "base" else repo_id(cfg, init)], "finetune"


USAGE_PLAIN = """```python
import json
from vllm import LLM, SamplingParams

llm = LLM("{repo}")
tok = llm.get_tokenizer()
tools = [{{"type": "function", "function": {{"name": "get_weather", "description": "Current weather for a city", "parameters": {{"type": "object", "properties": {{"city": {{"type": "string"}}}}, "required": ["city"]}}}}}}]
messages = [{{"role": "user", "content": "What is the weather in Paris?"}}]
prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
print(llm.generate([prompt], SamplingParams(temperature=0, max_tokens=512))[0].outputs[0].text)
```

Tool calls come back in the standard Qwen2.5 format, one `<tool_call>{{...}}</tool_call>` block per call, so any server that parses Qwen2.5 tool calls (for example vLLM with `--tool-call-parser hermes`) works unchanged."""

USAGE_CONSTRAINED = """```python
from huggingface_hub import hf_hub_download
import importlib.util
from vllm import LLM

path = hf_hub_download("{repo}", "toolsure_constrained.py")
spec = importlib.util.spec_from_file_location("toolsure_constrained", path)
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

llm = LLM("{repo}")
tools = [{{"name": "get_weather", "description": "Current weather for a city", "parameters": {{"type": "object", "properties": {{"city": {{"type": "string"}}}}, "required": ["city"]}}}}]
print(tc.generate(llm, [{{"role": "user", "content": "What is the weather in Paris?"}}], tools))
```

`toolsure_constrained.py` builds an xgrammar structural tag from the offered tools: free text is allowed, but once the model writes `<tool_call>` the rest of the call is forced to a real function name and schema valid arguments."""


def card(cfg, name):
    bases, relation = lineage(cfg, name)
    repo = repo_id(cfg, name)
    slug = get_meta(cfg, "collection_slug")
    coll = f"https://huggingface.co/collections/{slug}" if slug else f"https://huggingface.co/{cfg['hf_user']}"
    rel = f"base_model_relation: {relation}\n" if relation else ""
    imgs = "".join(f"\n![{p}](assets/{p}.png)\n" for p in ("scores", "reliability") if Path(wpath(cfg, "report", f"{p}.png")).exists())
    usage = (USAGE_CONSTRAINED if name == "constrained" else USAGE_PLAIN).format(repo=repo)
    weights = "Same weights as the source model listed below." if name == "constrained" else "LoRA adapter merged into full bf16 safetensors; the adapter itself is in `adapter/`."
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
{chr(10).join('  - ' + b for b in bases)}
{rel}datasets:
  - {dataset_repo(cfg)}
language:
  - en
tags:
  - toolsure
  - function_calling
  - tool_use
  - qwen2.5
  - {name}
---

# ToolSure {name.upper()} ({cfg['base_label']})

Small models break function calls: malformed JSON, invented function names, missing or mistyped arguments, and calls when no tool fits. Retrieval cannot fix that, because the failure is in how the model writes the call. ToolSure trains `{cfg['base_model']}` to call tools reliably. This model is one of six in the [ToolSure collection]({coll}): five training methods on the same data, plus constrained decoding, all on one table.

## Method

{METHOD[name]}

Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}). {weights}

## Why finetune instead of prompting

The table includes the base model prompted normally, the base model with a strict system prompt, and the base model with grammar constrained decoding. Constrained decoding guarantees valid JSON and real function names, but it cannot choose the right function, fill the right values, or decide when not to call. That gap is what training closes.

## Results

All rows use greedy decoding and the same checker. BFCL columns are AST accuracy on the Berkeley Function Calling Leaderboard non live categories, scored with a reimplementation of the BFCL AST checker (function name, required parameters, no unexpected parameters, values against the possible answers; parallel calls matched in any order). Irrelevance is the share of prompts where the model correctly made no call. xLAM held out is exact call accuracy on {cfg['data']['xlam_eval_n']} xLAM examples never used in training. JSON parse rate counts outputs that attempted a call. Latency is batch size 1 on vLLM; cost assumes ${cfg['gpu_cost_per_hour']}/hour GPU at the measured batched throughput.

{results_table(cfg, highlight=name)}
{imgs}
## Training hyperparameters

{hyperparams(cfg, name)}

## Usage

{usage}

## Files

The repository root holds bf16 safetensors. `adapter/` holds the LoRA adapter where one was trained, `training/` holds logs and curves, and `eval/` holds raw results and sample generations.
"""


def write_card(cfg, name, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(card(cfg, name))
