import shutil

from .common import POLICIES, banner, merged_dir, read_json, repo_id, resolve_model, results_path, wpath, write_json

HELPER = '''import json

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


def strict_schema(s):
    out = {k: s[k] for k in ("type", "enum") if k in s}
    if "properties" in s:
        out["properties"] = {k: strict_schema(v) for k, v in s["properties"].items()}
        if s.get("required"):
            out["required"] = list(s["required"])
        out["additionalProperties"] = False
    if "items" in s:
        out["items"] = strict_schema(s["items"])
    return out


def structural_tag(tools):
    structures = []
    for t in tools:
        fn = t.get("function", t)
        structures.append(
            {
                "begin": '<tool_call>\\n{"name": "' + json.dumps(fn["name"])[1:-1] + '", "arguments": ',
                "schema": strict_schema(fn.get("parameters", {"type": "object", "properties": {}})),
                "end": "}\\n</tool_call>",
            }
        )
    return json.dumps({"structures": structures, "triggers": ["<tool_call>"]})


def generate(llm, messages, tools, max_tokens=768):
    tok = llm.get_tokenizer()
    tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
    prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=GuidedDecodingParams(structural_tag=structural_tag(tools)))
    return llm.generate([prompt], sp)[0].outputs[0].text


if __name__ == "__main__":
    import sys

    llm = LLM(sys.argv[1] if len(sys.argv) > 1 else ".")
    tools = [{"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}]
    print(generate(llm, [{"role": "user", "content": "What is the weather in Paris in celsius?"}], tools))
'''


def pick_source(cfg):
    forced = cfg.get("constrained", {}).get("source")
    if forced:
        return forced
    scored = []
    for name in POLICIES:
        r = read_json(results_path(cfg, name, "full"))
        if r:
            scored.append((r["summary"]["overall"], name))
    if not scored:
        return "sft"
    scored.sort(reverse=True)
    print("candidates (overall, name):", scored, flush=True)
    return scored[0][1]


def build_constrained(cfg):
    src = pick_source(cfg)
    path = resolve_model(cfg, src)
    out = merged_dir(cfg, "constrained")
    if out.exists():
        shutil.rmtree(out)
    banner(f"constrained model: copying weights of {src} from {path}")
    shutil.copytree(path, out, ignore=shutil.ignore_patterns("README.md", "adapter", "training", "eval", "assets", ".cache"))
    (out / "toolsure_constrained.py").write_text(HELPER)
    src_res = read_json(results_path(cfg, src, "full"), {})
    write_json(wpath(cfg, "models", "constrained", "pick.json"), {"source": src, "source_repo": repo_id(cfg, src), "source_overall": src_res.get("summary", {}).get("overall")})
