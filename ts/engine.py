import json
import statistics
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .data import strict_schema


def structural_tag(tools):
    structures = [
        {
            "begin": '<tool_call>\n{"name": "' + json.dumps(t["name"])[1:-1] + '", "arguments": ',
            "schema": strict_schema(t["parameters"]),
            "end": "}\n</tool_call>",
        }
        for t in tools
    ]
    return json.dumps({"structures": structures, "triggers": ["<tool_call>"]})


class Engine:
    def __init__(self, cfg, model, gpu_util=None, max_model_len=None):
        self.cfg = cfg
        self.model = model
        self.max_model_len = max_model_len or cfg["gen"]["max_model_len"]
        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            seed=cfg["seed"],
            gpu_memory_utilization=gpu_util or cfg["gen"]["gpu_memory_utilization"],
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
        )
        self.tok = self.llm.get_tokenizer()

    def greedy(self, max_tokens):
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def sample(self, max_tokens, temperature):
        return SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)

    def constrained(self, ex, max_tokens):
        return SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=GuidedDecodingParams(structural_tag=structural_tag(ex["tools"])))

    def fits(self, prompt, max_tokens):
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"]) + max_tokens <= self.max_model_len

    def run(self, prompts, sp, progress=True):
        t = time.time()
        outs = self.llm.generate(prompts, sp, use_tqdm=progress)
        dt = time.time() - t
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs], dt

    def _one(self, o):
        return (o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason)

    def tag_ok(self, ex):
        import xgrammar as xgr

        try:
            s = json.loads(structural_tag(ex["tools"]))
            items = [xgr.StructuralTagItem(begin=x["begin"], schema=json.dumps(x["schema"]), end=x["end"]) for x in s["structures"]]
            xgr.Grammar.from_structural_tag(items, s["triggers"])
            return True
        except Exception:
            return False

    def run_constrained(self, prompts, exs, max_tokens, progress=True):
        t = time.time()
        sps, fallback = [], 0
        for ex in exs:
            if self.tag_ok(ex):
                sps.append(self.constrained(ex, max_tokens))
            else:
                fallback += 1
                sps.append(self.greedy(max_tokens))
        if fallback:
            print(f"{fallback} prompts had schemas xgrammar could not compile, decoded without constraints", flush=True)
        outs = self.llm.generate(prompts, sps, use_tqdm=progress) if prompts else []
        return [self._one(o) for o in outs], time.time() - t, fallback

    def latency(self, prompts, exs, max_tokens, constrained=False):
        times, toks = [], []
        for p, ex in zip(prompts, exs):
            t = time.time()
            if constrained:
                r, _, _ = self.run_constrained([p], [ex], max_tokens, progress=False)
            else:
                r, _ = self.run([p], self.greedy(max_tokens), progress=False)
            times.append(time.time() - t)
            toks.append(r[0][1])
        s = sorted(times)
        return {
            "n": len(times),
            "p50_s": statistics.median(times),
            "p95_s": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
            "mean_s": statistics.mean(times),
            "decode_tok_s": sum(toks) / max(1e-9, sum(times)),
        }
