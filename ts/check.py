import importlib
import json
import platform
import shutil
import subprocess
import sys
import time
import uuid

PARTS = ["sys", "hub", "data", "vllm", "train", "grpo"]

PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "trl",
    "vllm",
    "xgrammar",
    "huggingface_hub",
    "lm_eval",
    "safetensors",
    "yaml",
    "matplotlib",
    "hf_transfer",
]

WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"],
    },
}
STOCK = {
    "name": "get_stock_price",
    "description": "Get the latest stock price for a ticker",
    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
}


def ok(msg):
    print(f"  [PASS] {msg}", flush=True)


def fail(msg):
    print(f"  [FAIL] {msg}", flush=True)
    raise SystemExit(1)


def toy_examples():
    from .data import example, gold_from_calls

    tools = [WEATHER, STOCK]
    out = []
    for i, city in enumerate(["Paris", "Tokyo", "Lima", "Oslo"]):
        calls = [{"name": "get_weather", "arguments": {"city": city}}]
        out.append(example(f"toy_{i}", "toy", "toy", tools, [{"role": "user", "content": f"What is the weather in {city}?"}], gold_from_calls(calls, tools), calls))
    out.append(example("toy_nocall", "toy", "toy", tools, [{"role": "user", "content": "Tell me a short joke."}], [], [], "Why did the scarecrow win an award? He was outstanding in his field."))
    return out


def check_sys(cfg):
    import torch

    print(f"  python {platform.python_version()} at {sys.executable}", flush=True)
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            ok(f"import {name} {getattr(m, '__version__', '')}")
        except Exception as e:
            fail(f"import {name}: {e!r}")
    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    ok(f"CUDA available, torch CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)")
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 2**30
    ok(f"GPU 0: {p.name}, {vram:.1f} GiB, compute capability {p.major}.{p.minor}")
    if p.major < 8 or not torch.cuda.is_bf16_supported():
        fail("bf16 needs an Ampere or newer GPU (A100, A6000, L40S, H100, RTX 3090/4090)")
    ok("bf16 supported")
    if vram < 40:
        print(f"  [WARN] {vram:.0f} GiB VRAM. Configs target 48 GiB. Lower micro_batch and grpo per_device_batch.", flush=True)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    ok(f"bf16 matmul {20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOPS")
    free = shutil.disk_usage(cfg["paths"]["work"]).free / 2**30
    print(f"  config {cfg['config_path']}", flush=True)
    print(f"  work dir {cfg['paths']['work']}, free disk {free:.0f} GiB", flush=True)
    if free < 60:
        print("  [WARN] under 60 GiB free. Six models of about 3.1 GiB, a 7B reference model and caches need room.", flush=True)
    else:
        ok("disk space")


def check_hub(cfg):
    from huggingface_hub import HfApi

    from .hub import login

    try:
        login()
    except Exception as e:
        fail(f"login: {e}")
    api = HfApi()
    me = api.whoami()
    name = me.get("name")
    ok(f"logged in as {name}")
    if name != cfg["hf_user"] and cfg["hf_user"] not in [o.get("name") for o in me.get("orgs", [])]:
        fail(f"token belongs to {name} but config hf_user is {cfg['hf_user']}. Set HF_USER")
    role = me.get("auth", {}).get("accessToken", {}).get("role")
    print(f"  token role: {role}", flush=True)
    if role == "read":
        fail("token is read only. Create a write token at huggingface.co/settings/tokens")
    tag = uuid.uuid4().hex[:6]
    repo = f"{cfg['hf_user']}/toolsure_write_test_{tag}"
    try:
        api.create_repo(repo, private=True)
        api.upload_file(path_or_fileobj=b"ok", path_in_repo="probe.txt", repo_id=repo, commit_message="write probe")
        if "probe.txt" not in api.list_repo_files(repo):
            fail("uploaded file not visible")
        ok(f"create repo and upload file ({repo})")
        c = api.create_collection(title=f"toolsure write test {tag}", namespace=cfg["hf_user"], private=True)
        api.add_collection_item(c.slug, repo, "model", note="probe")
        ok("create collection and add item")
        api.delete_collection(c.slug)
        ok("delete test collection")
    finally:
        try:
            api.delete_repo(repo)
            ok("delete test repo")
        except Exception as e:
            print(f"  [WARN] could not delete {repo}: {e}", flush=True)


def check_data(cfg):
    from transformers import AutoTokenizer

    from . import data
    from .grade import grade, reward

    try:
        xe, xt = data.xlam_split(cfg)
    except Exception as e:
        fail(
            f"load {data.XLAM_REPO}: {e!r}\n"
            f"  This dataset is gated. Open https://huggingface.co/datasets/{data.XLAM_REPO} while logged in as {cfg['hf_user']}, accept the terms, then rerun."
        )
    ok(f"xLAM: {len(xe)} eval, {len(xt)} train pool, sample tool {xt[0]['tools'][0]['name']}")
    try:
        gc, gn = data.glaive(cfg, limit=6000)
    except Exception as e:
        fail(f"load {data.GLAIVE_REPO}: {e!r}")
    if not gc or not gn:
        fail(f"Glaive parser found {len(gc)} call and {len(gn)} no call examples in the first 6000 rows")
    ok(f"Glaive parser: {len(gc)} call and {len(gn)} no call examples from the first 6000 rows")
    try:
        b = data.bfcl(cfg)
    except Exception as e:
        fail(f"load BFCL: {e!r}")
    counts = {c: sum(e["category"] == c for e in b) for c in data.BFCL_CATS}
    if any(v == 0 for v in counts.values()):
        fail(f"BFCL categories empty: {counts}")
    ok(f"BFCL: {counts}")
    for ex in xt[:200]:
        if not grade(ex, data.format_calls(ex["calls"]))["correct"]:
            fail(f"grader rejects gold for {ex['id']}: {ex['calls']}")
    toy = toy_examples()
    if not grade(toy[0], data.format_calls(toy[0]["calls"]))["correct"]:
        fail("grader rejects a correct call")
    if grade(toy[0], '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>')["correct"]:
        fail("grader accepts a wrong argument")
    if grade(toy[0], '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"\n</tool_call>')["parse_ok"]:
        fail("grader accepts broken JSON")
    if not grade(toy[4], "Here is a joke.")["correct"] or grade(toy[4], data.format_calls(toy[0]["calls"]))["correct"]:
        fail("grader mishandles a prompt that needs no call")
    if not reward(toy[0]["gold"], ["get_weather"], data.format_calls(toy[0]["calls"])) == 1.0:
        fail("reward for a perfect call is not 1.0")
    ok("grader and reward on gold, wrong argument, broken JSON and no call cases")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    ex = toy[0]
    prompt = data.chat_prompt(tok, ex)
    full = tok.apply_chat_template(
        ex["messages"] + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": c} for c in ex["calls"]]}],
        tools=data.tools_for_template(ex["tools"]),
        tokenize=False,
    )
    mine = prompt + data.format_calls(ex["calls"]) + tok.eos_token
    if not full.rstrip().startswith(mine.rstrip()) and full.rstrip() != mine.rstrip():
        print(f"  [WARN] training target differs from the chat template rendering\n  template: {full[len(prompt):]!r}\n  ours:     {mine[len(prompt):]!r}", flush=True)
    else:
        ok("training target matches the Qwen2.5 tool call template exactly")
    ok(f"eos token {tok.eos_token}, prompt tokens for the toy example {len(tok(prompt)['input_ids'])}")


def check_vllm(cfg):
    from .data import bfcl, chat_prompt
    from .engine import Engine
    from .grade import grade

    t = time.time()
    eng = Engine(cfg, cfg["base_model"], gpu_util=0.5, max_model_len=4096)
    ok(f"vLLM loaded {cfg['base_model']} in {time.time() - t:.0f}s")
    b = bfcl(cfg)
    exs = [e for e in b if e["category"] == "simple"][:6] + [e for e in b if e["category"] == "parallel"][:3] + [e for e in b if e["category"] == "irrelevance"][:3]
    prompts = [chat_prompt(eng.tok, ex) for ex in exs]
    res, dt = eng.run(prompts, eng.greedy(512))
    g = [grade(ex, r[0]) for ex, r in zip(exs, res)]
    print(f"  prompted: {sum(x['correct'] for x in g)}/{len(g)} correct in {dt:.1f}s", flush=True)
    print("  sample output:\n" + res[0][0][:400], flush=True)
    cres, cdt, fb = eng.run_constrained(prompts, exs, 512)
    cg = [grade(ex, r[0]) for ex, r in zip(exs, cres)]
    print(f"  constrained: {sum(x['correct'] for x in cg)}/{len(cg)} correct in {cdt:.1f}s, {fb} fell back to unconstrained", flush=True)
    print("  sample constrained output:\n" + cres[0][0][:400], flush=True)
    if fb == len(exs):
        fail("no structural tag compiled, constrained decoding is not working in this build")
    if fb:
        print(f"  [WARN] {fb} prompt schemas could not be compiled and ran unconstrained", flush=True)
    bad = [ex["id"] for ex, x in zip(exs, cg) if x["attempted"] and not x["parse_ok"]]
    if bad:
        fail(f"constrained outputs did not parse: {bad}")
    if any(x["halluc"] for x in cg):
        fail("constrained decoding produced an unknown function name")
    ok("vLLM prompted and constrained generation, every constrained call parses")


def check_train(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .common import merged_dir
    from .data import chat_prompt, target_text
    from .train import build_model, collate, encode, orpo_loss, save_merged, sft_loss

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    toy = toy_examples()
    prompt = chat_prompt(tok, toy[0])
    good = target_text(toy[0]) + tok.eos_token
    bad = '<tool_call>\n{"name": "get_weather_info", "arguments": {"town": "Paris"}}\n</tool_call>' + tok.eos_token
    tc = {"logits_chunk": 256, "lam": 0.1}
    for mode in ["lora", "qlora", "dora"]:
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, cfg["base_model"], mode)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        batch = collate([encode(tok, prompt, good)], pad)
        losses = []
        for _ in range(4):
            loss, _ = sft_loss(model, batch, tc)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        print(f"  {mode} sft losses {[round(x, 4) for x in losses]}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if not losses[-1] < losses[0]:
            fail(f"{mode} SFT loss did not go down")
        ok(f"{mode} SFT steps")
        if mode == "lora":
            loss, st = orpo_loss(model, collate([encode(tok, prompt, good), encode(tok, prompt, bad)], pad), tc)
            loss.backward()
            opt.zero_grad()
            if not torch.isfinite(loss):
                fail("orpo loss not finite")
            ok(f"orpo loss {loss.item():.4f} log odds {st['log_odds']:.4f}")
        if mode == "qlora":
            model.eval()
            save_merged(cfg, "_smoke_train", model, tok, cfg["base_model"], cfg["base_model"])
            out = merged_dir(cfg, "_smoke_train")
            if not list(out.glob("*.safetensors")):
                fail("merged safetensors not written")
            m = AutoModelForCausalLM.from_pretrained(str(out), torch_dtype=torch.bfloat16)
            if any(p.dtype != torch.bfloat16 for p in m.parameters()):
                fail("merged model is not bf16")
            ok("merge QLoRA adapter into bf16 base, save safetensors, reload")
            del m
            shutil.rmtree(out.parent, ignore_errors=True)
        del model, opt
        torch.cuda.empty_cache()


def check_grpo(cfg):
    from transformers import AutoTokenizer

    from .common import model_dir
    from .data import chat_prompt
    from .grpo import train_grpo

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    rows = [{"prompt": chat_prompt(tok, ex), "gold_json": json.dumps(ex["gold"]), "tool_names_json": json.dumps([t["name"] for t in ex["tools"]])} for ex in toy_examples()[:4]]
    hist = train_grpo(
        cfg,
        smoke_rows=rows,
        smoke_overrides={"max_steps": 2, "num_generations": 2, "per_device_batch": 2, "grad_accum": 2, "max_completion_length": 128, "vllm_gpu_memory_utilization": 0.25},
    )
    rewards = [h.get("reward") for h in hist if "reward" in h]
    print(f"  rewards {rewards}", flush=True)
    if not rewards:
        fail("GRPO logged no rewards")
    ok("GRPO with vLLM colocate, LoRA and the verifiable tool reward")
    shutil.rmtree(model_dir(cfg, "_smoke_grpo"), ignore_errors=True)


FUNCS = {"sys": check_sys, "hub": check_hub, "data": check_data, "vllm": check_vllm, "train": check_train, "grpo": check_grpo}


def run_part(cfg, part):
    print(f"\nCHECK {part}", flush=True)
    FUNCS[part](cfg)
    print(f"CHECK {part} DONE", flush=True)
