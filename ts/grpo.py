import dataclasses
import json
import random

import torch

from .common import banner, init_repo_for, model_dir, read_jsonl, resolve_model, write_json
from .grade import calls_match, parse_output, reward
from .train import load_rows, lora_config, save_merged


def tool_reward(prompts, completions, gold_json=None, tool_names_json=None, **kwargs):
    return [reward(json.loads(g), json.loads(t), c) for c, g, t in zip(completions, gold_json, tool_names_json)]


def exact(prompts, completions, gold_json=None, **kwargs):
    out = []
    for c, g in zip(completions, gold_json):
        gold, p = json.loads(g), parse_output(c)
        if not gold:
            out.append(0.0 if p["attempted"] else 1.0)
        else:
            out.append(1.0 if p["attempted"] and p["parse_ok"] and calls_match(p["calls"], gold) else 0.0)
    return out


def valid_format(prompts, completions, **kwargs):
    return [1.0 if parse_output(c)["parse_ok"] else 0.0 for c in completions]


def build_config(cls, values):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(k for k in values if k not in names)
    if unknown:
        print(f"ignoring keys not in {cls.__name__}: {unknown}", flush=True)
    return cls(**{k: v for k, v in values.items() if k in names})


def grpo_args(cfg, g, out_dir):
    from trl import GRPOConfig

    return build_config(
        GRPOConfig,
        {
            "output_dir": str(out_dir),
            "learning_rate": g["lr"],
            "lr_scheduler_type": "cosine",
            "warmup_ratio": g["warmup_ratio"],
            "max_steps": g["max_steps"],
            "per_device_train_batch_size": g["per_device_batch"],
            "gradient_accumulation_steps": g["grad_accum"],
            "num_generations": g["num_generations"],
            "max_prompt_length": g["max_prompt_length"],
            "max_completion_length": g["max_completion_length"],
            "temperature": g["temperature"],
            "top_p": g["top_p"],
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": g["vllm_gpu_memory_utilization"],
            "loss_type": "dr_grpo",
            "scale_rewards": False,
            "beta": g["beta"],
            "mask_truncated_completions": True,
            "reward_weights": [1.0, 0.0, 0.0],
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "seed": cfg["seed"],
            "remove_unused_columns": False,
            "log_completions": False,
            "disable_dropout": True,
        },
    )


def train_grpo(cfg, smoke_rows=None, smoke_overrides=None):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    g = dict(cfg["train"]["grpo"])
    if smoke_overrides:
        g.update(smoke_overrides)
    init = "base" if smoke_rows is not None else g.get("init", "base")
    init_path = resolve_model(cfg, init)
    tok = AutoTokenizer.from_pretrained(init_path)
    if smoke_rows is None:
        rows = [r for r in load_rows(cfg, "prompts") if r["prompt_tokens"] <= g["max_prompt_length"]]
        random.Random(cfg["seed"]).shuffle(rows)
        need = g["max_steps"] * g["per_device_batch"] * g["grad_accum"] // g["num_generations"]
        rows = rows[: min(len(rows), g.get("max_prompts") or need)]
        out_dir = model_dir(cfg, "grpo") / "trainer"
    else:
        rows = smoke_rows
        out_dir = model_dir(cfg, "_smoke_grpo") / "trainer"
    ds = Dataset.from_list([{"prompt": r["prompt"], "gold_json": r["gold_json"], "tool_names_json": r["tool_names_json"]} for r in rows])
    banner(f"grpo: init {init}, {len(ds)} prompts, {g['max_steps']} steps, {g['num_generations']} samples per prompt")
    model = AutoModelForCausalLM.from_pretrained(init_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[tool_reward, exact, valid_format],
        args=grpo_args(cfg, g, out_dir),
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora_config(cfg, "lora"),
    )
    trainer.train()
    history = trainer.state.log_history
    if smoke_rows is not None:
        return history
    logs = model_dir(cfg, "grpo") / "training"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "log.jsonl", "w") as f:
        for h in history:
            if "loss" in h or "reward" in h:
                f.write(json.dumps({k.replace("/", "_"): v for k, v in h.items() if isinstance(v, (int, float))}) + "\n")
    write_json(logs / "summary.json", {"prompts": len(ds), "steps": g["max_steps"], "init": init, "config": g})
    plot_grpo(logs / "log.jsonl", logs / "curves.png")
    trainer.model.eval()
    save_merged(cfg, "grpo", trainer.model, tok, init_path, init_repo_for(cfg, init))


def plot_grpo(log_path, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_jsonl(log_path)
    keys = [k for k in ("reward", "rewards_exact_mean", "rewards_valid_format_mean", "completions_mean_length") if any(k in r for r in rows)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5), squeeze=False)
    for ax, k in zip(axes[0], keys):
        pts = [(r["step"], r[k]) for r in rows if k in r and "step" in r]
        ax.plot([p[0] for p in pts], [p[1] for p in pts])
        ax.set_title(f"grpo {k}")
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
