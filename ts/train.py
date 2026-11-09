import gc
import json
import math
import random
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import banner, dataset_repo, init_repo_for, model_dir, read_jsonl, resolve_model, wpath, write_json

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_rows(cfg, name):
    p = wpath(cfg, "data", f"{name}.jsonl")
    if p.exists():
        return read_jsonl(p)
    from datasets import load_dataset

    print(f"{p} missing, loading {name} from {dataset_repo(cfg)}", flush=True)
    return [dict(r) for r in load_dataset(dataset_repo(cfg), name, split="train")]


def lora_config(cfg, mode):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        use_dora=mode == "dora",
    )


def build_model(cfg, path, mode):
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    if mode == "qlora":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model.to(DEV)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, lora_config(cfg, mode))
    model.print_trainable_parameters()
    model.train()
    return model


def _chunk_logp(h, w, t):
    logits = F.linear(h.to(w.dtype), w).float()
    lse = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return picked - lse


def seq_logps(model, ids, mask, labels, chunk):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    w = base.lm_head.weight
    parts = []
    for s in range(0, hidden.size(1), chunk):
        h, t = hidden[:, s : s + chunk], targets[:, s : s + chunk]
        if torch.is_grad_enabled():
            parts.append(checkpoint(_chunk_logp, h, w, t, use_reentrant=False))
        else:
            parts.append(_chunk_logp(h, w, t))
    lp = torch.cat(parts, dim=1)
    m = (targets != -100).to(lp.dtype)
    return (lp * m).sum(-1), m.sum(-1)


def encode(tok, prompt, completion):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    return p + c, [-100] * len(p) + c


def collate(seqs, pad_id):
    n = max(len(i) for i, _ in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), n), -100, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for k, (i, l) in enumerate(seqs):
        ids[k, : len(i)] = torch.tensor(i)
        lab[k, : len(l)] = torch.tensor(l)
        att[k, : len(i)] = 1
    return ids.to(DEV), att.to(DEV), lab.to(DEV)


def make_batches(lengths, micro, seed, epoch):
    rng = random.Random(seed + epoch)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    bucket = micro * 64
    batches = []
    for s in range(0, len(idx), bucket):
        chunk = sorted(idx[s : s + bucket], key=lambda i: lengths[i])
        batches.extend(chunk[k : k + micro] for k in range(0, len(chunk), micro))
    rng.shuffle(batches)
    return batches


def log1mexp(x):
    x = x.clamp(max=-1e-6)
    return torch.log(-torch.expm1(x))


def sft_loss(model, batch, tcfg):
    ids, att, lab = batch
    lp, n = seq_logps(model, ids, att, lab, tcfg["logits_chunk"])
    loss = -lp.sum() / n.sum().clamp(min=1)
    return loss, {"nll": loss.item(), "tokens": n.sum().item()}


def orpo_loss(model, batch, tcfg):
    ids, att, lab = batch
    b = ids.size(0) // 2
    lp, n = seq_logps(model, ids, att, lab, tcfg["logits_chunk"])
    ac, ar = lp[:b] / n[:b].clamp(min=1), lp[b:] / n[b:].clamp(min=1)
    log_odds = (ac - ar) - (log1mexp(ac) - log1mexp(ar))
    ratio = -F.logsigmoid(log_odds).mean()
    nll = -lp[:b].sum() / n[:b].sum().clamp(min=1)
    loss = nll + tcfg["lam"] * ratio
    return loss, {
        "loss": loss.item(),
        "nll": nll.item(),
        "ratio_loss": ratio.item(),
        "log_odds": log_odds.mean().item(),
        "reward_acc": (log_odds > 0).float().mean().item(),
    }


def train_loop(cfg, name, model, data, loss_fn, tcfg, pad_id):
    out = model_dir(cfg, name) / "training"
    out.mkdir(parents=True, exist_ok=True)
    lengths = [d["len"] for d in data]
    epochs, accum, micro = tcfg["epochs"], tcfg["grad_accum"], tcfg["micro_batch"]
    total_steps = math.ceil(math.ceil(len(data) / micro) * epochs / accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], betas=(0.9, 0.95), weight_decay=0.0, fused=DEV == "cuda")
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(opt, max(1, int(tcfg["warmup_ratio"] * total_steps)), total_steps)
    banner(f"{name}: {len(data)} examples, {epochs} epochs, {total_steps} optimizer steps, effective batch {micro * accum}")
    log_f = open(out / "log.jsonl", "w")
    step, acc_count, t0, agg = 0, 0, time.time(), {}
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        batches = make_batches(lengths, micro, cfg["seed"], ep)
        for bi, idxs in enumerate(batches):
            seqs = [data[i]["seqs"][0] for i in idxs] + [s for i in idxs for s in data[i]["seqs"][1:]]
            loss, st = loss_fn(model, collate(seqs, pad_id), tcfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non finite loss at step {step}: {st}")
            (loss / accum).backward()
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
            acc_count += 1
            last = ep == epochs - 1 and bi == len(batches) - 1
            if acc_count == accum or last:
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {k: v / acc_count for k, v in agg.items()}
                rec.update({"step": step, "epoch": ep, "lr": sched.get_last_lr()[0], "grad_norm": gn, "elapsed_min": (time.time() - t0) / 60})
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if step % 5 == 0 or step == 1 or last:
                    eta = (time.time() - t0) / step * (total_steps - step) / 60
                    print(f"[{name}] " + " ".join(f"{k} {v:.4g}" for k, v in rec.items() if k != "epoch") + f" eta_min {eta:.1f}", flush=True)
                agg, acc_count = {}, 0
    log_f.close()
    plot_log(out / "log.jsonl", out / "loss.png", name)
    write_json(out / "summary.json", {"examples": len(data), "steps": step, "minutes": (time.time() - t0) / 60, "config": tcfg})


def plot_log(log_path, png, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_jsonl(log_path)
    if not rows:
        return
    keys = [k for k in ("loss", "nll", "log_odds", "reward_acc") if k in rows[0]]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ax.plot([r["step"] for r in rows], [r[k] for r in rows])
        ax.set_title(f"{name} {k}")
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def save_merged(cfg, name, model, tok, init_path, init_repo):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, GenerationConfig

    root = model_dir(cfg, name)
    adapter, merged = root / "adapter", root / "merged"
    for d in (adapter, merged):
        if d.exists():
            shutil.rmtree(d)
    model.peft_config["default"].base_model_name_or_path = init_repo
    model.save_pretrained(str(adapter))
    del model
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(init_path, torch_dtype=torch.bfloat16).to(DEV)
    full = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    full.config.use_cache = True
    full.save_pretrained(str(merged), safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(str(merged))
    try:
        GenerationConfig.from_pretrained(init_path).save_pretrained(str(merged))
    except Exception:
        pass
    banner(f"saved merged bf16 model to {merged}")


def prepare(name, tok, rows, kind, max_len, limit):
    data, dropped = [], 0
    for r in rows:
        if kind == "sft":
            s = encode(tok, r["prompt"], r["completion"])
            if len(s[0]) > max_len:
                dropped += 1
                continue
            data.append({"seqs": [s], "len": len(s[0])})
        else:
            c, j = encode(tok, r["prompt"], r["chosen"]), encode(tok, r["prompt"], r["rejected"])
            if max(len(c[0]), len(j[0])) > max_len:
                dropped += 1
                continue
            data.append({"seqs": [c, j], "len": len(c[0]) + len(j[0])})
        if limit and len(data) >= limit:
            break
    print(f"{name}: kept {len(data)} examples, dropped {dropped} over {max_len} tokens", flush=True)
    return data


def train_offline(cfg, name):
    from transformers import AutoTokenizer

    tcfg = cfg["train"][name]
    torch.manual_seed(cfg["seed"])
    init = tcfg.get("init", "base")
    init_path = resolve_model(cfg, init)
    banner(f"{name}: init from {init} ({init_path}), mode {tcfg['mode']}")
    tok = AutoTokenizer.from_pretrained(init_path)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kind = "pairs" if name == "orpo" else "sft"
    data = prepare(name, tok, load_rows(cfg, kind), kind, tcfg["max_len"], tcfg.get("max_examples") or 0)
    if not data:
        raise RuntimeError(f"{name}: no training data")
    model = build_model(cfg, init_path, tcfg["mode"])
    train_loop(cfg, name, model, data, orpo_loss if name == "orpo" else sft_loss, tcfg, pad_id)
    model.eval()
    save_merged(cfg, name, model, tok, init_path, init_repo_for(cfg, init))
