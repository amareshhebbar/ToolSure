# ToolSure

Tiny models that never break a function call. One base model (`Qwen/Qwen2.5-1.5B-Instruct`), one dataset, five training methods plus constrained decoding, and one comparison table against prompting baselines on BFCL and held out xLAM.

Every model is pushed to the Hugging Face Hub as bf16 safetensors, added to the ToolSure collection, downloaded back from the Hub and tested before the next model starts. No external APIs are used anywhere.

## Before the first run

`Salesforce/xlam-function-calling-60k` is gated. Open its page on Hugging Face while logged in as your account and accept the terms once, or the data check fails.

## Quick start on RunPod

Use a 48 GB GPU (L40S is a good fit) with a volume of at least 100 GB at `/workspace`.

```bash
cd /workspace
git clone https://github.com/amareshhebbar/ToolSure.git && cd ToolSure
bash setup.sh
export HF_TOKEN=hf_yourtokenhere
export HF_HOME=/workspace/hf_cache
source .venv/bin/activate
python run.py check
tmux new -s ts
python run.py pipeline
```

Do not export `TS_WORK`: each config sets its own run folder, and `TS_WORK` would override it.

## Two configs

<table>
<tr><th>File</th><th>Use</th><th>Rough cost on an L40S at $1/hour</th></tr>
<tr><td>configs/config.yaml</td><td>Small budget, the default</td><td>about 4 to 6 hours</td></tr>
<tr><td>configs/config_big_budget.yaml</td><td>Full run: 5x data, 2 epochs, 400 GRPO steps, MMLU</td><td>about 16 to 24 hours</td></tr>
</table>

To use the big config:

```bash
TS_CONFIG=configs/config_big_budget.yaml python run.py check
TS_CONFIG=configs/config_big_budget.yaml python run.py pipeline
```

The two configs write to separate run folders (`/workspace/toolsure_runs` and `/workspace/toolsure_runs_big`) and push to the same Hub repos, so a big run upgrades the small models in place.

## What the check covers

<table>
<tr><th>Check</th><th>What it proves</th></tr>
<tr><td>sys</td><td>Every library imports, CUDA and bf16 work, VRAM and disk are enough</td></tr>
<tr><td>hub</td><td>Token has write access: creates a private repo and a collection, then deletes both</td></tr>
<tr><td>data</td><td>xLAM, Glaive and BFCL load; the grader passes gold calls and catches wrong values, broken JSON and unneeded calls; the training target matches the Qwen2.5 tool template</td></tr>
<tr><td>vllm</td><td>Base model generates tool calls normally and with grammar constrained decoding</td></tr>
<tr><td>train</td><td>LoRA, QLoRA (4 bit) and DoRA losses go down, ORPO works, a QLoRA adapter merges into bf16 safetensors and reloads</td></tr>
<tr><td>grpo</td><td>Two real GRPO steps with vLLM colocated and the tool reward</td></tr>
</table>

## Pipeline

1. `baseline`: base model prompted, with a strict system prompt, and with constrained decoding.
2. `reference`: Qwen2.5 7B Instruct prompted, as a bigger model to beat (set `reference_model: null` to skip).
3. `baseline_mmlu`: MMLU for the base model (big config only).
4. `factory`: builds training data from xLAM and Glaive, collects real mistakes from the base model as negatives, pushes the dataset and checks row counts from the Hub.
5. For SFT, QLoRA, DoRA, ORPO, GRPO and CONSTRAINED: `train`, `push`, `verify` (download from the Hub, quick eval, stop if broken), `evalgen`, `mmlu`, `publish`.
6. `report`: charts and final cards for every model.

CONSTRAINED copies the best trained model and adds `toolsure_constrained.py`, which forces every tool call to a real function and schema valid arguments while still allowing plain text answers.

Finished steps are recorded, so after a pod restart run the same command again and it resumes.

```bash
python run.py status
python run.py run orpo:verify
python run.py pipeline from=grpo:train to=grpo:publish
python run.py reset grpo:train
```

## Hub names

<table>
<tr><td>Dataset</td><td>AmareshHebbar/ToolSure_Calls_Qwen2.5_1.5B</td></tr>
<tr><td>Models</td><td>AmareshHebbar/ToolSure_Qwen2.5_1.5B_SFT, _QLORA, _DORA, _ORPO, _GRPO, _CONSTRAINED</td></tr>
<tr><td>Collection</td><td>ToolSure</td></tr>
</table>

## Layout

<table>
<tr><td>run.py</td><td>Entry point: check, pipeline, run, status, reset</td></tr>
<tr><td>ts/data.py</td><td>xLAM, Glaive and BFCL loaders, schema conversion, prompt and target format</td></tr>
<tr><td>ts/grade.py</td><td>Tool call parser, BFCL style AST checker, GRPO reward</td></tr>
<tr><td>ts/engine.py</td><td>vLLM generation, constrained decoding with xgrammar structural tags, latency</td></tr>
<tr><td>ts/factory.py</td><td>Training sets, on policy and synthetic negatives, dataset push</td></tr>
<tr><td>ts/train.py</td><td>SFT with LoRA, QLoRA and DoRA, and ORPO</td></tr>
<tr><td>ts/grpo.py</td><td>GRPO with the verifiable tool reward</td></tr>
<tr><td>ts/constrained.py</td><td>Builds the CONSTRAINED model from the best trained model</td></tr>
<tr><td>ts/evaluate.py, ts/verify.py</td><td>Evaluation, MMLU, Hub download and gate</td></tr>
<tr><td>ts/hub.py, ts/card.py, ts/report.py</td><td>Uploads, collection, model cards, charts</td></tr>
</table>
