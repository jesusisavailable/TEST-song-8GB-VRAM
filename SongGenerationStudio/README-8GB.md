# SongGeneration Studio — 8GB VRAM Adaptation

This repository adapts [SongGeneration Studio](https://github.com/BazedFrog/SongGeneration-Studio)
to run on **8GB VRAM NVIDIA GPUs**. The stock project requires 10GB+ of VRAM.
With 4-bit quantization, FP16 mixed precision, CPU layer offloading and a 30s
duration cap, the base models run comfortably within 8GB.

## How it works (memory budget)

The heavy component is the *audiolm* — a custom Llama (dim 1536, 28 + 12 layers)
that weighs ~7GB in FP16. The following techniques keep peak usage under 8GB:

| Technique | Where | Effect |
|---|---|---|
| 4-bit bitsandbytes (nf4) quantization | `patches/gradio/levo_inference_lowmem.py` | audiolm ~7GB → ~2GB |
| FP16 autocast during generation | same file (already upstream) | halves activation memory |
| CPU layer offloading (`OffloadProfiler`) | the model's own `offload:` config | layers streamed CPU ↔ GPU |
| 30s max song duration (`SG_MAX_SONG_DURATION_SECONDS`) | same file | keeps LM KV-cache small |
| Aggressive `empty_cache()` / `gc.collect()` between stages | same file | LM freed before decoding |

The model checkpoints (`lglg666/SongGeneration-*`) already ship `offload:`
sections for the audiolm and the wav-tokenizer/diffusion decoder, so the patched
inference uses them automatically.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SG_TARGET_VRAM_GB` | `8` | Target VRAM. Values ≤ 8 auto-enable 4-bit quantization and the 30s duration cap |
| `SG_QUANTIZE` | auto | `4`, `8`, or `off` to force / disable audiolm quantization |
| `SG_MAX_SONG_DURATION_SECONDS` | `30` (≤8GB) / model default | Cap on generated song length |
| `SG_FORCE_OFFLOAD` | `1` | Keep CPU offloading enabled |

The values are resolved in `config.py`, written back into the environment, and
inherited by the model-server subprocess.

## Setup

### Via Pinokio
1. Open Pinokio → SongGeneration Studio → **Install** (models ~15GB download).
2. `bitsandbytes` is installed automatically by the updated `install.js`.
3. Start the app and open the web UI — the 8GB profile is active by default.

### Manual install
1. Create a venv, then install (mirrors `install.js`):
   ```
   uv pip install -r requirements.txt
   uv pip install -r requirements_nodeps.txt --no-deps
   uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
   uv pip install bitsandbytes>=0.44.0
   ```
2. Clone Tencent's SongGeneration repo into `app/` and download the model
   weights (see `install.js` steps 1–3):
   ```
   git clone https://github.com/tencent-ailab/SongGeneration app
   hf download lglg666/SongGeneration-Runtime --local-dir app
   ```
3. Apply the patches (as `install.js` does):
   - `patches/builders.py` → `app/codeclm/models/builders.py`
   - `patches/demucs/apply.py` → `app/third_party/demucs/models/apply.py`
   - `patches/gradio/levo_inference_lowmem.py` → `app/tools/gradio/levo_inference_lowmem.py`
   - copy `main.py`, `model_server.py`, `config.py`, `models.py`, `gpu.py`, `schemas.py`, `sse.py`, `generation.py`, `timing.py`, and `web/` into `app/`
4. Run:
   ```
   cd app
   python main.py --host 127.0.0.1
   ```
   The web UI opens at the printed URL.

## CLI: single-song generation

Prefer a quick test without the web app? Use `run_songgen.py`:

```
python run_songgen.py --lyrics "I want to be free tonight" --description "upbeat pop" --duration 30 --quantize 4 --output out.flac
```

Options: `--model`, `--lyrics`, `--description`, `--genre`, `--duration`,
`--quantize {4,8,off}`, `--output`, `--target-vram`, `--gpu`. See
`python run_songgen.py --help`.

## What changed vs. upstream

- `patches/gradio/levo_inference_lowmem.py` — added optional bitsandbytes
  4/8-bit quantization of the audiolm (failsafe fallback to FP16 + offload),
  30s duration cap, VRAM diagnostics, and fixed an unguarded
  `offload_profiler.stop()` that crashed when a config has no `offload:` section.
- `config.py` / `gpu.py` / `models.py` / `main.py` — 8GB-aware profile, VRAM
  detection and model registry (base models claimable at 8GB VRAM).
- `requirements.txt` / `install.js` — `bitsandbytes` dependency.
- `run_songgen.py` — new standalone CLI.

## Troubleshooting

- **`bitsandbytes` not installed** → the patch logs a warning and falls back to
  the original FP16 + offload path.
- **Pascal GPU (GTX 10-series, e.g. GTX 1070)** → bitsandbytes int8/nf4 kernels
  are limited on compute capability 6.x, so the patch auto-falls back to FP16 +
  CPU offloading (slower, but still fits 8GB thanks to layer streaming). You can
  also force `SG_QUANTIZE=off` to skip quantization up front.
- **OOM during decoding** → lower `SG_MAX_SONG_DURATION_SECONDS`, use the base
  model, and avoid reference/uploaded audio (the separator stage is memory-hungry).
- **Quantization fails on an exotic layer** → set `SG_QUANTIZE=off` to use the
  proven FP16 path (10GB+ cards).
- **Speed** → 4-bit + offload is slower than pure FP16; raise
  `SG_MAX_SONG_DURATION_SECONDS`/`SG_QUANTIZE=off` only if VRAM allows.

