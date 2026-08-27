"""
run_songgen.py - Single-song generation for 8GB VRAM GPUs.

Standalone entry point that drives the same patched low-memory LeVo inference as
the SongGeneration Studio web app, so you can generate a complete song straight
from the command line. It ties together the 8GB VRAM recipe from the build plan:

  * 4-bit / 8-bit bitsandbytes quantization of the audiolm
  * FP16 autocast mixed precision
  * CPU layer offloading (from the model's own `offload:` config)
  * a configurable max song duration (default 30s)

Prerequisites
-------------
The installer must have run once so the Tencent `codeclm` package and the model
weights exist under `app/`. Run the script from the repo root (it will find
`app/` automatically) or from inside `app/`:

    python run_songgen.py --lyrics "..." --description "upbeat pop" --output out.flac

Examples
--------
    # 30s pop song with auto 4-bit quantization (8GB mode)
    python run_songgen.py --lyrics "I want to be free tonight" --description "upbeat pop"

    # Longer song with explicit 8-bit quantization
    python run_songgen.py --lyrics "..." --description "sad ballad" --duration 60 --quantize 8

    # Genre-driven (uses tools/new_prompt.pt auto prompts)
    python run_songgen.py --lyrics "..." --genre Pop --duration 30
"""

import argparse
import os
import sys
import time
from pathlib import Path


def resolve_app_dir():
    """Locate the directory that contains the installed `codeclm` package."""
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "app",
    ]
    for cand in candidates:
        if (cand / "codeclm").is_dir():
            return cand
    raise SystemExit(
        "Could not find the 'codeclm' package. Run the installer first so 'app/' exists "
        "with Tencent's codeclm and the downloaded model weights (see README-8GB.md)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="SongGeneration Studio - single song generation (8GB VRAM)"
    )
    parser.add_argument(
        "--model", default="songgeneration_base",
        choices=["songgeneration_base", "songgeneration_base_new",
                 "songgeneration_base_full", "songgeneration_large"],
        help="model id (default: songgeneration_base)",
    )
    parser.add_argument("--lyrics", default="", help="lyrics text (whitespace separated lines)")
    parser.add_argument("--description", default="",
                        help="style description, e.g. 'upbeat pop, female vocal'")
    parser.add_argument("--genre", default=None,
                        help="genre key for auto-prompt (e.g. 'Pop', 'Rock', 'Jazz')")
    parser.add_argument("--duration", type=int, default=30,
                        help="max song duration in seconds (default: 30)")
    parser.add_argument("--quantize", choices=["4", "8", "off"], default=None,
                        help="audiolm quantization (default: auto - 4-bit when target VRAM <= 8GB)")
    parser.add_argument("--output", default="output/song.flac", help="output audio path (.flac/.wav)")
    parser.add_argument("--target-vram", type=int, default=8, help="target VRAM in GB (default: 8)")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES index (default: 0)")
    args = parser.parse_args()

    # --- Apply the 8GB VRAM profile (inherited by the inference patch) ---
    os.environ["SG_TARGET_VRAM_GB"] = str(args.target_vram)
    os.environ.setdefault("SG_MAX_SONG_DURATION_SECONDS", str(args.duration))
    if args.quantize:
        os.environ["SG_QUANTIZE"] = args.quantize
    os.environ.setdefault("SG_FORCE_OFFLOAD", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # --- Locate the installed app/codeclm environment --------------------
    app_dir = resolve_app_dir()
    sys.path.insert(0, str(app_dir))
    sys.path.insert(0, str(app_dir / "tools" / "gradio"))
    sys.path.insert(0, str(app_dir / "codeclm" / "tokenizer" / "Flow1dVAE"))

    import torch
    from levo_inference_lowmem import LeVoInference  # the patched low-memory inference

    model_path = app_dir / args.model
    if not (model_path / "config.yaml").exists():
        raise SystemExit(
            f"Model '{args.model}' not found at {model_path}.\n"
            "Download it first (installer or the web UI's model manager)."
        )

    print(f"[RUN] Model: {model_path}")
    print(f"[RUN] Duration cap: {args.duration}s | Quantize: {os.environ.get('SG_QUANTIZE', 'auto')}")
    if not torch.cuda.is_available():
        print("[RUN] WARNING: CUDA not available - generation will fail on most models.")

    start = time.time()
    inference = LeVoInference(str(model_path))

    gen_params = {"duration": min(args.duration, int(inference.max_duration))}

    auto_prompt_path = None
    genre = None
    if args.genre:
        candidate = app_dir / "tools" / "new_prompt.pt"
        if candidate.exists():
            auto_prompt_path = str(candidate)
            genre = args.genre
        else:
            print(f"[RUN] WARNING: auto-prompt file not found at {candidate}; using melody_is_wav mode")

    print("[RUN] Generating...", flush=True)
    audio = inference.forward(
        lyric=" ".join(str(args.lyrics).split()),
        description=args.description,
        prompt_audio_path=None,
        genre=genre,
        auto_prompt_path=auto_prompt_path,
        gen_type="mixed",
        params=gen_params,
    )

    import soundfile as sf

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = int(inference.cfg.sample_rate)
    sf.write(str(out_path), audio.cpu().permute(1, 0).float().numpy(), sample_rate)
    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
    print(f"[RUN] Saved: {out_path} ({elapsed:.0f}s, {sample_rate} Hz)")
    print(f"[RUN] Peak VRAM: {peak_vram:.2f}GB")


if __name__ == "__main__":
    main()
