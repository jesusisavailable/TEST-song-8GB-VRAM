"""
checks.py - Verify SongGeneration Studio runtime prerequisites and environment.

Run from the repo root (or SongGenerationStudio/); it resolves the launcher's
'ready-to-run' state and prints a PASS/FAIL report. Uses no heavy imports so it
works even before dependencies are installed.

    python SongGenerationStudio/checks.py
    :: or, from the repo root
    python checks.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "SongGenerationStudio":
    ROOT = ROOT.parent  # allow running from here or from the repo root

APP = ROOT / "app"
APP_MAIN = APP / "main.py"
APP_MODEL = APP / "songgeneration_base" / "model.pt"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"

FLOW1D = APP / "codeclm" / "tokenizer" / "Flow1dVAE"
PATCH_LEVO = APP / "tools" / "gradio" / "levo_inference_lowmem.py"


def _cli(name: str, *args, **kw):
    """Run a CLI tool; return (found: bool, rc: int, out: str)."""
    path = shutil.which(name)
    if not path:
        return False, -1, ""
    try:
        r = subprocess.run([name, *args], capture_output=True, text=True,
                           timeout=kw.get("timeout", 15))
        return True, r.returncode, (r.stdout or r.stderr)
    except (OSError, subprocess.TimeoutExpired):
        return True, -1, ""


def check(name, ok, detail=""):
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {name}" + (f"  -> {detail}" if detail else ""))
    return ok


def main():
    results = []

    # Python interpreter
    py = VENV_PY if VENV_PY.exists() else shutil.which("python")
    ok = py is not None
    results.append(check("Python interpreter", ok, str(py) if py else "none found"))

    # app/ runtime
    results.append(check("app/ exists", APP.is_dir(), str(APP)))
    results.append(check("app/main.py present", APP_MAIN.exists(), str(APP_MAIN)))
    results.append(check("app/ patch (levo_inference_lowmem) present",
                         PATCH_LEVO.exists(), str(PATCH_LEVO)))

    # Model weights
    if APP_MODEL.exists():
        gb = APP_MODEL.stat().st_size / (1024 ** 3)
        results.append(check("songgeneration_base/model.pt present", True, f"{gb:.1f} GB"))
    else:
        # Runtime repo stores weights under ckpt/ sometimes; report if any ckpt exists.
        ckpt = APP / "ckpt"
        results.append(check("songgeneration_base/model.pt present", False,
                             "model missing - download via model manager or re-run launcher"))
        if ckpt.is_dir():
            files = list(ckpt.rglob("*.pt"))
            print(f"  (found {len(files)} .pt files under ckpt/ - model may use a different layout)")

    # GPU
    found, _, out = _cli("nvidia-smi", "--query-gpu=name,memory.total",
                         "--format=csv,noheader,nounits")
    results.append(check("NVIDIA GPU detected (nvidia-smi)", found, out.strip().splitlines()[0] if out.strip() else ""))

    # PyTorch/CUDA in the venv (heavy, so best-effort)
    if py and VENV_PY.exists():
        try:
            r = subprocess.run([str(py), "-c",
                                "import torch;print(torch.__version__,'cuda=',torch.cuda.is_available())"],
                               capture_output=True, text=True, timeout=30)
            results.append(check("torch import (venv)", r.returncode == 0, r.stdout.strip() or r.stderr.strip()[:120]))
        except (OSError, subprocess.TimeoutExpired) as e:
            results.append(check("torch import (venv)", False, str(e)[:120]))
    else:
        results.append(check("torch import (venv)", False, "no .venv - run the launcher first"))

    # FFmpeg (trim + export)
    ff = _cli("ffmpeg", "-version")
    fp = _cli("ffprobe", "-version")
    results.append(check("ffmpeg available", ff[0] and ff[1] == 0, "trim/export ready" if ff[0] else "install ffmpeg & put on PATH"))
    results.append(check("ffprobe available", fp[0] and fp[1] == 0, ""))

    # 8GB profile env defaults
    tgt = os.environ.get("SG_TARGET_VRAM_GB", "8")
    results.append(check("8GB profile (SG_TARGET_VRAM_GB)", tgt == "8", f"{tgt}"))

    failed = sum(1 for r in results if not r)
    print()
    print(f"=== {len(results) - failed}/{len(results)} passed ===")
    if failed:
        print("Fix the FAIL items above, then re-run. Generated apps needing CUDA/ffmpeg")
        print("will not start correctly until they pass.")
        return 1
    print("All prerequisite checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())