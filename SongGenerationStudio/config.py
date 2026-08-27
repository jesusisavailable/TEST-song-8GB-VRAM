"""
SongGeneration Studio - Configuration
Directories, constants, and shared state initialization.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

# ============================================================================
# Optional .env support (no secrets/paths committed; see .env.example)
# ============================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================================
# Directory Configuration
# ============================================================================

ROOT_DIR = Path(__file__).parent  # This is app/
BASE_DIR = ROOT_DIR  # App resources (models, venv, etc.) are in app/
DEFAULT_MODEL = "songgeneration_base"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "web" / "static"  # Static files in app/web/static/
QUEUE_FILE = BASE_DIR / "queue.json"
VERIFIED_MODELS_FILE = BASE_DIR / "verified_models.json"
TIMING_FILE = BASE_DIR / "timing_history.json"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Upload / export safety limits (overridable via environment)
# ============================================================================
MAX_UPLOAD_MB = int(os.environ.get("SG_MAX_UPLOAD_MB", "100"))
MAX_REFERENCE_SECONDS = float(os.environ.get("SG_MAX_REFERENCE_SECONDS", "10"))
REFERENCE_TTL_HOURS = int(os.environ.get("SG_REFERENCE_TTL_HOURS", "24"))
TEMP_DIR = BASE_DIR / "tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Model Server Configuration
MODEL_SERVER_PORT = 42100
MODEL_SERVER_URL = f"http://127.0.0.1:{MODEL_SERVER_PORT}"
USE_MODEL_SERVER = True  # Set to False to use old subprocess method

# ============================================================================
# 8GB VRAM Adaptation Profile
# ============================================================================
# Tuning knobs for running on GPUs with limited VRAM (e.g. 8GB). Every value
# can be overridden with an environment variable, and the resolved defaults are
# written back to the environment so the model-server subprocess inherits them.

TARGET_VRAM_GB = int(os.environ.get("SG_TARGET_VRAM_GB", "8"))
QUANTIZE_LEVEL = os.environ.get("SG_QUANTIZE", "").strip().lower()  # "4" | "8" | "off" | "" (auto)
MAX_SONG_DURATION_SECONDS = int(os.environ.get("SG_MAX_SONG_DURATION_SECONDS", "0"))  # 0 = model default
FORCE_OFFLOAD = os.environ.get("SG_FORCE_OFFLOAD", "1").strip().lower() in ("1", "true", "yes")

# Auto-select 4-bit audiolm quantization when targeting a small card.
if QUANTIZE_LEVEL not in ("4", "8", "off"):
    QUANTIZE_LEVEL = "4" if TARGET_VRAM_GB <= 8 else "off"
# Cap the song duration to keep the LM KV-cache/activations in budget on small cards.
if MAX_SONG_DURATION_SECONDS <= 0:
    MAX_SONG_DURATION_SECONDS = 30 if TARGET_VRAM_GB <= 8 else 0

# Propagate the resolved values so the model-server subprocess sees them.
os.environ.setdefault("SG_TARGET_VRAM_GB", str(TARGET_VRAM_GB))
os.environ.setdefault("SG_QUANTIZE", QUANTIZE_LEVEL)
os.environ.setdefault("SG_MAX_SONG_DURATION_SECONDS", str(MAX_SONG_DURATION_SECONDS))
os.environ.setdefault("SG_FORCE_OFFLOAD", "1" if FORCE_OFFLOAD else "0")

# Timing History
MAX_TIMING_RECORDS = 1000  # Keep last 1000 successful generations

# ============================================================================
# Verified Models Cache
# ============================================================================

verified_models_cache: Dict[str, dict] = {}

def load_verified_models() -> dict:
    """Load verified models cache from disk"""
    try:
        if VERIFIED_MODELS_FILE.exists():
            with open(VERIFIED_MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_verified_models(cache: dict):
    """Save verified models cache to disk"""
    try:
        with open(VERIFIED_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def mark_model_verified(model_id: str, model_pt_size: int):
    """Mark a model as verified (size checked against HuggingFace)"""
    global verified_models_cache
    verified_models_cache[model_id] = {
        "verified": True,
        "model_pt_size": model_pt_size,
        "verified_at": datetime.now().isoformat()
    }
    save_verified_models(verified_models_cache)

def is_model_verified(model_id: str) -> bool:
    """Check if model has been verified"""
    return model_id in verified_models_cache and verified_models_cache[model_id].get("verified")

def get_verified_model_size(model_id: str) -> int:
    """Get verified model.pt size if available"""
    if model_id in verified_models_cache:
        return verified_models_cache[model_id].get("model_pt_size", 0)
    return 0

# Load verified models cache on startup
verified_models_cache = load_verified_models()

# ============================================================================
# Queue Storage
# ============================================================================

def load_queue() -> list:
    """Load queue from file"""
    try:
        if QUEUE_FILE.exists():
            with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[QUEUE] Error loading queue: {e}")
    return []

def save_queue(queue: list):
    """Save queue to file"""
    try:
        with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        print(f"[QUEUE] Error saving queue: {e}")

# ============================================================================
# Logging Helpers
# ============================================================================

def log_startup_info():
    """Log startup configuration info"""
    print(f"[CONFIG] Base dir: {BASE_DIR}")
    print(f"[CONFIG] Output dir: {OUTPUT_DIR}")
    print(f"[CONFIG] Python: {sys.executable}")
