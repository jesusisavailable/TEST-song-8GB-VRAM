"""
SongGeneration Studio - Model Registry & Download Manager
Model definitions, status checking, and download management.
"""

import sys
import shutil
import threading
import time
import subprocess
from typing import Dict, Optional, List
from pathlib import Path

from config import (
    BASE_DIR, mark_model_verified, is_model_verified,
    get_verified_model_size
)
from gpu import gpu_info

# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY: Dict[str, dict] = {
    "songgeneration_base": {
        "name": "SongGeneration - Base (2m30s)",
        "description": "Chinese + English, 8GB VRAM (4-bit quantized), max 2m30s",
        "vram_required": 8,
        "hf_repo": "lglg666/SongGeneration-base",
        "size_gb": 11.3,
        "priority": 1,
        "quantizable": True,
        "memory_notes": "8GB VRAM with 4-bit quantization + CPU offload",
    },
    "songgeneration_base_new": {
        "name": "SongGeneration - Base New (2m30s)",
        "description": "Updated base model, 8GB VRAM (4-bit quantized), max 2m30s",
        "vram_required": 8,
        "hf_repo": "lglg666/SongGeneration-base-new",
        "size_gb": 11.3,
        "priority": 2,
        "quantizable": True,
        "memory_notes": "8GB VRAM with 4-bit quantization + CPU offload",
    },
    "songgeneration_base_full": {
        "name": "SongGeneration - Base Full (4m30s)",
        "description": "Full duration up to 4m30s, 10GB VRAM (8GB with quantization at reduced duration)",
        "vram_required": 10,
        "hf_repo": "lglg666/SongGeneration-base-full",
        "size_gb": 11.3,
        "priority": 3,
        "quantizable": True,
        "memory_notes": "10GB VRAM; on 8GB cards reduce max duration",
    },
    "songgeneration_large": {
        "name": "SongGeneration - Large (4m30s)",
        "description": "Best quality, 22GB VRAM, max 4m30s",
        "vram_required": 22,
        "hf_repo": "lglg666/SongGeneration-large",
        "size_gb": 20.5,
        "priority": 4,
        "quantizable": True,
        "memory_notes": "Requires 22GB VRAM (not recommended for 8GB cards)",
    },
}

# ============================================================================
# Download State Tracking
# ============================================================================

download_states: Dict[str, dict] = {}
download_threads: Dict[str, threading.Thread] = {}
download_processes: Dict[str, subprocess.Popen] = {}
download_cancel_flags: Dict[str, threading.Event] = {}
expected_file_sizes_cache: Dict[str, dict] = {}
download_start_lock = threading.Lock()

# ============================================================================
# HuggingFace Helpers
# ============================================================================

def get_repo_file_sizes_from_hf(repo_id: str) -> dict:
    """Get exact file sizes in bytes from HuggingFace API."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)

        file_sizes = {}
        total_bytes = 0
        for sibling in repo_info.siblings:
            if hasattr(sibling, 'size') and sibling.size:
                filename = sibling.rfilename
                size_bytes = sibling.size
                file_sizes[filename] = size_bytes
                total_bytes += size_bytes

        file_sizes['__total__'] = total_bytes

        model_size = file_sizes.get('model.pt', 0)
        if model_size:
            print(f"[DOWNLOAD] HuggingFace API: {repo_id}/model.pt = {model_size:,} bytes ({model_size / (1024**3):.2f} GB)")
        print(f"[DOWNLOAD] HuggingFace API: {repo_id} total = {total_bytes:,} bytes ({total_bytes / (1024**3):.2f} GB)")

        return file_sizes
    except Exception as e:
        print(f"[DOWNLOAD] Could not get file sizes from HF API: {e}")
        return {}


def get_expected_file_sizes(model_id: str) -> dict:
    """Get expected file sizes for a model, using cache or fetching from HuggingFace."""
    global expected_file_sizes_cache

    if model_id in expected_file_sizes_cache:
        return expected_file_sizes_cache[model_id]

    if model_id not in MODEL_REGISTRY:
        return {}

    hf_repo = MODEL_REGISTRY[model_id]["hf_repo"]
    file_sizes = get_repo_file_sizes_from_hf(hf_repo)

    if file_sizes:
        expected_file_sizes_cache[model_id] = file_sizes

    return file_sizes


def get_directory_size(path: Path) -> int:
    """Get total size of all files in a directory in bytes"""
    if not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob('*'):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except (OSError, IOError):
                    pass
    except Exception:
        pass
    return total

# ============================================================================
# Model Status Functions
# ============================================================================

def get_model_status(model_id: str) -> str:
    """Get the status of a model: ready, downloading, not_downloaded"""
    if model_id in download_states and download_states[model_id].get("status") == "downloading":
        return "downloading"

    folder_path = BASE_DIR / model_id
    if not folder_path.exists():
        return "not_downloaded"

    if model_id not in MODEL_REGISTRY:
        return "not_downloaded"

    model_file = folder_path / "model.pt"
    if model_file.exists():
        try:
            actual_size = model_file.stat().st_size

            if is_model_verified(model_id):
                verified_size = get_verified_model_size(model_id)
                if verified_size > 0 and actual_size == verified_size:
                    return "ready"

            expected_sizes = get_expected_file_sizes(model_id)
            expected_size = expected_sizes.get('model.pt', 0)

            if expected_size > 0:
                size_diff_pct = abs(actual_size - expected_size) / expected_size * 100
                if size_diff_pct < 0.1:
                    mark_model_verified(model_id, actual_size)
                    return "ready"
            else:
                expected_size_gb = MODEL_REGISTRY[model_id]["size_gb"]
                min_required_bytes = int(expected_size_gb * 0.95 * 1000 * 1000 * 1000)
                if actual_size >= min_required_bytes:
                    mark_model_verified(model_id, actual_size)
                    return "ready"
        except (OSError, IOError) as e:
            print(f"[MODEL] Error checking {model_id}/model.pt: {e}")

    model_file_st = folder_path / "model.safetensors"
    if model_file_st.exists():
        try:
            actual_size = model_file_st.stat().st_size
            if is_model_verified(model_id):
                verified_size = get_verified_model_size(model_id)
                if verified_size > 0 and actual_size == verified_size:
                    return "ready"

            expected_sizes = get_expected_file_sizes(model_id)
            expected_size = expected_sizes.get('model.safetensors', 0)

            if expected_size > 0:
                size_diff_pct = abs(actual_size - expected_size) / expected_size * 100
                if size_diff_pct < 0.1:
                    mark_model_verified(model_id, actual_size)
                    return "ready"
            else:
                expected_size_gb = MODEL_REGISTRY[model_id]["size_gb"]
                min_required_bytes = int(expected_size_gb * 0.95 * 1000 * 1000 * 1000)
                if actual_size >= min_required_bytes:
                    mark_model_verified(model_id, actual_size)
                    return "ready"
        except (OSError, IOError):
            pass

    return "not_downloaded"


def get_model_status_quick(model_id: str) -> str:
    """Fast model status check (no HuggingFace API calls)."""
    if model_id in download_states and download_states[model_id].get("status") == "downloading":
        return "downloading"

    if model_id not in MODEL_REGISTRY:
        return "not_downloaded"

    folder_path = BASE_DIR / model_id
    if not folder_path.exists():
        return "not_downloaded"

    if is_model_verified(model_id):
        model_file = folder_path / "model.pt"
        model_file_st = folder_path / "model.safetensors"
        if model_file.exists() or model_file_st.exists():
            return "ready"

    expected_size_gb = MODEL_REGISTRY[model_id]["size_gb"]
    min_size_bytes = int(expected_size_gb * 0.9 * 1000 * 1000 * 1000)

    model_file = folder_path / "model.pt"
    if model_file.exists():
        try:
            if model_file.stat().st_size >= min_size_bytes:
                return "ready"
        except:
            pass

    model_file_st = folder_path / "model.safetensors"
    if model_file_st.exists():
        try:
            if model_file_st.stat().st_size >= min_size_bytes:
                return "ready"
        except:
            pass

    return "not_downloaded"


def is_model_ready_quick(model_id: str) -> bool:
    """Quick check if model is likely ready."""
    return get_model_status_quick(model_id) == "ready"


def get_download_progress(model_id: str) -> dict:
    """Get download progress for a model"""
    if model_id not in download_states:
        return {"status": "not_started", "progress": 0}
    return download_states[model_id]


def get_recommended_model(refresh: bool = False) -> Optional[str]:
    """Get the recommended model based on available VRAM.

    Args:
        refresh: If True, refresh GPU info before recommending
    """
    from gpu import gpu_info, refresh_gpu_info

    current_gpu_info = refresh_gpu_info() if refresh else gpu_info

    if not current_gpu_info['available']:
        return "songgeneration_base"

    vram = current_gpu_info['gpu']['total_gb']
    print(f"[MODEL] Recommending based on {vram}GB total VRAM")

    suitable_models = []
    for model_id, info in MODEL_REGISTRY.items():
        if info['vram_required'] <= vram:
            suitable_models.append((model_id, info['priority']))

    if not suitable_models:
        return "songgeneration_base"

    suitable_models.sort(key=lambda x: x[1], reverse=True)
    return suitable_models[0][0]


def get_best_ready_model(refresh: bool = False) -> Optional[str]:
    """Get the best model that is both ready AND fits in VRAM.

    This is the single source of truth for model selection.
    """
    from gpu import gpu_info, refresh_gpu_info

    current_gpu_info = refresh_gpu_info() if refresh else gpu_info

    if not current_gpu_info['available']:
        vram = 0
    else:
        vram = current_gpu_info['gpu']['total_gb']

    print(f"[MODEL] Selecting best ready model for {vram}GB total VRAM")

    # Get ready models that fit in VRAM, sorted by priority (highest first)
    suitable_ready = []
    for model_id, info in MODEL_REGISTRY.items():
        status = get_model_status_quick(model_id)
        if status == "ready" and info['vram_required'] <= vram:
            suitable_ready.append((model_id, info['priority']))

    if not suitable_ready:
        # Fallback: return any ready model (user may have limited VRAM)
        for model_id, info in MODEL_REGISTRY.items():
            status = get_model_status_quick(model_id)
            if status == "ready":
                return model_id
        return None

    suitable_ready.sort(key=lambda x: x[1], reverse=True)
    best = suitable_ready[0][0]
    print(f"[MODEL] Best ready model: {best}")
    return best


def get_available_models_sync() -> List[dict]:
    """Get only ready models (sync version for startup)"""
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        status = get_model_status_quick(model_id)
        if status == "ready":
            models.append({"id": model_id, "name": info["name"], "status": status})
    return models

# ============================================================================
# Download Management
# ============================================================================

def run_model_download(model_id: str, notify_callback=None):
    """Run model download in background thread."""
    global download_states, download_cancel_flags, download_processes

    if model_id not in MODEL_REGISTRY:
        download_states[model_id] = {"status": "error", "error": "Unknown model"}
        return

    model_info = MODEL_REGISTRY[model_id]
    hf_repo = model_info["hf_repo"]
    local_dir = BASE_DIR / model_id

    print(f"[DOWNLOAD] Starting download of {model_id} from {hf_repo}")

    expected_sizes = get_expected_file_sizes(model_id)
    total_bytes = expected_sizes.get('__total__', 0)
    actual_size_gb = total_bytes / (1024 * 1024 * 1024) if total_bytes else model_info["size_gb"]

    download_states[model_id] = {
        "status": "downloading",
        "progress": 0,
        "downloaded_gb": 0,
        "total_gb": round(actual_size_gb, 2),
        "speed_mbps": 0,
        "eta_seconds": 0,
        "started_at": time.time(),
    }

    cancel_event = threading.Event()
    download_cancel_flags[model_id] = cancel_event

    process = None
    try:
        cmd = [
            sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
            "download", hf_repo,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks", "False"
        ]

        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            creationflags=creationflags
        )

        download_processes[model_id] = process
        print(f"[DOWNLOAD] Started subprocess PID {process.pid}")

        total_bytes = int(actual_size_gb * 1024 * 1024 * 1024)
        last_downloaded = 0
        last_time = time.time()
        poll_interval = 1.0

        while process.poll() is None:
            if cancel_event.is_set():
                print(f"[DOWNLOAD] Cancel requested for {model_id}")
                try:
                    if sys.platform == 'win32':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                      capture_output=True, timeout=10)
                    else:
                        process.terminate()
                        process.wait(timeout=5)
                except Exception as e:
                    print(f"[DOWNLOAD] Error terminating process: {e}")
                    try:
                        process.kill()
                    except:
                        pass

                download_states[model_id] = {"status": "cancelled", "progress": 0}
                if local_dir.exists():
                    shutil.rmtree(local_dir, ignore_errors=True)
                return

            current_time = time.time()
            current_downloaded = get_directory_size(local_dir)

            if total_bytes > 0:
                progress = min(99, int((current_downloaded / total_bytes) * 100))
            else:
                progress = 0

            downloaded_gb = current_downloaded / (1024 * 1024 * 1024)

            time_diff = current_time - last_time
            if time_diff > 0:
                bytes_diff = current_downloaded - last_downloaded
                speed_mbps = (bytes_diff / time_diff) / (1024 * 1024)

                if speed_mbps > 0:
                    remaining_bytes = total_bytes - current_downloaded
                    eta_seconds = remaining_bytes / (speed_mbps * 1024 * 1024)
                else:
                    eta_seconds = 0

                if model_id in download_states:
                    download_states[model_id].update({
                        "progress": progress,
                        "downloaded_gb": round(downloaded_gb, 2),
                        "speed_mbps": round(speed_mbps, 1),
                        "eta_seconds": int(eta_seconds),
                    })

            last_downloaded = current_downloaded
            last_time = current_time
            time.sleep(poll_interval)

        if cancel_event.is_set():
            return

        exit_code = process.returncode
        print(f"[DOWNLOAD] Process finished with exit code {exit_code}")

        # Verify download
        expected_sizes = get_expected_file_sizes(model_id)
        expected_model_size = expected_sizes.get('model.pt', 0) or expected_sizes.get('model.safetensors', 0)

        model_file = local_dir / "model.pt"
        if not model_file.exists():
            model_file = local_dir / "model.safetensors"

        download_complete = False
        actual_model_size = 0

        if model_file.exists():
            try:
                actual_model_size = model_file.stat().st_size
                if expected_model_size > 0:
                    download_complete = (actual_model_size == expected_model_size)
                else:
                    min_bytes = int(model_info["size_gb"] * 0.99 * 1024 * 1024 * 1024)
                    download_complete = (actual_model_size >= min_bytes)
            except (OSError, IOError) as e:
                print(f"[DOWNLOAD] Error checking model file: {e}")

        final_size = get_directory_size(local_dir)
        final_gb = final_size / (1024 * 1024 * 1024)

        if download_complete:
            download_states[model_id] = {
                "status": "completed",
                "progress": 100,
                "downloaded_gb": round(final_gb, 2),
                "total_gb": round(actual_size_gb, 2),
            }
            print(f"[DOWNLOAD] Successfully downloaded {model_id}")
            if notify_callback:
                notify_callback()
            expected_file_sizes_cache[model_id] = expected_sizes
        else:
            download_states[model_id] = {
                "status": "error",
                "error": f"Download incomplete",
                "progress": download_states[model_id].get("progress", 0) if model_id in download_states else 0,
            }
            if local_dir.exists() and final_gb < 0.5:
                shutil.rmtree(local_dir, ignore_errors=True)

    except Exception as e:
        print(f"[DOWNLOAD] Error downloading {model_id}: {e}")
        import traceback
        traceback.print_exc()
        download_states[model_id] = {"status": "error", "error": str(e)}
    finally:
        download_cancel_flags.pop(model_id, None)
        download_processes.pop(model_id, None)
        download_threads.pop(model_id, None)


def start_model_download(model_id: str, notify_callback=None) -> dict:
    """Start downloading a model in the background"""
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}

    with download_start_lock:
        current_status = get_model_status(model_id)
        if current_status == "ready":
            return {"error": "Model already downloaded"}
        if current_status == "downloading":
            return {"error": "Model is already downloading"}

        download_states[model_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Initializing download..."
        }

        thread = threading.Thread(
            target=run_model_download,
            args=(model_id, notify_callback),
            daemon=True
        )
        download_threads[model_id] = thread
        thread.start()

    return {"status": "started", "model_id": model_id}


def cancel_model_download(model_id: str) -> dict:
    """Cancel an ongoing download"""
    if model_id not in download_states or download_states[model_id].get("status") != "downloading":
        return {"error": "No active download for this model"}

    if model_id in download_cancel_flags:
        download_cancel_flags[model_id].set()

    if model_id in download_processes:
        process = download_processes[model_id]
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                              capture_output=True, timeout=10)
            else:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        except Exception as e:
            print(f"[DOWNLOAD] Error killing process: {e}")

    download_states[model_id] = {"status": "cancelled", "progress": 0}

    local_dir = BASE_DIR / model_id
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)

    download_processes.pop(model_id, None)
    download_threads.pop(model_id, None)
    download_states.pop(model_id, None)

    return {"status": "cancelled", "model_id": model_id}


def delete_model(model_id: str) -> dict:
    """Delete a downloaded model to free up space"""
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}

    status = get_model_status(model_id)
    if status == "downloading":
        return {"error": "Cannot delete model while downloading"}
    if status == "not_downloaded":
        return {"error": "Model is not downloaded"}

    folder_path = BASE_DIR / model_id
    try:
        shutil.rmtree(folder_path)
        print(f"[MODEL] Deleted model {model_id}")
        return {"status": "deleted", "model_id": model_id}
    except Exception as e:
        return {"error": f"Failed to delete: {e}"}


def cleanup_download_states():
    """Clear stale download states on startup"""
    download_states.clear()
    download_threads.clear()
    download_processes.clear()
    download_cancel_flags.clear()
    print(f"[CONFIG] Cleared stale download states")
