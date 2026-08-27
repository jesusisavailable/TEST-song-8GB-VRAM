"""
SongGeneration Studio - Model Server Communication
Persistent model in VRAM for fast generation.
"""

import os
import sys
import time
import asyncio
import subprocess
from typing import Optional

import requests

from config import BASE_DIR, MODEL_SERVER_PORT, MODEL_SERVER_URL

# ============================================================================
# Model Server Process
# ============================================================================

model_server_process: Optional[subprocess.Popen] = None

# Cache for model server status to reduce HTTP calls
_status_cache = {"data": None, "timestamp": 0}
_STATUS_CACHE_TTL = 2.0  # Cache status for 2 seconds


def invalidate_status_cache():
    """Invalidate the status cache to force a fresh fetch."""
    global _status_cache
    _status_cache = {"data": None, "timestamp": 0}

def is_model_server_running() -> bool:
    """Check if model server is running and responsive (blocking)."""
    try:
        resp = requests.get(f"{MODEL_SERVER_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


async def is_model_server_running_async() -> bool:
    """Check if model server is running (non-blocking)."""
    return await asyncio.to_thread(is_model_server_running)


def kill_process_on_port(port: int) -> bool:
    """Kill any process using the specified port."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"[MODEL_SERVER] Killing process {pid} on port {port}", flush=True)
                        subprocess.run(["taskkill", "/F", "/PID", pid],
                                      capture_output=True, timeout=10)
                        time.sleep(2)
                        return True
        else:
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout.strip():
                    pid = result.stdout.strip().split('\n')[0]
                    print(f"[MODEL_SERVER] Killing process {pid} on port {port}", flush=True)
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=10)
                    time.sleep(2)
                    return True
            except FileNotFoundError:
                subprocess.run(["fuser", "-k", f"{port}/tcp"],
                              capture_output=True, timeout=10)
                time.sleep(2)
                return True
    except Exception as e:
        print(f"[MODEL_SERVER] Failed to kill process on port {port}: {e}", flush=True)
    return False


def get_model_server_status() -> dict:
    """Get model server status including loaded model info (blocking).

    Uses a short-lived cache to prevent excessive HTTP calls.
    """
    global model_server_process, _status_cache

    # Check cache first
    now = time.time()
    if _status_cache["data"] is not None and (now - _status_cache["timestamp"]) < _STATUS_CACHE_TTL:
        return _status_cache["data"]

    # Fast path: if server process isn't running, return immediately
    if model_server_process is None or model_server_process.poll() is not None:
        # Server process not running - do a quick check just in case it was started externally
        try:
            resp = requests.get(f"{MODEL_SERVER_URL}/status", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                data["running"] = True
                _status_cache = {"data": data, "timestamp": now}
                return data
        except:
            pass
        result = {"loaded": False, "running": False, "error": "not_running"}
        _status_cache = {"data": result, "timestamp": now}
        return result

    # Server process should be running - do proper status check
    max_retries = 2
    timeout_seconds = 3

    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{MODEL_SERVER_URL}/status", timeout=timeout_seconds)
            if resp.status_code == 200:
                data = resp.json()
                data["running"] = True
                _status_cache = {"data": data, "timestamp": now}
                return data
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(0.3)
            continue
        except requests.exceptions.ConnectionError:
            result = {"loaded": False, "running": False, "error": "not_running"}
            _status_cache = {"data": result, "timestamp": now}
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.3)
            continue

    result = {"loaded": False, "running": False, "error": "timeout"}
    _status_cache = {"data": result, "timestamp": now}
    return result


async def get_model_server_status_async() -> dict:
    """Get model server status (non-blocking)."""
    return await asyncio.to_thread(get_model_server_status)


async def start_model_server(preload_model: str = None) -> bool:
    """Start the model server process (async)."""
    global model_server_process

    if await is_model_server_running_async():
        print("[MODEL_SERVER] Already running", flush=True)
        return True

    await asyncio.to_thread(kill_process_on_port, MODEL_SERVER_PORT)

    print("[MODEL_SERVER] Starting model server...", flush=True)

    if sys.platform == "win32":
        python_exe = BASE_DIR / "env" / "Scripts" / "python.exe"
    else:
        python_exe = BASE_DIR / "env" / "bin" / "python"

    if not python_exe.exists():
        python_exe = sys.executable

    print(f"[MODEL_SERVER] Using Python: {python_exe}", flush=True)

    cmd = [str(python_exe), str(BASE_DIR / "model_server.py"),
           "--port", str(MODEL_SERVER_PORT), "--host", "127.0.0.1"]

    if preload_model:
        cmd.extend(["--preload", preload_model])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    flow_vae_dir = BASE_DIR / "codeclm" / "tokenizer" / "Flow1dVAE"
    pathsep = os.pathsep
    env["PYTHONPATH"] = f"{BASE_DIR}{pathsep}{flow_vae_dir}{pathsep}{env.get('PYTHONPATH', '')}"

    print(f"[MODEL_SERVER] Command: {' '.join(cmd)}", flush=True)

    try:
        model_server_process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        for i in range(60):
            await asyncio.sleep(1)

            if model_server_process.poll() is not None:
                print(f"[MODEL_SERVER] Process exited with code {model_server_process.returncode}", flush=True)
                return False

            if await is_model_server_running_async():
                print(f"[MODEL_SERVER] Server started successfully after {i+1}s", flush=True)
                return True

            if i % 10 == 9:
                print(f"[MODEL_SERVER] Still waiting... ({i+1}s)", flush=True)

        print("[MODEL_SERVER] Server failed to start in time (60s timeout)", flush=True)
        try:
            model_server_process.terminate()
        except:
            pass
        return False

    except Exception as e:
        print(f"[MODEL_SERVER] Failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False


def stop_model_server():
    """Stop the model server process."""
    global model_server_process

    if model_server_process:
        model_server_process.terminate()
        try:
            model_server_process.wait(timeout=5)
        except:
            model_server_process.kill()
        model_server_process = None
        invalidate_status_cache()  # Server stopped
        print("[MODEL_SERVER] Server stopped")


def load_model_on_server(model_id: str) -> dict:
    """Request model server to load a model (blocking)."""
    try:
        resp = requests.post(f"{MODEL_SERVER_URL}/load",
                           json={"model_id": model_id}, timeout=5)
        invalidate_status_cache()  # State changed
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def load_model_on_server_async(model_id: str) -> dict:
    """Request model server to load a model (non-blocking)."""
    return await asyncio.to_thread(load_model_on_server, model_id)


def generate_via_server(input_jsonl: str, save_dir: str, gen_type: str = "mixed") -> dict:
    """Send generation request to model server (blocking)."""
    try:
        invalidate_status_cache()  # Generating state will change
        resp = requests.post(f"{MODEL_SERVER_URL}/generate",
                           json={
                               "input_jsonl": input_jsonl,
                               "save_dir": save_dir,
                               "gen_type": gen_type
                           }, timeout=1800)
        invalidate_status_cache()  # Generation completed
        return resp.json()
    except Exception as e:
        invalidate_status_cache()
        return {"error": str(e)}


async def generate_via_server_async(input_jsonl: str, save_dir: str, gen_type: str = "mixed") -> dict:
    """Send generation request to model server (non-blocking)."""
    return await asyncio.to_thread(generate_via_server, input_jsonl, save_dir, gen_type)


def cancel_generation_on_server() -> dict:
    """Request cancellation of current generation."""
    try:
        resp = requests.post(f"{MODEL_SERVER_URL}/cancel", timeout=5)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def cancel_generation_on_server_async() -> dict:
    """Request cancellation of current generation (async)."""
    return await asyncio.to_thread(cancel_generation_on_server)


def unload_model_on_server() -> dict:
    """Unload model from VRAM."""
    try:
        resp = requests.post(f"{MODEL_SERVER_URL}/unload", timeout=10)
        invalidate_status_cache()  # State changed
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Actual Model Server (when run as script)
# ============================================================================

if __name__ == "__main__":
    import argparse
    import gc
    import json
    import traceback
    from pathlib import Path
    from datetime import datetime

    import torch
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    import soundfile as sf

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SongGeneration Model Server")
    parser.add_argument("--port", type=int, default=42100, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--preload", type=str, default=None, help="Model to preload")
    args = parser.parse_args()

    # Add paths for imports
    APP_DIR = Path(__file__).parent
    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(APP_DIR / "tools" / "gradio"))
    sys.path.insert(0, str(APP_DIR / "codeclm" / "tokenizer" / "Flow1dVAE"))

    from levo_inference_lowmem import LeVoInference

    # Create FastAPI app
    server_app = FastAPI(title="SongGeneration Model Server")

    # Model state
    class ModelState:
        def __init__(self):
            self.model: Optional[LeVoInference] = None
            self.model_id: Optional[str] = None
            self.loading: bool = False
            self.error: Optional[str] = None
            self.cancel_requested: bool = False
            self.generating: bool = False

    state = ModelState()

    # Request models
    class LoadRequest(BaseModel):
        model_id: str

    class GenerateRequest(BaseModel):
        input_jsonl: str
        save_dir: str
        gen_type: str = "mixed"

    @server_app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok"}

    @server_app.get("/status")
    def status():
        """Get current model server status."""
        return {
            "loaded": state.model is not None,
            "model_id": state.model_id,
            "loading": state.loading,
            "error": state.error,
            "generating": state.generating,
            "cancel_requested": state.cancel_requested
        }

    @server_app.post("/cancel")
    def cancel():
        """Request cancellation of current generation."""
        if state.generating:
            state.cancel_requested = True
            print("[MODEL_SERVER] Cancel requested", flush=True)
            return {"status": "cancel_requested"}
        return {"status": "not_generating"}

    @server_app.post("/load")
    def load_model(req: LoadRequest):
        """Load a model into VRAM."""
        if state.loading:
            return {"error": "Already loading a model"}

        if state.model is not None and state.model_id == req.model_id:
            return {"status": "already_loaded", "model_id": req.model_id}

        try:
            state.loading = True
            state.error = None

            # Unload existing model first
            if state.model is not None:
                del state.model
                state.model = None
                state.model_id = None
                gc.collect()
                torch.cuda.empty_cache()

            model_path = APP_DIR / req.model_id
            if not model_path.exists():
                state.error = f"Model not found: {req.model_id}"
                state.loading = False
                return {"error": state.error}

            print(f"[MODEL_SERVER] Loading model: {req.model_id}", flush=True)
            state.model = LeVoInference(str(model_path))
            state.model_id = req.model_id
            state.loading = False
            print(f"[MODEL_SERVER] Model loaded: {req.model_id}", flush=True)

            return {"status": "loaded", "model_id": req.model_id}

        except Exception as e:
            state.error = str(e)
            state.loading = False
            print(f"[MODEL_SERVER] Failed to load model: {e}", flush=True)
            traceback.print_exc()
            return {"error": str(e)}

    @server_app.post("/generate")
    def generate(req: GenerateRequest):
        """Generate audio from input."""
        if state.model is None:
            return {"error": "No model loaded"}

        # Reject concurrent generation requests
        if state.generating:
            print("[MODEL_SERVER] Rejecting request - generation already in progress", flush=True)
            return {"error": "Generation already in progress", "status": "busy"}

        # Reset cancel flag and set generating
        state.cancel_requested = False
        state.generating = True

        try:
            print(f"[MODEL_SERVER] Starting generation...", flush=True)

            # Read input JSONL
            with open(req.input_jsonl, 'r', encoding='utf-8') as f:
                input_data = json.loads(f.readline())

            lyric = input_data.get("gt_lyric", "")
            description = input_data.get("descriptions", None)
            prompt_audio = input_data.get("prompt_audio_path", None)
            auto_prompt_type = input_data.get("auto_prompt_audio_type", None)

            # Get advanced generation parameters
            gen_params = {}
            if "cfg_coef" in input_data:
                gen_params["cfg_coef"] = input_data["cfg_coef"]
            if "temperature" in input_data:
                gen_params["temperature"] = input_data["temperature"]
            if "top_k" in input_data:
                gen_params["top_k"] = input_data["top_k"]
            if "top_p" in input_data:
                gen_params["top_p"] = input_data["top_p"]
            if "extend_stride" in input_data:
                gen_params["extend_stride"] = input_data["extend_stride"]

            # Get auto prompt path
            auto_prompt_path = None
            if auto_prompt_type and auto_prompt_type != "Auto":
                auto_prompt_path = str(APP_DIR / "tools" / "new_prompt.pt")

            print(f"[MODEL_SERVER] Lyric: {lyric[:100]}...", flush=True)
            print(f"[MODEL_SERVER] Description: {description}", flush=True)
            print(f"[MODEL_SERVER] Gen type: {req.gen_type}", flush=True)
            if gen_params:
                print(f"[MODEL_SERVER] Gen params: {gen_params}", flush=True)

            # Run inference
            start_time = time.time()
            audio_result = state.model(
                lyric=lyric,
                description=description,
                prompt_audio_path=prompt_audio,
                genre=auto_prompt_type,
                auto_prompt_path=auto_prompt_path,
                gen_type=req.gen_type,
                params=gen_params
            )
            gen_time = time.time() - start_time

            # Check if cancelled during generation
            if state.cancel_requested:
                print(f"[MODEL_SERVER] Generation cancelled after {gen_time:.1f}s", flush=True)
                state.generating = False
                state.cancel_requested = False
                return {"status": "cancelled", "message": "Generation was cancelled"}

            print(f"[MODEL_SERVER] Generation completed in {gen_time:.1f}s", flush=True)

            # Save audio
            save_dir = Path(req.save_dir)
            audios_dir = save_dir / "audios"
            audios_dir.mkdir(parents=True, exist_ok=True)

            sample_rate = state.model.cfg.sample_rate
            idx = input_data.get('idx', 'output')

            # Handle 'separate' mode: audio_result is a dict with 'mixed', 'vocal', 'bgm' keys
            if req.gen_type == 'separate' and isinstance(audio_result, dict):
                output_file = audios_dir / f"{idx}.flac"
                output_file_vocal = audios_dir / f"{idx}_vocal.flac"
                output_file_bgm = audios_dir / f"{idx}_bgm.flac"

                audio_np = audio_result['mixed'].cpu().permute(1, 0).float().numpy()
                sf.write(str(output_file), audio_np, sample_rate)

                audio_np_vocal = audio_result['vocal'].cpu().permute(1, 0).float().numpy()
                sf.write(str(output_file_vocal), audio_np_vocal, sample_rate)

                audio_np_bgm = audio_result['bgm'].cpu().permute(1, 0).float().numpy()
                sf.write(str(output_file_bgm), audio_np_bgm, sample_rate)

                print(f"[MODEL_SERVER] Saved to: {output_file}, {output_file_vocal}, {output_file_bgm}", flush=True)
            else:
                # Single output mode (mixed, vocal, or bgm)
                audio_np = audio_result.cpu().permute(1, 0).float().numpy()
                output_file = audios_dir / f"{idx}.flac"
                sf.write(str(output_file), audio_np, sample_rate)
                print(f"[MODEL_SERVER] Saved to: {output_file}", flush=True)

            state.generating = False

            return {
                "status": "completed",
                "output_file": str(output_file),
                "generation_time": gen_time
            }

        except Exception as e:
            print(f"[MODEL_SERVER] Generation failed: {e}", flush=True)
            traceback.print_exc()
            state.generating = False
            return {"error": str(e)}

    @server_app.post("/unload")
    def unload():
        """Unload model from VRAM."""
        try:
            if state.model is not None:
                del state.model
                state.model = None
                state.model_id = None
                gc.collect()
                torch.cuda.empty_cache()
                print("[MODEL_SERVER] Model unloaded", flush=True)
            return {"status": "unloaded"}
        except Exception as e:
            return {"error": str(e)}

    # Preload model if specified
    if args.preload:
        print(f"[MODEL_SERVER] Preloading model: {args.preload}", flush=True)
        load_model(LoadRequest(model_id=args.preload))

    # Run server
    print(f"[MODEL_SERVER] Starting server on {args.host}:{args.port}", flush=True)
    uvicorn.run(server_app, host=args.host, port=args.port, log_level="info")
