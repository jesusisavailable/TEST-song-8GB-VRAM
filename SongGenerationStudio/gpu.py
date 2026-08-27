"""
SongGeneration Studio - GPU Detection
GPU and VRAM detection utilities.
"""

import subprocess
from typing import Optional
from pathlib import Path

# ============================================================================
# GPU/VRAM Detection
# ============================================================================

def get_gpu_info() -> dict:
    """Detect GPU and available VRAM."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'name': parts[0],
                        'total_mb': int(parts[1]),
                        'free_mb': int(parts[2]),
                        'used_mb': int(parts[3]),
                        'total_gb': round(int(parts[1]) / 1024, 1),
                        'free_gb': round(int(parts[2]) / 1024, 1),
                    })
            if gpus:
                gpu = gpus[0]  # Primary GPU
                if gpu['total_gb'] >= 24:
                    recommended = 'full'
                elif gpu['total_gb'] >= 10:
                    recommended = 'low'
                elif gpu['total_gb'] >= 8:
                    recommended = '8gb'
                else:
                    recommended = 'unsupported'
                return {
                    'available': True,
                    'gpu': gpu,
                    'recommended_mode': recommended,
                    'can_run_full': gpu['total_gb'] >= 24,
                    'can_run_low': gpu['total_gb'] >= 10,
                    # 8GB cards work with the quantization + CPU-offload profile.
                    'can_run_8gb': gpu['total_gb'] >= 8,
                }
    except Exception as e:
        print(f"[GPU] Detection error: {e}")

    return {
        'available': False,
        'gpu': None,
        'recommended_mode': 'low',
        'can_run_full': False,
        'can_run_low': False,
        'can_run_8gb': False,
    }

# ============================================================================
# Audio Duration Helper
# ============================================================================

def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"[AUDIO] Failed to get duration for {audio_path}: {e}")
    return None

# Global GPU info - initialized on import
gpu_info = get_gpu_info()

def log_gpu_info():
    """Log GPU detection results"""
    if gpu_info['available']:
        print(f"[GPU] Detected: {gpu_info['gpu']['name']}")
        print(f"[GPU] VRAM: {gpu_info['gpu']['free_gb']}GB free / {gpu_info['gpu']['total_gb']}GB total")
        print(f"[GPU] Recommended mode: {gpu_info['recommended_mode']}")
        if gpu_info['recommended_mode'] == '8gb':
            print("[GPU] 8GB VRAM mode: using quantization + CPU offload profile")
    else:
        print("[GPU] No NVIDIA GPU detected or nvidia-smi not available")

def refresh_gpu_info() -> dict:
    """Refresh GPU info and return updated data"""
    global gpu_info
    gpu_info = get_gpu_info()
    return gpu_info
