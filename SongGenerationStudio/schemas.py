"""
SongGeneration Studio - Pydantic Schemas
Data models for API requests and responses.
"""

from typing import Optional, List
from pydantic import BaseModel

# ============================================================================
# Request/Response Models
# ============================================================================

class Section(BaseModel):
    type: str
    lyrics: Optional[str] = None


class SongRequest(BaseModel):
    title: str = "Untitled"
    sections: List[Section]
    gender: str = "female"
    timbre: str = ""
    genre: str = ""
    emotion: str = ""
    instruments: str = ""
    custom_style: Optional[str] = None  # Additional free-text style descriptors
    bpm: int = 120
    output_mode: str = "mixed"
    auto_prompt_type: Optional[str] = None
    reference_audio_id: Optional[str] = None
    seed: Optional[int] = None           # reproducible generation (passed to worker)
    model: str = "songgeneration_base"
    memory_mode: str = "auto"
    # Advanced generation parameters
    cfg_coef: float = 1.5          # Classifier-free guidance (0.1-3.0)
    temperature: float = 0.8       # Sampling randomness (0.1-2.0)
    top_k: int = 50                # Top-K sampling (1-250)
    top_p: float = 0.0             # Nucleus sampling, 0 = disabled (0.0-1.0)
    extend_stride: int = 5         # Extension stride for longer songs


class UpdateGenerationRequest(BaseModel):
    title: Optional[str] = None
