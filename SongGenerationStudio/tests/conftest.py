"""Test helpers: put SongGenerationStudio/ on path and run from it."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIO = HERE.parent  # SongGenerationStudio/

# Run from the module dir so top-level `from config import ...` resolves.
os.chdir(STUDIO)
sys.path.insert(0, str(STUDIO))