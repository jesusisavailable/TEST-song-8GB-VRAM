# Background dependency installer for SongGeneration Studio (8GB VRAM)
$ErrorActionPreference = 'Continue'
cd 'H:\Projetos\Coding\SongGeneration-Studio'
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'

Write-Output "=== [1/4] Installing PyTorch CUDA 12.8 ==="
uv pip install -p $py torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --no-deps
if ($LASTEXITCODE -ne 0) { Write-Output "ERROR torch install: $LASTEXITCODE" }

Write-Output "=== [2/4] Installing requirements.txt ==="
uv pip install -p $py -r app/requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Output "ERROR requirements install: $LASTEXITCODE" }

Write-Output "=== [3/4] Installing requirements_nodeps.txt (--no-deps) ==="
uv pip install -p $py -r app/requirements_nodeps.txt --no-deps
if ($LASTEXITCODE -ne 0) { Write-Output "ERROR nodeps install: $LASTEXITCODE" }

Write-Output "=== [4/4] Installing bitsandbytes ==="
uv pip install -p $py 'bitsandbytes>=0.44.0'
if ($LASTEXITCODE -ne 0) { Write-Output "ERROR bitsandbytes install: $LASTEXITCODE" }

Write-Output "=== DONE. Verifying imports ==="
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
& $py -c "import transformers, tokenizers; print('transformers', transformers.__version__, 'tokenizers', tokenizers.__version__)"
Write-Output "=== INSTALL SCRIPT COMPLETE ==="
