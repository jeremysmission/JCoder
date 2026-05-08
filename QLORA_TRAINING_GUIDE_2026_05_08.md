# QLoRA Fine-Tuning Guide: Qwen3.6-27B on RTX 3090 (Windows)

**Author:** Jeremy Randall / CoPilot+
**Hardware:** Beast — Dual RTX 3090 (24GB each), Windows 11, CUDA 12.8
**Date:** 2026-05-08
**Model:** Qwen3.6-27B (released April 2026, 77.2% SWE-bench Verified)

---

## What This Guide Does

Fine-tunes Qwen3.6-27B using QLoRA (4-bit quantized LoRA adapters) on a single
RTX 3090. The trained model exports to GGUF format and imports back into Ollama
as a custom model called `jcoder-qwen27b`. Total time: ~7-9 hours for first run,
~6-8 hours for subsequent runs (model is cached after first download).

## Prerequisites

Before starting, you need:

- **Windows 11** with NVIDIA driver supporting CUDA 12.8+
- **RTX 3090** (24GB VRAM) — single GPU is used for training
- **Python 3.12** installed system-wide
- **Ollama** installed (for importing the final model)
- **~80GB free disk space** on C: drive (model weights + checkpoints + GGUF)
- **Internet connection** (first run downloads ~15-54GB of model weights)

---

## Step 0: Kill Ollama (Free GPU Memory)

**What this does:** Ollama keeps models loaded in GPU memory. On a dual-GPU
system, it splits the model across both cards (~13GB each). Training needs the
full 24GB on one GPU, so Ollama must be stopped first.

**What to watch:** Your V2 import or other CPU-only tasks are NOT affected.
Only GPU-dependent tasks (LLM queries, embeddings) will stop working.

```powershell
taskkill /IM "ollama.exe" /F
```

Verify both GPUs are mostly free:

```powershell
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader
```

You should see VRAM under 1000 MiB on each GPU. If not, wait 10 seconds and
check again.

---

## Step 1: Create the Training Virtual Environment

**What this does:** Creates a separate Python virtual environment at
`C:\QLoRA_Training\.venv` specifically for training. This keeps training
dependencies (Unsloth, TRL, etc.) isolated from JCoder's main `.venv`.

**Do this once. Skip on subsequent runs.**

```powershell
python -m venv C:\QLoRA_Training\.venv
```

---

## Step 2: Install PyTorch with CUDA 12.8

**What this does:** Installs PyTorch 2.7.1 with CUDA 12.8 support. This is the
proven version on Beast — do NOT use cu124 or any other CUDA version.

**IMPORTANT:** PyTorch must be installed BEFORE Unsloth. If you reverse this
order, Unsloth will pull a CPU-only PyTorch from PyPI and overwrite the CUDA
version.

**Do this once. Skip on subsequent runs unless you need to reinstall.**

```powershell
C:\QLoRA_Training\.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

Verify CUDA works:

```powershell
$env:PYTHONUTF8 = "1"
C:\QLoRA_Training\.venv\Scripts\python.exe -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}')"
```

Expected output: `torch=2.7.1+cu128 cuda=12.8 gpus=2`

---

## Step 3: Install Unsloth and Training Dependencies

**What this does:** Installs Unsloth (2x faster fine-tuning, 50% less VRAM),
TRL (training library), bitsandbytes (4-bit quantization), and all satellite
dependencies with version-matched extras.

**IMPORTANT:** Use the `[cu128-torch271]` extras syntax. Do NOT use bare
`pip install unsloth` — it pulls mismatched dependency versions that cause
silent crashes on Windows.

**Do this once. Skip on subsequent runs.**

```powershell
C:\QLoRA_Training\.venv\Scripts\pip.exe install "unsloth[cu128-torch271]"
```

Then install SSL support (needed for HuggingFace downloads on Beast):

```powershell
C:\QLoRA_Training\.venv\Scripts\pip.exe install truststore pip-system-certs
```

Verify everything loads:

```powershell
$env:PYTHONUTF8 = "1"
C:\QLoRA_Training\.venv\Scripts\python.exe -c "import datasets, torch, triton, bitsandbytes, xformers; from unsloth import FastLanguageModel; print('ALL OK')"
```

Expected output: Unsloth banner message + `ALL OK`

**WINDOWS DLL FIX:** If you get exit code 5 (silent crash) when importing
Unsloth, the fix is import order. Preload compiled extensions BEFORE importing
Unsloth:

```python
import datasets, torch, triton, bitsandbytes, xformers  # MUST come first
from unsloth import FastLanguageModel  # THEN import Unsloth
```

This is a Windows DLL loading order issue, not an installation problem. The
smoke test script already includes this fix.

---

## Step 4: Download the Model (First Run Only)

**What this does:** Downloads the Qwen3.6-27B model weights from HuggingFace in
safetensors format. Ollama has its own GGUF copy, but Unsloth needs HuggingFace
format for training. The download is cached at `C:\hybridrag_cache\huggingface\`
and reused on future runs.

**Size:** ~15-54GB depending on whether Unsloth downloads the full BF16 weights
or a pre-quantized variant. First download takes 5-30 minutes depending on
internet speed.

**What to watch:** If the download stalls or fails with SSL errors, make sure
`truststore` is installed (Step 3) and that the firewall isn't blocking
HuggingFace.

```powershell
$env:PYTHONUTF8 = "1"
$env:HF_HOME = "C:\hybridrag_cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
C:\QLoRA_Training\.venv\Scripts\python.exe -c "
import truststore; truststore.inject_into_ssl()
from huggingface_hub import snapshot_download
snapshot_download('unsloth/Qwen3.6-27B', cache_dir=r'C:\hybridrag_cache\huggingface\hub')
print('Download complete')
"
```

Verify the download:

```powershell
Get-ChildItem "C:\hybridrag_cache\huggingface\hub" | Where-Object { $_.Name -like "*qwen*" -or $_.Name -like "*Qwen*" }
```

You should see a `models--unsloth--Qwen3.6-27B` directory.

---

## Step 5: Run the Smoke Test (~20 minutes)

**What this does:** Proves the entire pipeline works end-to-end with minimal
data before committing to hours of full training. Loads the model in 4-bit,
trains on 5 hardcoded examples for 10 steps, exports to GGUF, imports to
Ollama, and runs a canary comparison test.

**What to watch:**
- VRAM usage should peak at 20-23GB (if it OOMs, another process is using GPU)
- Training loss should decrease over 10 steps
- GGUF file should be ~15-16GB
- Total time should be ~15-25 minutes

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HOME = "C:\hybridrag_cache\huggingface"
cd C:\Users\jerem\JCoder
C:\QLoRA_Training\.venv\Scripts\python.exe scripts\smoke_test_qlora.py
```

The script will:
1. Auto-select the best GPU (lowest VRAM used, then coolest temperature)
2. Load Qwen3.6-27B in 4-bit (~16-18GB VRAM)
3. Configure LoRA adapters (rank=8, minimal for smoke test)
4. Train 10 steps on 5 examples
5. Export to GGUF Q4_K_M format
6. Import to Ollama as `jcoder-smoke-test`
7. Run a canary prompt on both base and fine-tuned models

Results are saved to `models/smoke_test/smoke_test_report.json`.

**If the smoke test passes, the pipeline works. Proceed to full training.**

---

## Step 6: Prepare Training Data

**What this does:** Merges all training Q&A pairs into a single file, then
converts to ChatML format (required by Qwen3.6).

### 6a: Merge Training Corpus

```powershell
$env:PYTHONUTF8 = "1"
C:\QLoRA_Training\.venv\Scripts\python.exe -c "
import json
from pathlib import Path

all_data = []
for batch_dir in sorted(Path('evaluation').glob('results_codex_marathon*')):
    results_file = batch_dir / 'marathon_results.json'
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            batch = json.load(f)
        # Filter out timeouts and empties
        good = [x for x in batch if x.get('answer') and len(x['answer']) > 100]
        all_data.extend(good)
        print(f'{batch_dir.name}: {len(good)}/{len(batch)} usable')

# Deduplicate by ID
seen = set()
unique = []
for item in all_data:
    if item['id'] not in seen:
        seen.add(item['id'])
        unique.append(item)

Path('data').mkdir(exist_ok=True)
with open('data/training_corpus_full.json', 'w', encoding='utf-8') as f:
    json.dump(unique, f, indent=2)
print(f'Total: {len(unique)} unique training examples')
"
```

### 6b: Convert to ChatML Format

```powershell
$env:PYTHONUTF8 = "1"
C:\QLoRA_Training\.venv\Scripts\python.exe -c "
import json

with open('data/training_corpus_full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

chatml = []
for item in data:
    chatml.append({
        'messages': [
            {'role': 'system', 'content': 'You are JCoder, a Python coding assistant specialized in RAG systems.'},
            {'role': 'user', 'content': item['question']},
            {'role': 'assistant', 'content': item['answer']},
        ]
    })

with open('data/training_chatml.json', 'w', encoding='utf-8') as f:
    json.dump(chatml, f, indent=2)
print(f'Converted {len(chatml)} examples to ChatML format')
"
```

---

## Step 7: Run Full Training (~6-8 hours)

**What this does:** Fine-tunes the full model on all training data for 2 epochs.
This is the real training run — plan to start it before bed and let it run
overnight.

**What to watch:**
- Training loss should start at ~2-3 and decrease to ~0.5-1.0
- VRAM should stay under 23GB (if it OOMs, reduce max_seq_length to 1024)
- Machine should stay cool (RTX 3090 throttles at 83C)
- Do NOT open other GPU-heavy programs during training

**Hyperparameters:**

| Parameter | Value | Why |
|:---|:---|:---|
| LoRA rank | 16 | Good balance of quality vs VRAM |
| LoRA alpha | 32 | 2x rank (standard) |
| Learning rate | 1e-4 | Conservative for small datasets |
| Batch size | 1 | No headroom at 24GB |
| Gradient accumulation | 8 | Effective batch size = 8 |
| Epochs | 2 | Max 3, beyond risks overfitting |
| Max seq length | 2048 | Longer = more VRAM, 2048 is safe |
| Optimizer | adamw_8bit | Half-precision optimizer saves VRAM |
| Gradient checkpointing | ON (mandatory) | Trades compute for VRAM |

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HOME = "C:\hybridrag_cache\huggingface"
cd C:\Users\jerem\JCoder
C:\QLoRA_Training\.venv\Scripts\python.exe scripts\full_training_qlora.py
```

**Note:** The full training script (`scripts/full_training_qlora.py`) follows
the same pattern as the smoke test but uses real training data, higher LoRA
rank, and runs for 2 full epochs. See the smoke test script for the code
pattern.

---

## Step 8: Export to GGUF and Import to Ollama

**What this does:** After training completes, the LoRA adapters are merged back
into the base model and exported as a GGUF file (the format Ollama uses).

This happens automatically at the end of the training script. If you need to
do it manually:

```python
# Inside your training script, after training:
model.save_pretrained_gguf(
    "models/jcoder_qlora/gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

Then import to Ollama:

```powershell
# Create a Modelfile pointing to the GGUF
$gguf = Get-ChildItem "models\jcoder_qlora\gguf\*.gguf" | Select-Object -First 1
@"
FROM $($gguf.FullName)
PARAMETER num_ctx 131072
PARAMETER temperature 0
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
SYSTEM "You are JCoder, a Python coding assistant specialized in RAG systems."
"@ | Set-Content "models\jcoder_qlora\Modelfile" -Encoding UTF8

# Import to Ollama
ollama create jcoder-qwen27b -f models\jcoder_qlora\Modelfile
```

---

## Step 9: Test the Trained Model

**What this does:** Runs the trained model through 60 previously-failed Qwen
questions to measure improvement.

```powershell
# Restart Ollama (it was killed in Step 0)
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

# Wait for Ollama to start
Start-Sleep -Seconds 5

# Quick manual test
ollama run jcoder-qwen27b "Write a Python function to deduplicate text chunks using content hashing."

# Full retest against failure set
$env:PYTHONUTF8 = "1"
C:\QLoRA_Training\.venv\Scripts\python.exe scripts\run_eval_local.py --model jcoder-qwen27b --eval evaluation/qwen_failure_retest.json
```

Compare scores between base `qwen3.6:27b-q4_K_M` and trained `jcoder-qwen27b`.
Look for improvement on the specific failure categories (rag_ingestion,
python_advanced, rl_training, python_async).

---

## Step 10: Restart Ollama for Normal Use

```powershell
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
```

Verify your models are available:

```powershell
ollama list
```

You should see both `qwen3.6:27b-q4_K_M` (base) and `jcoder-qwen27b` (trained).

---

## Troubleshooting

### Exit Code 5 (Silent Crash) When Importing Unsloth

This is a Windows DLL loading order issue. Fix: preload compiled extensions
before importing Unsloth. Add these lines at the top of your script:

```python
import datasets, torch, triton, bitsandbytes, xformers
from unsloth import FastLanguageModel  # AFTER preloading
```

### SSL Certificate Errors During Model Download

Beast requires `truststore` for HuggingFace downloads:

```powershell
C:\QLoRA_Training\.venv\Scripts\pip.exe install truststore pip-system-certs
```

Then in your script:

```python
import truststore
truststore.inject_into_ssl()
```

### Unsloth Pulls Wrong PyTorch Version

If `pip install unsloth` overwrites your CUDA PyTorch with CPU-only:

```powershell
C:\QLoRA_Training\.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

### OOM During Training

Reduce memory usage:
1. Set `max_seq_length=1024` (from 2048)
2. Verify `use_gradient_checkpointing="unsloth"` is ON
3. Kill ALL other GPU processes: `taskkill /IM "ollama.exe" /F`
4. Check nvidia-smi — nothing should be using VRAM

### UnicodeEncodeError in TRL/Jinja Templates

Set `PYTHONUTF8=1` before running anything:

```powershell
$env:PYTHONUTF8 = "1"
```

### Unsloth Says "Not Recommended" for 4-bit QLoRA on Qwen3.5

Qwen3.6 may inherit this warning from Qwen3.5. If training quality is poor
(loss doesn't decrease, model outputs garbage), the fallback is bf16 LoRA
which requires dual GPU (48GB total) via WSL2. The smoke test catches this
in 20 minutes before you commit to hours of training.

---

## Environment Variables (Set Before Every Run)

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HOME = "C:\hybridrag_cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

---

## Key File Locations

| File | Purpose |
|:---|:---|
| `C:\QLoRA_Training\.venv\` | Training virtual environment |
| `C:\hybridrag_cache\huggingface\` | HuggingFace model cache |
| `scripts/smoke_test_qlora.py` | 20-minute pipeline smoke test |
| `data/training_corpus_full.json` | Merged training examples |
| `data/training_chatml.json` | ChatML-formatted training data |
| `models/smoke_test/` | Smoke test output + GGUF |
| `models/jcoder_qlora/` | Full training output + GGUF |
| `evaluation/qwen_failure_retest.json` | 60-question post-training eval |

---

## Hardware Notes

- **GPU 0 vs GPU 1:** The smoke test auto-selects the GPU with lowest VRAM
  usage and coolest temperature. Do not hardcode a GPU — always check
  `nvidia-smi` numbers at runtime.
- **Ollama splits models across both GPUs** by default. Kill Ollama before
  training to free both cards.
- **Thermal throttling:** RTX 3090 throttles at 83C. During 6-8 hour training
  runs, monitor temperature. Beast's thermal mods should keep it under control.
- **Single GPU training only.** Multi-GPU (DDP) is broken in Unsloth on
  Windows. Do not attempt dual-GPU training without WSL2.

---

*Document by Jeremy Randall / CoPilot+, 2026-05-08*
