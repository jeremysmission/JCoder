# JCODER QLORA SMOKE TEST
# Proves entire pipeline: install -> load -> train -> GGUF -> Ollama -> verify
# Should complete in ~20 minutes on a single 3090.
# Run from: C:/Users/jerem/JCoder
# Venv: C:/QLoRA_Training/.venv/Scripts/python.exe scripts/smoke_test_qlora.py

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Windows DLL loading order matters — preload compiled extensions
# BEFORE importing unsloth to avoid exit code 5 crashes.
import datasets
import torch
import triton
import bitsandbytes
import xformers

import json
import subprocess
import sys
import time
from pathlib import Path


def pick_best_gpu() -> int:
    """Pick the best GPU for training: lowest VRAM used, then coolest temp as tiebreaker."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        gpus = []
        for line in lines:
            parts = line.split(",")
            gpus.append((int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())))
        best = min(gpus, key=lambda g: (g[1], g[2]))
        print("GPU selection (lowest VRAM, then coolest temp):")
        for idx, mem, temp in gpus:
            marker = " <-- selected" if idx == best[0] else ""
            print(f"  GPU {idx}: {mem} MiB used, {temp}C{marker}")
        return best[0]
    except Exception as e:
        print(f"nvidia-smi failed ({e}), defaulting to GPU 0")
        return 0


TRAINING_GPU = pick_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(TRAINING_GPU)
print(f"Training on GPU {TRAINING_GPU}")

OUTPUT_DIR = Path("models/smoke_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_DATA = [
    {
        "messages": [
            {"role": "system", "content": "You are JCoder, a Python coding assistant specialized in RAG systems."},
            {"role": "user", "content": "Write a Python function that deduplicates a list of text chunks using SHA-256 hashing."},
            {"role": "assistant", "content": (
                "import hashlib\n\n"
                "def deduplicate_chunks(chunks: list[str]) -> list[str]:\n"
                "    seen: set[str] = set()\n"
                "    unique: list[str] = []\n"
                "    for chunk in chunks:\n"
                "        h = hashlib.sha256(chunk.encode('utf-8')).hexdigest()\n"
                "        if h not in seen:\n"
                "            seen.add(h)\n"
                "            unique.append(chunk)\n"
                "    return unique"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are JCoder, a Python coding assistant specialized in RAG systems."},
            {"role": "user", "content": "Write a Python function that splits text into overlapping windows of N tokens with stride S."},
            {"role": "assistant", "content": (
                "def sliding_window_chunks(text: str, window_size: int = 512, stride: int = 448) -> list[str]:\n"
                "    tokens = text.split()\n"
                "    if len(tokens) <= window_size:\n"
                "        return [text]\n"
                "    chunks = []\n"
                "    for i in range(0, len(tokens) - window_size + 1, stride):\n"
                "        chunks.append(' '.join(tokens[i:i + window_size]))\n"
                "    if i + window_size < len(tokens):\n"
                "        chunks.append(' '.join(tokens[-window_size:]))\n"
                "    return chunks"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are JCoder, a Python coding assistant specialized in RAG systems."},
            {"role": "user", "content": "Write a Python function that computes cosine similarity between two lists of floats."},
            {"role": "assistant", "content": (
                "import math\n\n"
                "def cosine_similarity(a: list[float], b: list[float]) -> float:\n"
                "    if len(a) != len(b):\n"
                "        raise ValueError(f'Vectors must be same length: {len(a)} != {len(b)}')\n"
                "    dot = sum(x * y for x, y in zip(a, b))\n"
                "    norm_a = math.sqrt(sum(x * x for x in a))\n"
                "    norm_b = math.sqrt(sum(x * x for x in b))\n"
                "    if norm_a == 0 or norm_b == 0:\n"
                "        return 0.0\n"
                "    return dot / (norm_a * norm_b)"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are JCoder, a Python coding assistant specialized in RAG systems."},
            {"role": "user", "content": "Write a Python class for a simple LRU cache with max_size, get, and put methods."},
            {"role": "assistant", "content": (
                "from collections import OrderedDict\n"
                "from typing import Any\n\n"
                "class LRUCache:\n"
                "    def __init__(self, max_size: int = 128):\n"
                "        if max_size < 1:\n"
                "            raise ValueError('max_size must be >= 1')\n"
                "        self._cache: OrderedDict[str, Any] = OrderedDict()\n"
                "        self._max_size = max_size\n\n"
                "    def get(self, key: str) -> Any | None:\n"
                "        if key in self._cache:\n"
                "            self._cache.move_to_end(key)\n"
                "            return self._cache[key]\n"
                "        return None\n\n"
                "    def put(self, key: str, value: Any) -> None:\n"
                "        if key in self._cache:\n"
                "            self._cache.move_to_end(key)\n"
                "        self._cache[key] = value\n"
                "        if len(self._cache) > self._max_size:\n"
                "            self._cache.popitem(last=False)"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are JCoder, a Python coding assistant specialized in RAG systems."},
            {"role": "user", "content": "Write a Python function that reads a YAML config file and returns a nested dict with dot-notation access."},
            {"role": "assistant", "content": (
                "import yaml\n"
                "from pathlib import Path\n\n"
                "class DotDict(dict):\n"
                "    def __getattr__(self, key):\n"
                "        try:\n"
                "            val = self[key]\n"
                "            return DotDict(val) if isinstance(val, dict) else val\n"
                "        except KeyError:\n"
                "            raise AttributeError(f'No key: {key}')\n\n"
                "def load_config(path: str | Path) -> DotDict:\n"
                "    p = Path(path)\n"
                "    if not p.exists():\n"
                "        raise FileNotFoundError(f'Config not found: {p}')\n"
                "    with open(p, 'r', encoding='utf-8') as f:\n"
                "        data = yaml.safe_load(f)\n"
                "    if not isinstance(data, dict):\n"
                "        raise ValueError('Config root must be a mapping')\n"
                "    return DotDict(data)"
            )},
        ]
    },
]


def step_banner(step_num: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'=' * 60}")


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    step_banner(1, "Verify CUDA")
    # ------------------------------------------------------------------
    import torch
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. Cannot proceed.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU 0: {gpu_name} ({gpu_mem:.1f} GB)")

    # ------------------------------------------------------------------
    step_banner(2, f"Load Qwen3.6-27B in 4-bit ({len(SMOKE_DATA)} training examples)")
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.6-27B",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    vram_after_load = torch.cuda.memory_allocated(0) / 1e9
    print(f"VRAM after model load: {vram_after_load:.1f} GB")

    # ------------------------------------------------------------------
    step_banner(3, "Configure LoRA (rank=8, minimal)")
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ------------------------------------------------------------------
    step_banner(4, "Train — 10 steps only")
    # ------------------------------------------------------------------
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=SMOKE_DATA,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=10,
            learning_rate=2e-4,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        ),
    )

    stats = trainer.train()
    vram_after_train = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Training loss: {stats.training_loss:.4f}")
    print(f"Peak VRAM during training: {vram_after_train:.1f} GB")

    # ------------------------------------------------------------------
    step_banner(5, "Export to GGUF Q4_K_M")
    # ------------------------------------------------------------------
    gguf_dir = OUTPUT_DIR / "gguf"
    print(f"Saving to {gguf_dir} ...")
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        print("FATAL: No GGUF file produced.")
        sys.exit(1)
    gguf_path = gguf_files[0]
    gguf_size = gguf_path.stat().st_size / 1e9
    print(f"GGUF file: {gguf_path.name} ({gguf_size:.1f} GB)")

    # ------------------------------------------------------------------
    step_banner(6, "Create Ollama Modelfile + import")
    # ------------------------------------------------------------------
    modelfile_content = (
        f"FROM {gguf_path.resolve()}\n"
        "PARAMETER num_ctx 2048\n"
        "PARAMETER temperature 0\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|endoftext|>"\n'
        'SYSTEM "You are JCoder, a Python coding assistant specialized in RAG systems."\n'
    )
    modelfile_path = OUTPUT_DIR / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    print("Running: ollama create jcoder-smoke-test ...")
    result = subprocess.run(
        ["ollama", "create", "jcoder-smoke-test", "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"FATAL: ollama create failed:\n{result.stderr}")
        sys.exit(1)
    print(f"Ollama: {result.stdout.strip()}")

    # ------------------------------------------------------------------
    step_banner(7, "Canary test — base vs fine-tuned")
    # ------------------------------------------------------------------
    canary = "Write a Python function to deduplicate text chunks using content hashing."
    models_to_test = ["qwen3.6:27b-q4_K_M", "jcoder-smoke-test"]

    for model_name in models_to_test:
        print(f"\n--- {model_name} ---")
        try:
            r = subprocess.run(
                ["ollama", "run", model_name, canary],
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = r.stdout.strip()[:600]
            print(output if output else "(no output)")
        except subprocess.TimeoutExpired:
            print("(timed out)")
        except FileNotFoundError:
            print("(ollama not found — skip)")

    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    step_banner(0, f"SMOKE TEST COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    # ------------------------------------------------------------------
    print(f"""
Results:
  GPU:            {gpu_name}
  VRAM after load: {vram_after_load:.1f} GB
  Peak VRAM train: {vram_after_train:.1f} GB
  Training loss:   {stats.training_loss:.4f}
  GGUF size:       {gguf_size:.1f} GB
  Ollama import:   {'OK' if result.returncode == 0 else 'FAILED'}
  Total time:      {elapsed:.0f}s

If both models produced different output, the pipeline works.
Next: run full training with real data.
""")

    report = {
        "gpu": gpu_name,
        "vram_load_gb": round(vram_after_load, 2),
        "vram_peak_gb": round(vram_after_train, 2),
        "training_loss": round(stats.training_loss, 4),
        "gguf_size_gb": round(gguf_size, 2),
        "ollama_ok": result.returncode == 0,
        "elapsed_seconds": round(elapsed, 1),
    }
    report_path = OUTPUT_DIR / "smoke_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
