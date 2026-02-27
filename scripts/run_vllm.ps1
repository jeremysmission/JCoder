# JCoder vLLM Server Launcher
# Starts three vLLM processes: LLM (TP=2), Embedder (GPU1), Reranker (GPU1)

$ModelDir = if ($env:JCODER_MODEL_DIR) { $env:JCODER_MODEL_DIR } else { Join-Path $PSScriptRoot '..\data\models' }

# LLM -- tensor parallel across both 3090s
Start-Process -NoNewWindow python -ArgumentList @(
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", "$ModelDir\Qwen3-Coder-Next-80B",
    "--tensor-parallel-size", "2",
    "--port", "8000",
    "--gpu-memory-utilization", "0.80",
    "--max-model-len", "32768"
)

# Embedder -- single GPU
Start-Process -NoNewWindow python -ArgumentList @(
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", "$ModelDir\nomic-embed-code-v1",
    "--port", "8001",
    "--gpu-memory-utilization", "0.08"
)

# Reranker -- single GPU
Start-Process -NoNewWindow python -ArgumentList @(
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", "$ModelDir\Qwen3-Reranker-4B",
    "--port", "8002",
    "--gpu-memory-utilization", "0.04"
)

Write-Host "[OK] vLLM servers starting on ports 8000, 8001, 8002"
Write-Host "     Total GPU utilization: 0.92 (0.08 headroom for OS/CUDA)"
