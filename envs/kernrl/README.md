# kernrl

RL environment for GPU kernel optimization. Train LLM agents to write fast CUDA/Triton kernels.

## Overview

Agents receive a PyTorch reference implementation and must write an optimized GPU kernel that:
1. Produces the same output (within tolerance)
2. Runs faster than the baseline

Each submission is evaluated with:
- Compilation checking
- Correctness verification against reference
- Benchmark timing for speedup measurement
- NSight Systems profiling (optional)
- NSight Compute profiling (optional)

## Installation

```bash
cd envs/kernrl
pip install -e .
```

Requires: NVIDIA GPU with CUDA toolkit, PyTorch, Triton

## Quick Start

```python
from openenv.envs.kernrl import kernrl_env, KernelAction

# Connect to server
env = kernrl_env(base_url="http://localhost:8000")

# Start episode
obs = env.reset(problem_id="L1_23_Softmax")
print(obs.problem_description)

# Submit a kernel
action = KernelAction(code='''
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * n_cols
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_start + col_offsets, softmax_output, mask=mask)

class Model(torch.nn.Module):
    def forward(self, x):
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        return output
''')

result = env.step(action)
print(f"Speedup: {result.observation.speedup}x")
print(f"Correct: {result.observation.correctness_pass}")
```

## Running the Server

```bash
# Development
uvicorn kernrl.server.app:app --reload --host 0.0.0.0 --port 8000

# Docker (GPU required)
docker build -t kernrl -f server/Dockerfile .
docker run --gpus all -p 8000:8000 kernrl
```

## Problem Levels

| Level | Name | Count | Description |
|-------|------|-------|-------------|
| 1 | Simple Operators | 15 | matmul, softmax, conv, norms |
| 2 | Fused Operations | 15 | matmul+activation chains |
| 3 | Single Blocks | 3 | attention, transformer block |
| 4 | Novel Layers | 8 | MLA, MoE, GQA, FP8, INT4 |
| 5 | Scientific Computing | 8 | N-body, stencil, SpMV |
| 6 | Graphics | 8 | ray tracing, histogram, blur |
| 7 | Signal Processing | 8 | FFT, convolution, median filter |
| 8 | Video Processing | 8 | motion estimation, optical flow |
| 9 | Parallel Primitives | 8 | scan, reduction, radix sort |
| 10 | Cryptography | 8 | SHA-256, AES, ChaCha20 |

**Total: 89 problems**

## Reward Structure

| Component | Reward | Description |
|-----------|--------|-------------|
| Compilation | +0.1 | Code compiles successfully |
| Correctness | +0.3 | Output matches reference |
| Beats baseline | +0.3 | Speedup > 1.0x |
| Speedup bonus | +0.3 | Scales with log2(speedup) |

## License

BSD-3-Clause (following OpenEnv licensing)
