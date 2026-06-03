#!/usr/bin/env bash
set -euo pipefail
cd /home/mechanism/work/gemma2b
UV="$HOME/.local/bin/uv"

echo "=== [1/3] creating venv (python 3.11) ==="
"$UV" venv --python 3.11 .venv

echo "=== [2/3] installing CPU torch (only needed for a type union in array_typing) ==="
"$UV" pip install --python .venv "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu

echo "=== [3/3] installing jax[cuda12] + flax/orbax stack (openpi pins) ==="
"$UV" pip install --python .venv \
  "jax[cuda12]==0.5.3" \
  "flax==0.10.2" \
  "orbax-checkpoint==0.11.13" \
  "ml-dtypes==0.4.1" \
  "tensorstore==0.1.74" \
  "jaxtyping==0.2.36" \
  "beartype==0.19.0" \
  "einops>=0.8.0" \
  "numpy<2.0"

echo "=== verifying imports ==="
.venv/bin/python - <<'PY'
import jax, flax, orbax.checkpoint, jaxtyping, beartype, einops, numpy, torch
print("jax", jax.__version__)
print("flax", flax.__version__)
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("jax devices:", jax.devices())
PY
echo "ENV_BUILD_DONE"
