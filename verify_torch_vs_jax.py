#!/usr/bin/env python3
"""Sanity check: PyTorch backbone (step.py path) vs JAX gemma2b backbone on the
SAME tokens, both fp32 on CPU. Confirms the einsum->linear weight conversion."""
import os, pathlib, sys
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(pathlib.Path(__file__).parent / "openpi" / "src"))

import numpy as np, torch, jax, jax.numpy as jnp
import flax.linen as nn
import gemma2b, gemma_torch as gt, step

T = 16
tokens_np = np.array([[3, 17, 42, 100, 7, 256, 11, 900, 5, 64, 128, 1, 333, 12, 808, 2]], dtype=np.int32)

# ---- JAX backbone (fp32) ----
cfg_j = gemma2b.get_config("gemma_2b")
mod = gemma2b.Module(configs=[cfg_j], embed_dtype="float32")
from forward_pass import restore_params, select_expert0
full = restore_params(step.WEIGHTS)
pj = select_expert0(full["PaliGemma"]["llm"], mod)
emb = mod.apply({"params": pj}, jnp.asarray(tokens_np), method=gemma2b.Module.embed)
positions = jnp.arange(T)[None]
mask = jnp.ones((1, T, T), dtype=bool)
(h_jax,), _ = mod.apply({"params": pj}, [emb], positions, mask)
h_jax = np.asarray(h_jax).astype(np.float32)

# ---- PyTorch backbone (fp32) ----
cfg = gt.GemmaConfig()
embed_table, final_norm_w, lw = step.load_backbone(cfg)
blocks = [gt.GemmaBlock(cfg, lw[i], i) for i in range(cfg.depth)]
x = embed_table[torch.from_numpy(tokens_np).long()] * (cfg.width ** 0.5)
pos = torch.arange(T).unsqueeze(0)
am = torch.zeros(1, 1, T, T)
cos, sin = gt.precompute_freqs_cis(cfg.head_dim, T, cfg.rope_base)
for b in blocks:
    x, _ = b.forward(x, cos, sin, am, pos)
x = gt.rms_norm(x, final_norm_w, cfg.rms_norm_eps)
h_pt = x.detach().numpy().astype(np.float32)

d = np.abs(h_jax - h_pt)
print(f"hidden shape jax={h_jax.shape} torch={h_pt.shape}")
print(f"max|jax-torch| = {d.max():.3e}   mean = {d.mean():.3e}")
print(f"jax  norm/elt = {np.linalg.norm(h_jax)/h_jax.size:.4e}")
print(f"rel max diff  = {d.max()/ (np.abs(h_jax).max()+1e-9):.3e}")
print("MATCH" if d.max() < 1e-2 else "MISMATCH -- check conversion")
