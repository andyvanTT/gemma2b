#!/usr/bin/env python3
"""
step.py -- init + run a forward pass through ONLY the Gemma 1 2B backbone
(PyTorch reference impl in gemma_torch.py), printing the tensor shape at every
step in a formatted table.

- Uses the PyTorch reference (gemma_torch.GemmaBlock/Attention/MLP), NOT the JAX
  model. The JAX checkpoint is only read to get the trained weights, which are
  converted into the PyTorch (q/k/v/o, gate/up/down, layernorm) layout.
- Loads the real Gemma 2B backbone weights from pi05_base (PaliGemma/llm, expert 0).
- Pure PyTorch CPU reference -- the intended starting point for a tt_metal/CUDA port.
"""
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # weights are read via jax/orbax on CPU

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import gemma_torch as gt  # noqa: E402

HERE = pathlib.Path(__file__).parent.resolve()
WEIGHTS = HERE / "weights" / "pi05_base" / "params"
#SEQ_LEN = 16
#SEQ_LEN = 32
SEQ_LEN = 64
DTYPE = torch.float32  # faithful fp32 reference

# ----------------------------------------------------------------------------
# pretty shape logging
# ----------------------------------------------------------------------------
_W = 40


def header(title):
    print(f"\n\033[1m{title}\033[0m\n" + "-" * (_W + 28))


def log(label, t, indent=0):
    pad = "  " * indent
    shape = "x".join(str(s) for s in t.shape)
    print(f"{pad}{label:<{_W - len(pad)}} ({shape})  {str(t.dtype).replace('torch.','')}")


# ----------------------------------------------------------------------------
# weight loading + JAX(einsum) -> PyTorch(linear) conversion
# ----------------------------------------------------------------------------
def _t(np_arr):
    return torch.from_numpy(np.ascontiguousarray(np_arr)).to(DTYPE)


def load_backbone(cfg: gt.GemmaConfig):
    import jax
    import orbax.checkpoint as ocp
    from flax import traverse_util

    with ocp.PyTreeCheckpointer() as c:
        meta = c.metadata(WEIGHTS)
        item = {"params": meta["params"]}
        params = c.restore(
            WEIGHTS,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item),
            ),
        )["params"]

    flat = traverse_util.flatten_dict(params)
    if flat and all(k[-1] == "value" for k in flat):
        flat = {k[:-1]: v for k, v in flat.items()}
    L = lambda *suffix: flat[("PaliGemma", "llm", *suffix)]  # noqa: E731

    N, D, H, hid = cfg.num_heads, cfg.width, cfg.head_dim, cfg.mlp_dim
    K = cfg.num_kv_heads

    q_w = L("layers", "attn", "q_einsum", "w")          # (depth, N, D, H)
    kv_w = L("layers", "attn", "kv_einsum", "w")        # (depth, 2, K, D, H)
    o_w = L("layers", "attn", "attn_vec_einsum", "w")   # (depth, N, H, D)
    gate_w = L("layers", "mlp", "gating_einsum")        # (depth, 2, D, hid)
    down_w = L("layers", "mlp", "linear")               # (depth, hid, D)
    pre_attn = L("layers", "pre_attention_norm", "scale")  # (depth, D)
    pre_ffw = L("layers", "pre_ffw_norm", "scale")         # (depth, D)

    layers = []
    for i in range(cfg.depth):
        w = {
            # (N,D,H) -> (N,H,D) -> (N*H, D)
            "self_attn.q_proj.weight": _t(q_w[i].transpose(0, 2, 1).reshape(N * H, D)),
            # (K,D,H) -> (K,H,D) -> (K*H, D)
            "self_attn.k_proj.weight": _t(kv_w[i, 0].transpose(0, 2, 1).reshape(K * H, D)),
            "self_attn.v_proj.weight": _t(kv_w[i, 1].transpose(0, 2, 1).reshape(K * H, D)),
            # (N,H,D) -> (D,N,H) -> (D, N*H)
            "self_attn.o_proj.weight": _t(o_w[i].transpose(2, 0, 1).reshape(D, N * H)),
            # (D,hid) -> (hid,D)
            "mlp.gate_proj.weight": _t(gate_w[i, 0].T),
            "mlp.up_proj.weight": _t(gate_w[i, 1].T),
            # (hid,D) -> (D,hid)
            "mlp.down_proj.weight": _t(down_w[i].T),
            "input_layernorm.weight": _t(pre_attn[i]),
            "post_attention_layernorm.weight": _t(pre_ffw[i]),
        }
        layers.append(w)

    embed = _t(L("embedder", "input_embedding"))   # (vocab, D)
    final_norm = _t(L("final_norm", "scale"))      # (D,)
    return embed, final_norm, layers


# ----------------------------------------------------------------------------
# memory analytics: how much VRAM each block (and the model) needs per dtype
# ----------------------------------------------------------------------------
# (label, bytes-per-element). fp16 and bf16 are the SAME size -- they differ in
# precision (bf16 = more exponent, less mantissa), not in footprint.
_DTYPES = [("fp32", 4), ("fp16/bf16", 2), ("fp8/int8", 1)]


def _fmt(nbytes):
    for unit, div in (("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if nbytes >= div:
            return f"{nbytes / div:7.2f} {unit}"
    return f"{nbytes:7.0f}  B"


def _row(label, n_elem, extra=""):
    cells = "  ".join(_fmt(n_elem * b) for _, b in _DTYPES)
    print(f"{label:<26}{n_elem:>16,}  {cells}  {extra}")


def memory_report(cfg, layer_weights, embed_table, final_norm_w, seq_len, batch=1):
    w0 = layer_weights[0]
    attn_p = sum(w0[k].numel() for k in w0 if k.startswith("self_attn"))
    mlp_p = sum(w0[k].numel() for k in w0 if k.startswith("mlp"))
    norm_p = sum(w0[k].numel() for k in w0 if "layernorm" in k)
    block_p = attn_p + mlp_p + norm_p
    embed_p = embed_table.numel()
    total_p = block_p * cfg.depth + embed_p + final_norm_w.numel()

    header("WEIGHTS (static) -- per dtype footprint")
    print(f"{'component':<26}{'#elements':>16}  " +
          "  ".join(f"{n:>10}" for n, _ in _DTYPES) + "   note")
    _row("  attn (q,k,v,o)", attn_p)
    _row("  mlp (gate,up,down)", mlp_p)
    _row("  norms (2x)", norm_p)
    _row("ONE block", block_p, "<- per-block weights")
    _row(f"all blocks (x{cfg.depth})", block_p * cfg.depth)
    _row("embedding table", embed_p, "(tied: also the LM head)")
    _row("TOTAL model", total_p, "<- full backbone")

    # ---- activations (dynamic) for a single block at this batch/seq ----
    B, T, D, hid = batch, seq_len, cfg.width, cfg.mlp_dim
    N, K, Hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    acts = {
        "residual stream x": B * T * D,
        "rmsnorm out": B * T * D,
        "q (B,N,T,Hd)": B * N * T * Hd,
        "k,v (B,K,T,Hd)": 2 * B * K * T * Hd,
        "attn scores+probs": 2 * B * N * T * T,
        "attn ctx + out": 2 * B * T * D,
        "mlp gate+up+geglu": 3 * B * T * hid,
        "mlp out": B * T * D,
    }
    act_total = sum(acts.values())
    header(f"ACTIVATIONS (dynamic) -- one block @ batch={B}, seq_len={T}")
    print(f"{'tensor':<26}{'#elements':>16}  " +
          "  ".join(f"{n:>10}" for n, _ in _DTYPES))
    for name, ne in acts.items():
        _row(f"  {name}", ne)
    _row("block working set", act_total, "<- transient per block")

    # ---- what actually sits in VRAM during a forward ----
    header("PEAK VRAM ESTIMATE for a forward pass")
    resid = B * T * D                         # the only thing kept between blocks
    logits = B * T * cfg.vocab_size           # final LM head output
    print("peak ~= all weights + residual stream + one block's working set "
          "(+ logits at the end)")
    for name, nbytes in [("fp32", 4), ("fp16/bf16", 2), ("fp8/int8", 1)]:
        wt = total_p * nbytes
        peak = wt + (resid + act_total) * nbytes + logits * 4  # logits usually fp32
        ratio = (act_total / block_p) if block_p else 0
        print(f"  {name:<10} weights {_fmt(wt)} | +acts {_fmt((resid+act_total)*nbytes)} "
              f"| +logits {_fmt(logits*4)} | PEAK ~ {_fmt(peak)}")
    print(f"\nactivation 'overhead' per block = working-set / block-weights = "
          f"{act_total}/{block_p} = {act_total/block_p:.4f}x  "
          f"(grows with batch x seq; weights are fixed)")


# ----------------------------------------------------------------------------
# forward pass with per-step shape logging
# ----------------------------------------------------------------------------
def main():
    cfg = gt.GemmaConfig()
    print(f"Gemma 1 2B backbone | width={cfg.width} depth={cfg.depth} mlp_dim={cfg.mlp_dim} "
          f"heads={cfg.num_heads} kv_heads={cfg.num_kv_heads} head_dim={cfg.head_dim} "
          f"vocab={cfg.vocab_size}")
    print(f"device=cpu dtype={str(DTYPE).replace('torch.','')} seq_len={SEQ_LEN}")

    header("INIT: load + convert weights (JAX einsum -> PyTorch linear)")
    embed_table, final_norm_w, layer_weights = load_backbone(cfg)
    blocks = [gt.GemmaBlock(cfg, layer_weights[i], i) for i in range(cfg.depth)]
    n_params = embed_table.numel() + final_norm_w.numel() + sum(
        t.numel() for w in layer_weights for t in w.values())
    print(f"built {len(blocks)} GemmaBlocks | {n_params:,} params ({n_params/1e9:.2f}B)")
    log("embedder.input_embedding", embed_table)
    log("final_norm.weight", final_norm_w)
    log("per-layer q_proj.weight", layer_weights[0]["self_attn.q_proj.weight"])
    log("per-layer k_proj.weight", layer_weights[0]["self_attn.k_proj.weight"])
    log("per-layer o_proj.weight", layer_weights[0]["self_attn.o_proj.weight"])
    log("per-layer gate_proj.weight", layer_weights[0]["mlp.gate_proj.weight"])
    log("per-layer down_proj.weight", layer_weights[0]["mlp.down_proj.weight"])

    # ---- inputs ----
    header("INPUTS")
    torch.manual_seed(0)
    tokens = torch.randint(0, 1000, (1, SEQ_LEN))
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    # full (bidirectional, PaliGemma-prefix-style) attention -> additive zeros.
    # For autoregressive decoding use a causal mask of -inf above the diagonal.
    attn_mask = torch.zeros(1, 1, SEQ_LEN, SEQ_LEN, dtype=DTYPE)
    cos, sin = gt.precompute_freqs_cis(cfg.head_dim, SEQ_LEN, cfg.rope_base, DTYPE)
    log("token ids", tokens)
    log("position_ids", position_ids)
    log("attention_mask (additive)", attn_mask)
    log("rope cos / sin", cos)

    # ---- embedding ----
    header("EMBEDDING")
    x = embed_table[tokens] * (cfg.width ** 0.5)   # Gemma scales embeddings by sqrt(width)
    log("embed lookup x sqrt(width)", x)

    # ---- block 0 (detailed sub-step shapes) ----
    header("BLOCK 00  (detailed sub-steps)")
    b0 = blocks[0]
    eps = cfg.rms_norm_eps
    n1 = gt.rms_norm(x, b0.input_layernorm_weight, eps)
    log("input_layernorm (RMSNorm)", n1, 1)
    q = F.linear(n1, b0.attention.q_proj).view(1, SEQ_LEN, cfg.num_heads, cfg.head_dim).transpose(1, 2)
    k = F.linear(n1, b0.attention.k_proj).view(1, SEQ_LEN, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
    v = F.linear(n1, b0.attention.v_proj).view(1, SEQ_LEN, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
    log("q_proj  (B,Nh,T,Dh)", q, 1)
    log("k_proj  (B,Nkv,T,Dh)", k, 1)
    log("v_proj  (B,Nkv,T,Dh)", v, 1)
    qr, kr = gt.apply_rotary_emb(q, k, cos, sin, position_ids)
    log("q,k after RoPE", qr, 1)
    k_exp = kr.expand(1, cfg.num_heads, SEQ_LEN, cfg.head_dim)
    scores = torch.matmul(qr, k_exp.transpose(-2, -1)) * b0.attention.scale
    log("attn scores (B,Nh,T,T)", scores, 1)
    attn_out, _ = b0.attention.forward(n1, cos, sin, attn_mask, position_ids)
    log("attn output (after o_proj)", attn_out, 1)
    x = x + attn_out
    log("+ residual", x, 1)
    n2 = gt.rms_norm(x, b0.post_attention_layernorm_weight, eps)
    log("post_attention_layernorm", n2, 1)
    gate = F.linear(n2, b0.mlp.gate_proj)
    log("gate_proj  (B,T,mlp_dim)", gate, 1)
    mlp_out = b0.mlp.forward(n2)
    log("mlp output (after down_proj)", mlp_out, 1)
    x = x + mlp_out
    log("+ residual  -> block 00 out", x, 1)

    # ---- blocks 1..depth-1 ----
    header(f"BLOCKS 01..{cfg.depth - 1:02d}  (output per block)")
    for i in range(1, cfg.depth):
        x, _ = blocks[i].forward(x, cos, sin, attn_mask, position_ids)
        log(f"block {i:02d} out", x)

    # ---- final norm + logits ----
    header("HEAD")
    x = gt.rms_norm(x, final_norm_w, eps)
    log("final_norm", x)
    logits = torch.matmul(x, embed_table.T)   # tied embedding -> vocab logits
    log("logits (B,T,vocab)", logits)
    print("\nargmax token id per position:",
          torch.argmax(logits[0], dim=-1).tolist())
    print("\n\033[1mForward pass OK (Gemma 1 2B, PyTorch reference).\033[0m")

    # ---- memory analytics ----
    memory_report(cfg, layer_weights, embed_table, final_norm_w, SEQ_LEN, batch=1)


if __name__ == "__main__":
    main()
