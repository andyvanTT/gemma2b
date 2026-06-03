#!/usr/bin/env python3
"""Run a forward pass through the Gemma 1 2B (PaliGemma) backbone used by
Physical Intelligence's pi0 / pi0.5, loading the real pi05_base weights.

What this does:
  1. Pins JAX to the RTX 4090 only (GPU 0) -- the 2B params fit in 24GB and
     splitting across the two GPUs would only add a bottleneck.
  2. Imports `gemma2b` (openpi's gemma.py) and builds the `gemma_2b` variant
     as a single-expert transformer (i.e. just the PaliGemma LLM, no action
     expert -- that second expert is what pi0.5's `pi05=True` adds adarms to).
  3. Restores params from ./weights/pi05_base/params, pulls out PaliGemma/llm,
     keeps only expert-0 (the gemma_2b backbone), and runs a forward pass.

pi0 vs pi0.5: the backbone here is byte-for-byte the SAME architecture in both.
`pi05=True` only changes the action expert (expert 1) + conditioning, not this.
"""
import os
import pathlib
import sys

# --- pin to the 4090 (GPU 0) BEFORE importing jax ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# leave a little headroom rather than preallocating the whole card
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

HERE = pathlib.Path(__file__).parent.resolve()
# make both gemma2b.py (this dir) and the openpi package importable
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "openpi" / "src"))

import functools  # noqa: E402

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
from flax import traverse_util  # noqa: E402

import gemma2b  # noqa: E402  <- the file the user asked us to import

WEIGHTS = HERE / "weights" / "pi05_base" / "params"
VARIANT = "gemma_2b"
SEQ_LEN = 16
EMBED_DTYPE = "bfloat16"


def restore_params(path: pathlib.Path, dtype=None):
    """Minimal copy of openpi.models.model.restore_params (avoids importing the
    whole openpi.models.model chain, which would pull in sentencepiece etc.)."""
    path = path.resolve()
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(path)
        item = {"params": metadata["params"]}
        restored = ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=jax.Array, dtype=dtype
                    ),
                    item,
                ),
            ),
        )["params"]
    # checkpoints saved via nnx training end every keypath with "value"; strip it
    flat = traverse_util.flatten_dict(restored)
    if flat and all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
        restored = traverse_util.unflatten_dict(flat)
    return restored


def select_expert0(llm_params, module):
    """Keep only the params that the single-expert gemma_2b Module expects.

    The loaded PaliGemma/llm subtree contains BOTH expert 0 (gemma_2b backbone,
    un-suffixed names) and expert 1 (the action expert, '_1'-suffixed). We get
    the abstract param tree of a single-expert Module and intersect by keypath.
    """
    # NB: gemma2b.Module defines its own `init(use_adarms)` convenience method, which
    # SHADOWS flax's nn.Module.init. So we call the base linen init explicitly and point
    # it at that method via method="init" (exactly how openpi does it through nnx_bridge).
    abstract = jax.eval_shape(
        functools.partial(nn.Module.init, module, jax.random.key(0), [False], method="init")
    )["params"]
    want = traverse_util.flatten_dict(abstract)
    have = traverse_util.flatten_dict(llm_params)
    selected = {}
    missing = []
    for kp, spec in want.items():
        if kp not in have:
            missing.append(kp)
            continue
        arr = have[kp]
        if tuple(arr.shape) != tuple(spec.shape):
            raise ValueError(f"shape mismatch at {kp}: ckpt {arr.shape} vs model {spec.shape}")
        selected[kp] = arr
    if missing:
        raise ValueError(f"{len(missing)} expert-0 params not found in checkpoint, e.g. {missing[:3]}")
    return traverse_util.unflatten_dict(selected)


def main():
    print("JAX devices (should be the 4090 only):", jax.devices(), flush=True)
    if not WEIGHTS.exists():
        sys.exit(f"weights not found at {WEIGHTS} -- run download_weights.py first")

    # 1. build the single-expert gemma_2b backbone
    cfg = gemma2b.get_config(VARIANT)
    print(f"\n{VARIANT} config: width={cfg.width} depth={cfg.depth} "
          f"mlp_dim={cfg.mlp_dim} heads={cfg.num_heads} kv_heads={cfg.num_kv_heads} "
          f"head_dim={cfg.head_dim}", flush=True)
    module = gemma2b.Module(configs=[cfg], embed_dtype=EMBED_DTYPE)

    # 2. load real pi05_base weights and isolate the gemma_2b backbone
    print(f"\nrestoring params from {WEIGHTS} ...", flush=True)
    full = restore_params(WEIGHTS)
    print("top-level checkpoint keys:", list(full.keys()), flush=True)
    llm = full["PaliGemma"]["llm"]
    params = select_expert0(llm, module)
    n = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
    print(f"loaded gemma_2b backbone: {n:,} params "
          f"({n/1e9:.2f}B), dtype={jax.tree.leaves(params)[0].dtype}", flush=True)

    # 3. forward pass on dummy tokens
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, 1000, size=(1, SEQ_LEN)), dtype=jnp.int32)
    positions = jnp.arange(SEQ_LEN, dtype=jnp.int32)[None, :]
    # PaliGemma attends over its prefix bidirectionally -> full all-ones mask.
    # (For autoregressive decoding you'd pass a causal mask instead.)
    mask = jnp.ones((1, SEQ_LEN, SEQ_LEN), dtype=bool)

    variables = {"params": params}
    embedded = module.apply(variables, tokens, method=gemma2b.Module.embed)
    print(f"\nembedded tokens -> {embedded.shape} {embedded.dtype}", flush=True)

    (hidden,), _kv = jax.jit(module.apply)(variables, [embedded], positions, mask)
    hidden.block_until_ready()
    print(f"backbone hidden states -> {hidden.shape} {hidden.dtype}", flush=True)

    # project back to vocab with the tied embedding table to show next-token logits
    embed_table = params["embedder"]["input_embedding"]
    logits = jnp.dot(hidden.astype(jnp.float32), embed_table.astype(jnp.float32).T)
    print(f"logits -> {logits.shape}  (vocab={gemma2b.PALIGEMMA_VOCAB_SIZE})", flush=True)
    print("argmax token id at each position:", np.asarray(jnp.argmax(logits[0], axis=-1)), flush=True)
    print("\nForward pass OK.", flush=True)


if __name__ == "__main__":
    main()
