# Gemma 1 2B backbone (PaliGemma) — π0 / π0.5

Loads the **same Gemma 1 2B backbone weights Physical Intelligence uses** and runs a
forward pass through it. The Gemma code (`gemma2b.py`) is openpi's `gemma.py`.

## π0 vs π0.5 — same backbone

Both π0 and π0.5 use the **identical** Gemma 1 2B (`gemma_2b`) PaliGemma backbone as
expert 0. In openpi, π0.5 is literally `Pi0Config(pi05=True)` — same model class,
same `gemma.Module`. `pi05=True` only changes the **action expert** (expert 1) and its
conditioning (adaptive RMSNorm driven by a time-MLP, dropped discrete `state_proj`).
The expert-0 PaliGemma LLM forward is the same architecture in both, so this backbone +
script serves π0.5. We load the trained weights from `pi05_base` since that's the π0.5
checkpoint.

## Layout

    gemma2b.py            openpi's gemma.py (Gemma adaptation for Pi)
    forward_pass.py       builds gemma_2b, loads real weights, runs a forward pass
    download_weights.py   mirrors the public pi05_base checkpoint into ./weights
    build_env.sh          builds the .venv
    .venv/                Python 3.11 env (jax[cuda12]==0.5.3, flax, orbax, ...)
    openpi/               cloned openpi (provides openpi.models.lora etc.)
    weights/pi05_base/params/   the orbax checkpoint (~12.4 GB)

## Setup

    bash build_env.sh            # build the venv (jax GPU + flax/orbax stack)
    python3 download_weights.py  # download pi05_base params (~12.4 GB) into ./weights

## Run a forward pass

    .venv/bin/python forward_pass.py

Pinned to the **RTX 4090 (GPU 0)** via `CUDA_VISIBLE_DEVICES=0` — the 2B params fit in
24 GB, and splitting across both GPUs would only add a bottleneck.

The script builds the single-expert `gemma_2b` Module (just the PaliGemma LLM, no action
expert), restores params, isolates `PaliGemma/llm` expert 0, runs a forward pass on dummy
tokens, and prints hidden-state and logit shapes.

## Weight source

`gs://openpi-assets/checkpoints/pi05_base/params` — public, no auth. The Gemma backbone
lives at `params["PaliGemma"]["llm"]` (expert 0 = un-suffixed names; the `_1`-suffixed
names are the action expert, which we drop here).
