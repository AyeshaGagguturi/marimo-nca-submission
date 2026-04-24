# The Language of Life — NCA for marimo × alphaXiv

A marimo notebook submission based on *Growing Neural Cellular Automata*
(Mordvintsev et al., 2020).

## Files

- **`train.py`** — trains NCA on emoji targets. Run on Nebius GPU.
- **`notebook.py`** — the marimo notebook. Your actual submission.
- **`STORYBOARD.md`** — narrative plan, timing, visual decisions.

## Pipeline

### 1. Train on Nebius

```bash
# On your Nebius H100 (or any CUDA GPU):
pip install torch torchvision pillow requests numpy
python train.py --target all --steps 8000
```

Produces `weights/{target}.pt` and `weights/{target}_target.npy` for each
of: `lizard`, `heart`, `seedling`, `fish`, `star`.

Total training time: ~40 minutes on H100, ~2 hours on T4.

If you're compute-rich, bump `--steps 15000` for cleaner convergence.

### 2. Test locally

```bash
pip install marimo torch pillow matplotlib numpy
marimo edit notebook.py
```

Open `http://localhost:2718`. The notebook will auto-load from `./weights/`.

### 3. Upload to molab

1. Go to https://molab.marimo.io
2. Create a new notebook, paste in contents of `notebook.py`
3. Upload the `weights/` directory as a persistent asset
4. Verify each cell renders without error
5. Make the notebook public
6. Submit the molab URL via the competition form

## Key implementation notes

- The NCA class is defined in **both** `train.py` and `notebook.py` (the
  notebook must be self-contained in molab — no local imports).
  If you modify the architecture, update both.
- Inference is CPU-only in molab, but 40×40×16 is trivial on CPU —
  expect ~30ms per 100 steps.
- GIFs are embedded as base64 data URLs. Keep frame count × resolution low
  enough that GIFs stay under 2MB each.
- The `mo.ui.run_button` pattern is used for expensive cells so they only
  run on explicit click — keeps the notebook responsive.

## Competition deadline

April 26, 11:59 PM PST. Submit at:
https://form.jotform.com/260916218322049

## The original contribution

The notebook's Chapter V includes a side-by-side visualization of the RGB
view and a hidden latent channel during the healing process — showing
information waves propagating from the damage site. This is not present in
the original paper or any public implementation.

If the judges ask, "what's novel here?" — that's the answer.
