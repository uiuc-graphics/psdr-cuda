# GPU tutorials

These notebooks use Dr.Jit 1.5.0 and the nanobind-based psdr-jit extension.
Rendering requires an NVIDIA GPU and a compatible CUDA/OptiX driver.

From the repository root, activate your Python environment and install:

```sh
python -m pip install ".[tutorial,test]"
python -m jupyterlab tutorials
```

Select the kernel from that environment, restart it, and run every cell in
order. Each notebook enables Dr.Jit's debug checks and uses only the bundled
scene assets. Paths resolve relative to `image_util.py`, not a hardcoded
checkout location. Torch and OpenCV are not required.

| Notebook | What it demonstrates |
| --- | --- |
| `different_integrator.ipynb` | Path, direct, collocated, silhouette, and depth rendering |
| `Forward_AD.ipynb` | Geometry derivatives with interior and boundary terms |
| `Forward_AD_envmap.ipynb` | Separate derivative terms under environment lighting |
| `batch_render.ipynb` | Deterministic pixel batches checked against a full render |
| `secondary_edge_guiding.ipynb` | Secondary-edge gradients before/after importance sampling |

The defaults use 128-pixel images (160×120 for batching) and modest sample
counts so the examples can be rerun interactively. Increase resolution and
sample counts for smoother Monte Carlo estimates. Each example checks finite
outputs; derivative examples also check for a nonzero gradient. The guiding
example compares two noisy estimates, not identical per-pixel samples.

Dr.Jit RGB arrays have shape `(3, height * width)`. Use
`to_image(pixels, height, width)` before plotting: directly reshaping them to
`(height, width, 3)` mixes the channels. Apply sRGB conversion to primal images
for display, not to signed derivatives.

To rerun every notebook in a fresh kernel using the current Python executable:

```sh
python tutorials/run_notebooks.py
```

Executed notebooks and plot PNGs are saved under `build/executed-tutorials`.
Execution stops at the first failing cell. Before publishing notebook changes,
refresh their checked-in outputs with:

```sh
python -m pytest -q
python tutorials/run_notebooks.py --in-place
```

You can pass one or more notebook filenames to rerun a particular example.
