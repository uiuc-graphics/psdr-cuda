# Local GPU demo

From the repository root, with an NVIDIA GPU and a CUDA/OptiX driver:

```sh
python -m pip install ".[web]"
python -m psdr_jit.web
```

Open **http://127.0.0.1:8080**. The first render starts automatically. Change
settings and press **Render** or **Ctrl+Enter** (Cmd+Enter on macOS keyboards).
The first render of a scene or derivative can take longer while kernels compile.
Both image panels show the last completed render; “Settings changed” means the
controls have unapplied edits. The arrows save the displayed PNGs.

The demo uses the bundled Cornell-box and environment-lit bunny assets. Choose
path tracing (two bounces) or direct lighting, move the small sphere or bunny,
and adjust roughness, diffuse albedo, or illumination. Position values are
normalized: one unit moves the sphere 100 scene units or the bunny 30 units.

The right panel shows the **signed mean RGB derivative of linear radiance**,
computed with Dr.Jit forward AD. Blue is negative; orange is positive. Its
symmetric display range is the 99th percentile of absolute derivatives (falling
back to the maximum when sparse), shown numerically under the image. Values
outside this range clip for display. Each render rescales independently, so use
the labels when comparing magnitudes. Exposure affects only the left display.
Geometry derivatives include primary and secondary boundary samples, each at
the selected sample count. Low sample counts produce noisy estimates.

Rendering runs in one spawned GPU process, with one active job at a time.
Linked kernels are cleared between jobs to prevent reuse of an old OptiX
pipeline with a new scene's shader table; the on-disk compilation cache remains.
The server binds only to loopback, uses no remote services or CDN, and accepts
only bundled scenes and bounded settings. This is a local demo, not a public
multi-user render service. Stop it with Ctrl+C; an active GPU job finishes before
shutdown. To use another port or launch outside the checkout:

```sh
python -m psdr_jit.web --port 8081 --assets /path/to/psdr-jit/tutorials/data
```

The wheel includes the web UI; scene assets remain in the repository. No Node.js
build step is needed. Install `.[web,test]` and run `python -m pytest -q` to check
scene derivatives, illumination linearity, input validation, and the HTTP-to-GPU
pipeline. GPU checks require CUDA. The webpage should also be checked in a browser
after changes to its controls or layout.
