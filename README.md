[![PyPI version](https://badge.fury.io/py/psdr-jit.svg?kill_cache=1)](https://badge.fury.io/py/psdr-jit)
[![DOI](https://zenodo.org/badge/530067467.svg)](https://zenodo.org/doi/10.5281/zenodo.13777324)
# psdr-jit
Path-space differentiable renderer with [`drjit`](https://drjit.readthedocs.io/en/latest/) as the numerical backend.
Old repo: https://github.com/uci-rendering/psdr-cuda


## Installation

This source version uses **Dr.Jit 1.5.0** and **nanobind 2.13.0**. The nanobind
version and `drjit` binding domain must match the ABI used by the Dr.Jit wheels.
Build and runtime Dr.Jit versions are pinned together in `pyproject.toml`.

Requirements:

- Python 3.10 or newer, with a Dr.Jit wheel available for your platform.
- A C++17 compiler (MSVC on Windows or GCC/Clang on Linux) and CMake 3.19+.
- An NVIDIA GPU and driver supported by Dr.Jit's CUDA/OptiX backend to render.

The build links against the installed Dr.Jit package. It does not compile a
second Dr.Jit runtime or require a CUDA toolkit/submodule checkout. The unused
standalone CUDA helper kernels are not part of the Python extension.

```bash
git clone https://github.com/AI-yank/psdr-jit.git
cd psdr-jit
python -m pip install .
```

For development:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install scikit-build-core nanobind==2.13.0 drjit==1.5.0
python -m pip install --no-build-isolation -e ".[test]"
python -m pytest tests -q
```

The tests exercise GPU rendering, typed Python bindings, forward/reverse AD,
finite differences, batch rendering, and scene ownership/reconfiguration.
They skip when a CUDA backend is unavailable. The current build was validated
on Windows with Python 3.12 and an RTX 4090; Linux build/runtime validation is
still needed.

Debug rendering is covered by a subprocess regression that compares images
and gradients against normal rendering. The renderer omits OptiX's additional
`OPTIX_EXCEPTION_FLAG_DEBUG` instrumentation, which crashes during kernel
compilation on Windows driver 610.88. Stack/depth exception checks and Dr.Jit's
`JitFlag.Debug` checks remain enabled. This matches the exception flags used in debug mode by
[Dr.Jit 1.5's default pipeline](https://github.com/mitsuba-renderer/drjit-core/blob/3b96411288b9938889045308654ba37d4ef24b71/src/optix_core.cpp).
The standalone probe is `python tests/repro_debug_dispatch.py`.

The migration retains the existing Python entry points. Default secondary-edge
samples are now zero. Configured meshes added to a scene own independent caches, objects
are registered when configured and unregistered on destruction, and the
OptiX binding table is rebuilt when the mesh count changes. Seeded batch
rendering uses the same sample streams as the corresponding full-image pixels; edge gradients
are accumulated at full resolution before gathering batch pixels. Collocated
ray misses are masked after division to preserve silhouette gradients.
Intersection mesh pointers retain their scene owner, and environment-only
lighting has a nonzero emitter sampling weight.

## Getting Started
```python
import drjit
import psdr_jit
```

### Example of Diff rendering

```python
import cv2

import psdr_jit as psdr
import drjit
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD
from drjit.cuda import Float as FloatC, Matrix4f as Matrix4fC

sc = psdr.Scene()
sc.opts.spp = 32 # Interior Term
sc.opts.sppe = 32 # Primary Edge
sc.opts.sppse = 32 # Secondary Edge
sc.opts.height = 512
sc.opts.width = 512

integrator = psdr.PathTracer(3)	


sensor = psdr.PerspectiveCamera(60, 0.000001, 10000000.)
to_world = Matrix4fD([[1.,0.,0.,208.],
                     [0.,1.,0.,273.],
                     [0.,0.,1.,-800.],
                     [0.,0.,0.,1.],])
sensor.to_world = to_world
sc.add_Sensor(sensor)

sc.add_BSDF(psdr.DiffuseBSDF([0.0, 0.0, 0.0]), "light")
sc.add_BSDF(psdr.DiffuseBSDF(), "cat")
sc.add_BSDF(psdr.DiffuseBSDF([0.95, 0.95, 0.95]), "white")
sc.add_BSDF(psdr.DiffuseBSDF([0.20, 0.90, 0.20]), "green")
sc.add_BSDF(psdr.DiffuseBSDF([0.90, 0.20, 0.20]), "red")

sc.add_Mesh("./tutorials/data/cbox/cbox_luminaire.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,-0.5],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "light", psdr.AreaLight([20.0, 20.0, 8.0]))
sc.add_Mesh("./tutorials/data/cbox/cbox_smallbox.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "cat", None)
sc.add_Mesh("./tutorials/data/cbox/cbox_largebox.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "cat", None)
sc.add_Mesh("./tutorials/data/cbox/cbox_floor.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
sc.add_Mesh("./tutorials/data/cbox/cbox_ceiling.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
sc.add_Mesh("./tutorials/data/cbox/cbox_back.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
sc.add_Mesh("./tutorials/data/cbox/cbox_greenwall.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "green", None)
sc.add_Mesh("./tutorials/data/cbox/cbox_redwall.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "red", None)

P = FloatD(0.)
drjit.enable_grad(P)

sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,P*100.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],]))


sc.configure()
sc.configure([0])

img = integrator.renderD(sc, 0)
org_img = img.numpy().reshape((3, sc.opts.height, sc.opts.width)).transpose(1, 2, 0)
output = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("psdr_jit_forward.exr", output)


drjit.set_grad(P, 1.0)
drjit.forward_to(img)
diff_img = drjit.grad(img)
diff_img = diff_img.numpy().reshape((3, sc.opts.height, sc.opts.width)).transpose(1, 2, 0)
output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("psdr_jit_diff_debug.exr", output)
```

### More tutorials

For a compact browser interface with scene controls and side-by-side renders and
gradients, see the [local GPU web demo](tutorials/WEB_DEMO.md):

```sh
python -m pip install ".[web]"
python -m psdr_jit.web
```

See the [tutorial setup and execution guide](tutorials/README.md) for the five
notebooks, optional dependencies, and commands to rerun them before publishing.

Object level optimization example: [`Neural-PBIR`](https://neural-pbir.github.io/) (ICCV 2023)

Scene level optimization example: [`psdr-room`](https://github.com/andyyankai/psdr-room) (SIGGRAPH ASIA 2023)

SDF+Nerf optimization example: to be released

### Project based on psdr-jit
TBA

AI Free checkpoint: 476ba6ef39e24474e85953ba3743bb0b967c797e
