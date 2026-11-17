"""Regression probe for the former OptiX debug compilation crash.

Run: python tests/repro_debug_dispatch.py [--no-debug] [--output results.npz]
Each invocation starts a fresh process and creates its OptiX context in the
requested mode, including context validation when debug mode is enabled.
"""

import argparse

import drjit as dr
import numpy as np
import psdr_jit as psdr
from drjit.cuda import Array3f, Array3i
from drjit.cuda.ad import Float

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-debug", dest="debug", action="store_false")
parser.add_argument("--output")
args = parser.parse_args()
dr.set_flag(dr.JitFlag.Debug, args.debug)


def quad(x0, x1, z):
    mesh = psdr.Mesh()
    mesh.load_raw(
        Array3f([[x0, x1, x1, x0], [-2, -2, 2, 2], [z] * 4]),
        Array3i([[0, 0], [2, 3], [1, 2]]),
    )
    return mesh


scene = psdr.Scene()
scene.opts = psdr.RenderOption(12, 8, 4, 0, 0)
scene.opts.log_level = 0
scene.add_Sensor(psdr.PerspectiveCamera(45, .01, 100.))
scene.add_BSDF(psdr.DiffuseBSDF([.5, .5, .5]), "back")
scene.add_BSDF(psdr.DiffuseBSDF([.2, .7, .9]), "front")
scene.add_Mesh(quad(-2, 2, 3), "back", None)
scene.add_Mesh(quad(-1, 0, 2), "front", None)
scene.configure()

power = Float(1)
dr.enable_grad(power)
renderer = psdr.CollocatedIntegrator(power)
results = {"collocated": renderer.renderC(scene, seed=31).numpy()}
image = renderer.renderD(scene, seed=31)
dr.forward(power)
results["collocated_ad"] = image.numpy()
results["collocated_grad"] = dr.grad(image).numpy()
np.testing.assert_allclose(results["collocated_grad"], results["collocated"], rtol=2e-5)

# Exercise direct lighting, BSDF callables, AD, and scene reconfiguration too.
light = quad(-2, -.8, 1)
light.face_indices = dr.cuda.ad.Array3i([[0, 0], [1, 2], [2, 3]])
scene.add_Mesh(light, "back", psdr.AreaLight([10., 10., 10.]))
scene.configure()
for name, renderer in [("path", psdr.PathTracer(2)), ("direct", psdr.Direct(2))]:
    results[name] = renderer.renderC(scene, seed=31).numpy()
    results[name + "_ad"] = renderer.renderD(scene, seed=31).numpy()
    np.testing.assert_allclose(results[name + "_ad"], results[name], rtol=1e-3, atol=2e-5)

for image in results.values():
    assert image.shape == (3, 96)
    assert np.isfinite(image).all() and image.mean() > 0
assert dr.flag(dr.JitFlag.Debug) == args.debug
assert dr.flag(dr.JitFlag.SymbolicCalls)
if args.output:
    np.savez(args.output, **results)
print(f"OptiX rendering and gradients passed (debug={args.debug}).")
