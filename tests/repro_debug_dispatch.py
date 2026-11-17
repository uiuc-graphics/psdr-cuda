"""Manual reproducer: native OptiX debug-mode crash on Windows/driver 610.88.

Run separately: python tests/repro_debug_dispatch.py
This deliberately reproduces a process crash, so it is not a pytest test.
"""

import drjit as dr
import psdr_jit as psdr
from drjit.cuda import Array3f, Array3i

dr.set_flag(dr.JitFlag.Debug, True)
scene = psdr.Scene()
scene.opts = psdr.RenderOption(12, 8, 4, 0, 0)
scene.opts.log_level = 0
scene.add_Sensor(psdr.PerspectiveCamera(45, .01, 100.))
for name, color, z in [("back", [.5, .5, .5], 3), ("front", [.2, .7, .9], 2)]:
    scene.add_BSDF(psdr.DiffuseBSDF(color), name)
    mesh = psdr.Mesh()
    mesh.load_raw(
        Array3f([[-2, 2, 2, -2], [-2, -2, 2, 2], [z] * 4]),
        Array3i([[0, 0], [2, 3], [1, 2]]),
    )
    scene.add_Mesh(mesh, name, None)
scene.configure()
print(psdr.CollocatedIntegrator(1.).renderC(scene).numpy())
