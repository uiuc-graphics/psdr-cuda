"""Scene construction and real forward-mode derivatives (GPU worker only)."""

import base64
import io
import math
import time
from pathlib import Path

DEFAULTS = dict(scene="cornell", integrator="path", derivative="x", x=0., y=0.,
                roughness=.3, albedo=.7, light=1., resolution=128, spp=8, seed=42,
                exposure=0.)
ENUMS = dict(scene=("cornell", "bunny"), integrator=("path", "direct"),
             derivative=("x", "y", "roughness", "albedo", "light"),
             resolution=(64, 128, 256), spp=(4, 8, 16, 32))
BOUNDS = dict(x=(-1, 1), y=(-1, 1), roughness=(.08, 1), albedo=(.05, 1),
              light=(.1, 3), seed=(0, 65535), exposure=(-3, 3))


def translate(matrix_type, xyz):
    x, y, z = xyz
    return matrix_type([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def validate(data):
    if not isinstance(data, dict) or data.keys() - DEFAULTS.keys():
        raise ValueError("Unknown render settings")
    settings = DEFAULTS | data
    for key, value in settings.items():
        if key in ENUMS:
            if isinstance(value, bool) or value not in ENUMS[key]:
                raise ValueError(f"Invalid {key}")
        elif (isinstance(value, bool) or not isinstance(value, (float, int))
              or not math.isfinite(value) or not BOUNDS[key][0] <= value <= BOUNDS[key][1]):
            raise ValueError(f"Invalid {key}")
    for key in ("resolution", "spp", "seed"):
        if not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    return settings


def find_assets(path=None):
    candidates = [Path(path)] if path else [Path.cwd() / "tutorials/data",
                                          Path(__file__).resolve().parents[3] / "tutorials/data"]
    required = ["mesh/bunny_low.obj", "envmap/ballroom_1k.exr"] + [
        f"cbox/cbox_{part}.obj" for part in
        ("luminaire", "smallball", "largeball", "floor", "ceiling", "back", "greenwall", "redwall")]
    for candidate in candidates:
        if all((candidate / item).is_file() for item in required):
            return str(candidate.resolve())
    raise ValueError("Scene assets missing. Run from the repository root or pass --assets PATH/to/tutorials/data.")


def render_arrays(settings, assets):
    import drjit as dr
    import numpy as np
    import psdr_jit as psdr
    from drjit.cuda import Matrix4f as MatrixC
    from drjit.cuda.ad import Array3f, Float, Matrix4f

    if not dr.has_backend(dr.JitBackend.CUDA):
        raise RuntimeError("Rendering requires an NVIDIA GPU and a CUDA/OptiX driver.")
    dr.set_flag(dr.JitFlag.Debug, True)
    # Each job creates new OptiX program groups. Dr.Jit's in-memory kernel cache
    # can otherwise reuse a linked pipeline from the previous scene with the new
    # SBT, causing UNKNOWN_ENTRY_FUNCTION. The single worker has no other GPU
    # jobs in flight; discard linked kernels here, retaining the disk cache.
    dr.flush_kernel_cache()
    p = {key: Float(settings[key]) for key in ("x", "y", "roughness", "albedo", "light")}
    dr.enable_grad(p[settings["derivative"]])
    sc = psdr.Scene()
    size, spp = settings["resolution"], settings["spp"]
    sc.opts = psdr.RenderOption(size, size, spp, 0, 0)
    sc.opts.log_level = 0
    cornell = settings["scene"] == "cornell"
    camera = psdr.PerspectiveCamera(60 if cornell else 80, .000001, 10000000.)
    camera.to_world = (translate(Matrix4f, [278., 273., -500.]) if cornell else
                       Matrix4f([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
    sc.add_Sensor(camera)
    sc.add_BSDF(psdr.MicrofacetBSDF([.12, .12, .12], [.7, .5, .15], .3), "target")
    root = Path(assets)
    if cornell:
        for name, color in dict(white=[.95]*3, gray=[.5]*3, green=[.2, .9, .2],
                                red=[.9, .2, .2], cyan=[.2, .8, .8]).items():
            sc.add_BSDF(psdr.DiffuseBSDF(color), name)
        for part, material in (("luminaire", "white"), ("smallball", "target"),
                               ("largeball", "cyan"), ("floor", "white"), ("ceiling", "white"),
                               ("back", "gray"), ("greenwall", "green"), ("redwall", "red")):
            emitter = psdr.AreaLight([20, 20, 8]) if part == "luminaire" else None
            transform = translate(MatrixC, [0, -.5, 0]) if emitter else dr.identity(MatrixC)
            sc.add_Mesh(str(root / f"cbox/cbox_{part}.obj"), transform, material, emitter)
        target, units = "Mesh[1]", 100
        sc.param_map["Emitter[0]"].radiance = Array3f(20, 20, 8) * p["light"]
    else:
        sc.add_Mesh(str(root / "mesh/bunny_low.obj"), translate(MatrixC, [0, 0, -100]), "target", None)
        sc.add_EnvironmentMap(str(root / "envmap/ballroom_1k.exr"), dr.identity(dr.scalar.Matrix4f), 1.)
        sc.param_map["Emitter[0]"].scale = p["light"]
        target, units = "Mesh[0]", 30
    material = sc.param_map["BSDF[id=target]"]
    material.roughness.data = p["roughness"]
    material.diffuseReflectance.data = Array3f(*([1, .7, .2] if cornell else [.2, .85, .8])) * p["albedo"]
    # Initialize the camera before configuring the geometry boundary estimators.
    with dr.suspend_grad():
        sc.configure()
    sc.param_map[target].set_transform(translate(Matrix4f, [p["x"] * units, p["y"] * units, 0]))
    if settings["derivative"] in ("x", "y"):
        sc.opts.sppe = spp
        sc.opts.sppse = spp
    sc.configure([0])
    integrator = psdr.PathTracer(2) if settings["integrator"] == "path" else psdr.Direct(2)
    pixels = integrator.renderD(sc, 0, seed=settings["seed"])
    primal = pixels.numpy().T.reshape(size, size, 3).copy()
    dr.forward(p[settings["derivative"]])
    gradient = dr.grad(pixels).numpy().T.reshape(size, size, 3).copy()
    if not np.isfinite(primal).all() or not np.isfinite(gradient).all():
        raise RuntimeError("Renderer produced non-finite pixels; reduce the settings and retry.")
    return primal, gradient


def render_job(settings, assets):
    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    primal, gradient = render_arrays(settings, assets)
    linear = np.maximum(primal * 2 ** settings["exposure"], 0)
    srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
    signed = gradient.mean(axis=2)
    limit = float(np.quantile(np.abs(signed), .99)) or float(np.max(np.abs(signed))) or 1.
    normalized = np.clip(signed / limit, -1, 1)[..., None]
    neutral, negative, positive = np.array([.075, .085, .10]), np.array([.20, .57, 1.]), np.array([1., .30, .16])
    colors = neutral + np.abs(normalized) * (np.where(normalized < 0, negative, positive) - neutral)

    def png(array):
        buffer = io.BytesIO()
        Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    return dict(render=png(srgb), gradient=png(colors), limit=limit,
                seconds=round(time.perf_counter() - started, 3), settings=settings,
                gradient_max=float(np.max(np.abs(gradient))), mean=float(primal.mean()))
