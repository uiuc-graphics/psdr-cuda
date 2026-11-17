"""GPU regressions for the Dr.Jit migration and scene ownership fixes."""

import gc
import subprocess
import sys
import textwrap
from pathlib import Path

import drjit as dr
import numpy as np
import pytest

if not dr.has_backend(dr.JitBackend.CUDA):
    pytest.skip("CUDA backend required", allow_module_level=True)

import psdr_jit as psdr
from drjit.cuda import Array3f, Array3i, Int
from drjit.cuda.ad import Float, Matrix4f


def quad(x0=-2, x1=2, z=3):
    mesh = psdr.Mesh()
    mesh.load_raw(
        Array3f([[x0, x1, x1, x0], [-2, -2, 2, 2], [z] * 4]),
        Array3i([[0, 0], [2, 3], [1, 2]]),
    )
    return mesh


def scene(material=None):
    result = psdr.Scene()
    result.opts = psdr.RenderOption(12, 8, 4, 0, 0)
    result.opts.log_level = 0
    result.add_Sensor(psdr.PerspectiveCamera(45, .01, 100.))
    result.add_BSDF(material if material is not None else psdr.DiffuseBSDF([.5, .5, .5]), "mat")
    result.add_Mesh(quad(), "mat", None)
    result.configure()
    return result


def test_defaults():
    opts = psdr.RenderOption()
    assert (opts.spp, opts.sppe, opts.sppse) == (1, 0, 0)
    empty = psdr.Scene()
    assert (empty.num_sensors, empty.num_meshes) == (0, 0)


def test_typed_bindings_and_parameter_lifetime():
    sc = scene()
    mesh = sc.param_map["Mesh[0]"]
    matrix = dr.identity(Matrix4f)
    mesh.set_transform(matrix)
    assert dr.allclose(mesh.to_world_left, matrix)
    with pytest.raises(TypeError):
        mesh.set_transform(psdr.RenderOption())
    del sc
    gc.collect()
    assert mesh.num_faces == 2
    assert dr.width(mesh.vertex_positions) == 4


def test_intersection_pointer_array_binding():
    sc = scene()
    hit = sc.unit_ray_intersect(
        psdr.RayC(Array3f(0, 0, 0), Array3f(0, 0, 1)), dr.cuda.Bool(True)
    )
    assert dr.all(hit.is_valid())
    assert isinstance(hit.shape, psdr.MeshPtrC)
    pointers = hit.shape
    mesh = list(pointers)[0]
    assert mesh.num_faces == 2
    assert "Mesh" in repr(pointers)
    del sc, hit, pointers
    gc.collect()
    assert mesh.num_faces == 2
    assert dr.width(mesh.vertex_positions) == 4


def test_invalid_field_is_rejected():
    for field in ("", " ", "unknown"):
        with pytest.raises(RuntimeError):
            psdr.FieldExtractionIntegrator(field)


def test_primal_forward_and_reverse_gradients():
    sc = scene()
    power = Float(1)
    dr.enable_grad(power)
    integrator = psdr.CollocatedIntegrator(power)
    primal = integrator.renderC(sc, seed=42).numpy()
    image = integrator.renderD(sc, seed=42)
    np.testing.assert_allclose(image.numpy(), primal, rtol=2e-5)
    assert primal.shape == (3, 96)
    assert np.isfinite(primal).all() and primal.mean() > 0
    dr.forward(power)
    np.testing.assert_allclose(dr.grad(image).numpy(), primal, rtol=2e-5)

    dr.set_grad(power, 0)
    image = integrator.renderD(sc, seed=42)
    dr.backward(dr.sum(image))
    np.testing.assert_allclose(dr.grad(power).numpy(), primal.sum(), rtol=2e-5)


def test_material_gradient_matches_finite_difference():
    sc = scene()
    material = sc.param_map["BSDF[id=mat]"]
    reflectance = Float(.5)
    dr.enable_grad(reflectance)
    material.reflectance.data = dr.cuda.ad.Array3f(reflectance)
    integrator = psdr.CollocatedIntegrator(1.)
    image = integrator.renderD(sc, seed=17)
    dr.forward(reflectance)
    analytic = dr.grad(image).numpy()
    epsilon = .001
    material.reflectance.data = dr.cuda.ad.Array3f(.5 + epsilon)
    plus = integrator.renderC(sc, seed=17).numpy()
    material.reflectance.data = dr.cuda.ad.Array3f(.5 - epsilon)
    minus = integrator.renderC(sc, seed=17).numpy()
    np.testing.assert_allclose(analytic, (plus - minus) / (2 * epsilon), rtol=3e-4)


def test_per_vertex_material_gradient_matches_finite_difference():
    color = Float([.5] * 4)
    material = psdr.MicrofacetBSDFPerVertex(
        dr.cuda.ad.Array3f(0), dr.cuda.ad.Array3f(color), dr.cuda.ad.Array1f(.8)
    )
    sc = scene(material)
    material = sc.param_map["BSDF[id=mat]"]
    dr.enable_grad(color)
    material.diffuseReflectance = dr.cuda.ad.Array3f(color)
    renderer = psdr.CollocatedIntegrator(1.)
    image = renderer.renderD(sc, seed=17)
    dr.forward(color)
    analytic = dr.grad(image).numpy()
    assert np.max(np.abs(analytic)) > 0
    epsilon = .001
    material.diffuseReflectance = dr.cuda.ad.Array3f([ [.5 + epsilon] * 4 ] * 3)
    plus = renderer.renderC(sc, seed=17).numpy()
    material.diffuseReflectance = dr.cuda.ad.Array3f([ [.5 - epsilon] * 4 ] * 3)
    minus = renderer.renderC(sc, seed=17).numpy()
    np.testing.assert_allclose(analytic, (plus - minus) / (2 * epsilon), rtol=3e-4)


def test_batch_matches_full_render():
    sc = scene()
    integrator = psdr.CollocatedIntegrator(1.)
    indices = Int([0, 5, 27, 95])
    full = integrator.renderC(sc, seed=23).numpy()
    batch = integrator.renderC(sc, seed=23, batch_pix=indices).numpy()
    np.testing.assert_allclose(batch, full[:, [0, 5, 27, 95]], rtol=2e-5)
    batch_ad = integrator.renderD(sc, seed=23, batch_pix=dr.cuda.ad.Int(indices)).numpy()
    np.testing.assert_allclose(batch_ad, batch, rtol=2e-5)


def test_reconfigure_after_adding_mesh_and_material():
    sc = scene()
    sc.add_BSDF(psdr.DiffuseBSDF([.2, .7, .9]), "second")
    sc.add_Mesh(quad(-1, 0, 2), "second", None)
    sc.configure()
    image = psdr.CollocatedIntegrator(1.).renderC(sc).numpy()
    assert sc.num_meshes == 2
    assert np.isfinite(image).all()
    assert np.max(np.abs(image[0] - image[2])) > .01
    sc.configure()
    assert np.isfinite(psdr.CollocatedIntegrator(1.).renderD(sc).numpy()).all()


@pytest.mark.parametrize("integrator", [psdr.PathTracer, psdr.Direct])
def test_area_light_integrators(integrator):
    sc = scene()
    light = quad(-2, -.8, 1)
    light.face_indices = dr.cuda.ad.Array3i([[0, 0], [1, 2], [2, 3]])
    sc.add_Mesh(light, "mat", psdr.AreaLight([10., 10., 10.]))
    sc.configure()
    renderer = integrator(2)
    primal = renderer.renderC(sc, seed=31).numpy()
    image = renderer.renderD(sc, seed=31).numpy()
    assert np.isfinite(image).all() and image.mean() > 0
    np.testing.assert_allclose(image, primal, rtol=1e-3, atol=2e-5)


def test_environment_map_and_multiple_scene_lifetimes():
    first, second = scene(), scene()
    env = psdr.EnvironmentMap()
    env.radiance = psdr.Bitmap3fD(2, 2, dr.cuda.ad.Array3f([[1.] * 4] * 3))
    first.add_EnvironmentMap(env)
    first.configure()
    image = psdr.PathTracer(2).renderC(first, seed=19).numpy()
    assert np.isfinite(image).all() and image.mean() > 0
    del first
    gc.collect()
    assert psdr.CollocatedIntegrator(1.).renderC(second).numpy().mean() > 0


def test_batch_edge_gradients_match_full_render():
    sc = scene()
    sc.opts.sppe = 8
    vertices = sc.param_map["Mesh[0]"].vertex_positions
    vertices.x *= .3
    vertices.y *= .3
    sc.param_map["Mesh[0]"].vertex_positions = vertices
    position = Float(0)
    dr.enable_grad(position)
    transform = dr.identity(Matrix4f)
    transform[0, 3] = position
    sc.param_map["Mesh[0]"].set_transform(transform)
    sc.configure([0])
    renderer = psdr.CollocatedIntegrator(1.)
    full = renderer.renderD(sc, seed=13)
    indices = dr.cuda.ad.Int([0, 5, 27, 95])
    batch = renderer.renderD(sc, seed=13, batch_pix=indices)
    dr.forward(position)
    full_grad = dr.grad(full).numpy()
    assert np.max(np.abs(full_grad)) > 0
    np.testing.assert_allclose(dr.grad(batch).numpy(), full_grad[:, [0, 5, 27, 95]],
                               rtol=2e-4, atol=1e-6)


def test_configured_mesh_copy_does_not_share_owned_caches():
    # Isolate native memory errors so pytest reports a failure instead of dying.
    code = textwrap.dedent("""
        import gc
        from test_renderer import quad, scene, psdr
        mesh = quad()
        mesh.configure()
        sc = scene()
        sc.add_Mesh(mesh, 'mat', None)
        sc.configure()
        del mesh
        gc.collect()
        image = psdr.CollocatedIntegrator(1.).renderC(sc)
        assert image.numpy().mean() > 0
        del sc
        gc.collect()
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_optix_debug_matches_normal_rendering(tmp_path):
    # Native compiler crashes must fail the test instead of killing pytest.
    probe = Path(__file__).with_name("repro_debug_dispatch.py")
    for mode in ("normal", "debug"):
        command = [sys.executable, str(probe), "--output", str(tmp_path / f"{mode}.npz")]
        if mode == "normal":
            command.append("--no-debug")
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, f"{mode}: {result.returncode}\n{result.stdout}\n{result.stderr}"
    with np.load(tmp_path / "normal.npz") as normal, np.load(tmp_path / "debug.npz") as debug:
        assert set(normal.files) == set(debug.files)
        for name in normal.files:
            np.testing.assert_allclose(debug[name], normal[name], rtol=1e-3, atol=2e-5,
                                       err_msg=name)
