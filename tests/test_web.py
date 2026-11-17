"""Exercise the demo's real scene derivatives and local HTTP job pipeline."""

import base64
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time

import drjit as dr
import numpy as np
from PIL import Image
import pytest

from psdr_jit.web.render import find_assets, render_arrays, validate
from psdr_jit.web.server import Handler, Jobs

ASSETS = str(Path(__file__).resolve().parents[1] / "tutorials/data")
GPU = pytest.mark.skipif(not dr.has_backend(dr.JitBackend.CUDA), reason="CUDA required")


@pytest.mark.parametrize("settings", [None, [], {"scene":"missing"}, {"spp":4096},
    {"x":float("nan")}, {"light":float("inf")}, {"seed":True}, {"seed":1.5},
    {"resolution":128.0}, {"roughness":0}, {"exposure":4}, {"file":"/tmp/scene"}])
def test_reject_invalid_settings(settings):
    with pytest.raises(ValueError):
        validate(settings)


def test_missing_assets(tmp_path):
    with pytest.raises(ValueError, match="--assets"):
        find_assets(tmp_path)


@GPU
@pytest.mark.parametrize("scene", ["cornell", "bunny"])
@pytest.mark.parametrize("derivative", ["x", "y", "roughness", "albedo", "light"])
def test_scene_gradients(scene, derivative):
    settings = validate(dict(scene=scene, derivative=derivative, resolution=64, spp=4))
    image, gradient = render_arrays(settings, ASSETS)
    assert image.shape == gradient.shape == (64, 64, 3)
    assert np.isfinite(image).all() and image.mean() > 0
    assert np.isfinite(gradient).all() and np.max(np.abs(gradient)) > 0
    if derivative == "light":
        # Both scenes are linear in illumination. This also checks the AD connection
        # survives scene configuration, emitter sampling, and the virtual dispatch.
        np.testing.assert_allclose(gradient, image / settings["light"], rtol=2e-4, atol=2e-5)
        brighter, _ = render_arrays(settings | {"light":1.5}, ASSETS)
        # Separate scene builds and float32 sampling can perturb a few pixels.
        # Keep the same-render AD identity above tighter than this comparison.
        np.testing.assert_allclose(brighter, image * 1.5, rtol=1e-3, atol=2e-5)
        np.testing.assert_allclose(brighter.mean(), image.mean() * 1.5, rtol=5e-5)


@GPU
@pytest.mark.parametrize("scene", ["cornell", "bunny"])
def test_direct_geometry(scene):
    image, gradient = render_arrays(validate(dict(scene=scene, integrator="direct", resolution=64, spp=4)), ASSETS)
    assert image.mean() > 0 and np.max(np.abs(gradient)) > 0


@GPU
def test_http_render_pipeline():
    jobs = Jobs(ASSETS)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, jobs=jobs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET" if body is None else "POST", path,
                               None if body is None else json.dumps(body),
                               headers or {"Content-Type":"application/json"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    try:
        assert request("/")[0] == 200
        assert request("/app.js")[0] == 200
        assert request("/style.css")[0] == 200
        assert json.loads(request("/api/config")[1])["defaults"]["scene"] == "cornell"
        assert request("/../server.py")[0] == 404
        assert request("/api/jobs/0")[0] == 404
        assert request("/api/render", {"spp":100000})[0] == 400
        assert request("/api/render", {}, {"Content-Type":"application/json", "Origin":"https://example.com"})[0] == 403
        for scene in ("cornell", "bunny"):
            status, body = request("/api/render", dict(scene=scene, resolution=64, spp=4, derivative="light"))
            assert status == 202
            identifier = json.loads(body)["id"]
            assert request("/api/render", {})[0] == 409
            deadline = time.monotonic() + 180
            while True:
                status, body = request(f"/api/jobs/{identifier}")
                result = json.loads(body)
                assert status == 200
                if result["state"] != "running":
                    break
                assert time.monotonic() < deadline, "GPU worker timed out"
                time.sleep(.1)
            assert result["state"] == "done", result
            output = result["result"]
            assert output["gradient_max"] > 0
            for kind in ("render", "gradient"):
                png = base64.b64decode(output[kind].split(",", 1)[1])
                with Image.open(io.BytesIO(png)) as image:
                    assert image.size == (64, 64) and image.mode == "RGB"
                    assert np.asarray(image).std() > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        jobs.close()
