#include <psdr/psdr.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <drjit/python.h>

#include <psdr/core/ray.h>
#include <psdr/core/frame.h>
#include <psdr/core/bitmap.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/records.h>
#include <psdr/core/warp.h>
#include <psdr/core/transform.h>

#include <psdr/edge/edge.h>

#include <psdr/sensor/sensor.h>

#include <psdr/sensor/perspective.h>
#include <psdr/sensor/orthographic.h>

#include <psdr/bsdf/bsdf.h>
#include <psdr/bsdf/diffuse.h>
#include <psdr/bsdf/ggx.h>
#include <psdr/bsdf/roughconductor.h>
#include <psdr/bsdf/roughdielectric.h>
#include <psdr/bsdf/microfacet.h>
#include <psdr/bsdf/microfacet_pv.h>

#include <psdr/bsdf/normalmap.h>

#include <psdr/core/pmf.h>

#include <psdr/emitter/emitter.h>

#include <psdr/emitter/area.h>
#include <psdr/emitter/envmap.h>

#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/scene/scene_loader.h>

#include <psdr/integrator/integrator.h>
#include <psdr/integrator/field.h>
#include <psdr/integrator/collocated.h>
#include <psdr/integrator/path.h>
#include <psdr/integrator/direct.h>

#include <psdr/jit_optix_test.h>

namespace py = nanobind;
using namespace py::literals;
using namespace psdr_jit;

using namespace drjit;

// Dr.Jit's default scalar pointer conversion takes ownership. Scene objects
// are borrowed: retain the pointer array (and its originating intersection)
// instead, for indexing, iteration, and formatting alike.
template <typename Array>
void bind_mesh_array(py::module_ &m, const char *name) {
    drjit::ArrayBinding binding;
    auto type = drjit::bind_array_t<Array>(binding, m, name);
    py::type_supplement<drjit::ArraySupplement>(type).item =
        [](PyObject *object, Py_ssize_t index) noexcept -> PyObject * {
            try {
                auto &array = py::cast<const Array &>(py::handle(object));
                if (array.size() == 1) index = 0;
                if (index < 0 || size_t(index) >= array.size()) {
                    PyErr_SetString(PyExc_IndexError, "Mesh pointer index out of range");
                    return nullptr;
                }
                return py::cast(array.entry(size_t(index)),
                                py::rv_policy::reference_internal,
                                py::handle(object)).release().ptr();
            } catch (py::python_error &error) {
                error.restore();
            } catch (const std::exception &error) {
                PyErr_SetString(PyExc_RuntimeError, error.what());
            }
            return nullptr;
        };
}

void optix_jit_test() {
    jit_test test;
    test.trace_ray();
}

void drjit_test() {
    jit_init((uint32_t)JitBackend::CUDA);

    FloatD a = arange<FloatD>(10);
    enable_grad(a);

    FloatD b = a * 2.f;
    std::cout << "val: " << b << std::endl;

    backward(b);
    std::cout << "grad: " << grad(a) << std::endl;

    Matrix4fD           m_to_world_raw = identity<Matrix4fD>();
    std::cout << m_to_world_raw << std::endl;

}

void drjit_memory() {
    FloatC a(1.f, 2.f, 3.f, 4.f);
    std::vector<float> b;
    b.resize(a.size());
    // std::cout << a.slices() << std::endl;

    std::cout << slices<Vector3fC>(a) << std::endl;
    drjit::store(b.data(), a);
}

NB_MODULE(psdr_jit, m) {
    py::module_::import_("drjit");
    py::module_::import_("drjit.cuda");
    py::module_::import_("drjit.cuda.ad");

    m.doc() = "Path-space differentiable renderer";
    m.attr("__name__") = "psdr_jit";

    m.def("optix_jit_test", &optix_jit_test);

    m.def("drjit_test", &drjit_test);

    m.def("drjit_memory", &drjit_memory);

    // Core classes

    py::class_<Object>(m, "Object")
        .def("type_name", &Object::type_name)
        .def_ro("id", &Object::m_id)
        .def("__repr__", &Object::to_string);

    py::class_<RenderOption>(m, "RenderOption")
        .def(py::init<>())
        .def(py::init<int, int, int>(), "width"_a, "height"_a, "spp/sppe"_a)
        .def(py::init<int, int, int, int>(), "width"_a, "height"_a, "spp"_a, "sppe"_a)
        .def(py::init<int, int, int, int, int>(), "width"_a, "height"_a, "spp"_a, "sppe"_a, "sppse"_a)
        .def_rw("width", &RenderOption::width)
        .def_rw("height", &RenderOption::height)
        .def_rw("spp", &RenderOption::spp)
        .def_rw("sppe", &RenderOption::sppe)
        .def_rw("sppse", &RenderOption::sppse)
        .def_rw("log_level", &RenderOption::log_level)
        .def("__repr__",
            [](const RenderOption &ro) {
                std::stringstream oss;
                oss << "[width: " << ro.width << ", height: " << ro.height
                    << ", spp: " << ro.spp << ", sppe: " << ro.sppe << ", sppse: " << ro.sppse
                    << ", log_level: " << ro.log_level << "]";
                return oss.str();
            }
        );
    py::class_<EdgeSortOption>(m, "EdgeSortOption")
        .def(py::init<>())
        .def_rw("enable_sort", &EdgeSortOption::enable_sort)
        .def_rw("local_angle", &EdgeSortOption::local_angle)
        .def_rw("global_angle", &EdgeSortOption::global_angle)
        .def_rw("min_global_step", &EdgeSortOption::min_global_step)
        .def_rw("max_depth", &EdgeSortOption::max_depth);

    py::class_<RayC>(m, "RayC")
        .def(py::init<>())
        .def(py::init<const Vector3fC &, const Vector3fC &>())
        .def("reversed", &RayC::reversed)
        .def_rw("o", &RayC::o)
        .def_rw("d", &RayC::d);

    py::class_<RayD>(m, "RayD")
        .def(py::init<>())
        .def(py::init<const Vector3fD &, const Vector3fD &>())
        .def("reversed", &RayD::reversed)
        .def_rw("o", &RayD::o)
        .def_rw("d", &RayD::d);

    py::class_<FrameC>(m, "FrameC")
        .def(py::init<>())
        .def(py::init<const Vector3fC &>())
        .def_rw("s", &FrameC::s)
        .def_rw("t", &FrameC::t)
        .def_rw("n", &FrameC::n);

    py::class_<FrameD>(m, "FrameD")
        .def(py::init<>())
        .def(py::init<const Vector3fD &>())
        .def_rw("s", &FrameD::s)
        .def_rw("t", &FrameD::t)
        .def_rw("n", &FrameD::n);

    py::class_<Sampler>(m, "Sampler")
        .def(py::init<>())
        .def("seed", &Sampler::seed)
        .def("next_1d", &Sampler::next_1d<false>)
        .def("next_2d", &Sampler::next_2d<false>);

    py::class_<DiscreteDistribution>(m, "DiscreteDistribution")
        .def(py::init<>())
        .def("init", &DiscreteDistribution::init)
        .def("sample", &DiscreteDistribution::sample)
        .def_ro("sum", &DiscreteDistribution::m_sum)
        .def("pmf", &DiscreteDistribution::pmf);

    py::class_<Bitmap1fD>(m, "Bitmap1fD")
        .def(py::init<>())
        .def(py::init<float>())
        .def(py::init<int, int, const FloatD&>())
        .def(py::init<const char*>())
        .def("load_openexr", &Bitmap1fD::load_openexr)
        .def("eval", &Bitmap1fD::eval<true>, "uv"_a, "flip_v"_a = true, "envmap_mode"_a = true)
        .def_rw("resolution", &Bitmap1fD::m_resolution)
        .def_rw("data", &Bitmap1fD::m_data)
        .def_rw("translate", &Bitmap1fD::m_trans)
        .def_rw("rotate", &Bitmap1fD::m_rot)
        .def_rw("scale", &Bitmap1fD::m_scale);

    py::class_<Bitmap3fD>(m, "Bitmap3fD")
        .def(py::init<>())
        .def(py::init<const ScalarVector3f&>())
        .def(py::init<int, int, const Vector3fD&>())
        .def(py::init<const char*>())
        .def("load_openexr", &Bitmap3fD::load_openexr)
        .def("eval", &Bitmap3fD::eval<true>, "uv"_a, "flip_v"_a = true, "envmap_mode"_a = false)
        .def_rw("resolution", &Bitmap3fD::m_resolution)
        .def_rw("data", &Bitmap3fD::m_data)
        .def_rw("translate", &Bitmap3fD::m_trans)
        .def_rw("rotate", &Bitmap3fD::m_rot)
        .def_rw("scale", &Bitmap3fD::m_scale);

    
    py::class_<InteractionC>(m, "InteractionC")
        .def("is_valid", &InteractionC::is_valid)
        .def_ro("wi", &InteractionC::wi)
        .def_ro("p", &InteractionC::p)
        .def_ro("t", &InteractionC::t);

    py::class_<InteractionD>(m, "InteractionD")
        .def("is_valid", &InteractionD::is_valid)
        .def_ro("wi", &InteractionD::wi)
        .def_ro("p", &InteractionD::p)
        .def_ro("t", &InteractionD::t);

    py::class_<IntersectionC, InteractionC>(m, "IntersectionC")
        .def_ro("shape", &IntersectionC::shape)
        .def_ro("n", &IntersectionC::n)
        .def_ro("sh_frame", &IntersectionC::sh_frame)
        .def_ro("uv", &IntersectionC::uv)
        .def_ro("J", &IntersectionC::J);

    py::class_<IntersectionD, InteractionD>(m, "IntersectionD")
        .def_ro("shape", &IntersectionD::shape)
        .def_ro("n", &IntersectionD::n)
        .def_ro("sh_frame", &IntersectionD::sh_frame)
        .def_ro("uv", &IntersectionD::uv)
        .def_ro("J", &IntersectionD::J);

 
    // Records

    py::class_<SampleRecordC>(m, "SampleRecordC")
        .def_ro("pdf", &SampleRecordC::pdf)
        .def_ro("is_valid", &SampleRecordC::is_valid);

    py::class_<SampleRecordD>(m, "SampleRecordD")
        .def_ro("pdf", &SampleRecordD::pdf)
        .def_ro("is_valid", &SampleRecordD::is_valid);

    py::class_<PositionSampleC, SampleRecordC>(m, "PositionSampleC")
        .def_ro("p", &PositionSampleC::p)
        .def_ro("J", &PositionSampleC::J);

    py::class_<PositionSampleD, SampleRecordD>(m, "PositionSampleD")
        .def_ro("p", &PositionSampleD::p)
        .def_ro("J", &PositionSampleD::J);

    py::class_<BSDF, Object>(m, "BSDF")
        .def_rw("twoSide", &BSDF::m_twoSide)
        .def("anisotropic", &BSDF::anisotropic);

    py::class_<NormalMap, BSDF>(m, "NormalMapBSDF")
        .def(py::init<>())
        .def(py::init<const ScalarVector3f&>())
        .def_rw("nested_bsdf", &NormalMap::m_bsdf)
        .def_rw("normal_map", &NormalMap::m_nmap);

    py::class_<Diffuse, BSDF>(m, "DiffuseBSDF")
        .def(py::init<>())
        .def(py::init<const ScalarVector3f&>())
        .def(py::init<const char*>())
        .def(py::init<const Bitmap3fD&>())
        .def_rw("reflectance", &Diffuse::m_reflectance);

    py::class_<RoughConductor, BSDF>(m, "RoughConductorBSDF")
        .def(py::init<>())
        .def(py::init<const Bitmap1fD&, const Bitmap3fD&, const Bitmap3fD&>())
        .def_rw("alpha_u", &RoughConductor::m_alpha_u)
        .def_rw("alpha_v", &RoughConductor::m_alpha_v)
        .def_rw("eta", &RoughConductor::m_eta)
        .def_rw("k", &RoughConductor::m_k)
        .def_rw("specular_reflectance", &RoughConductor::m_specular_reflectance);

    py::class_<RoughDielectric, BSDF>(m, "RoughDielectricBSDF");
    // Sensors

    py::class_<Microfacet, BSDF>(m, "MicrofacetBSDF")
        .def(py::init<>())
        .def(py::init<const ScalarVector3f&, const ScalarVector3f&, float>())
        .def(py::init<const Bitmap3fD&, const Bitmap3fD&, const Bitmap1fD&>())
        .def_rw("roughness", &Microfacet::m_roughness)
        .def_rw("diffuseReflectance", &Microfacet::m_diffuseReflectance)
        .def_rw("specularReflectance", &Microfacet::m_specularReflectance);

    py::class_<MicrofacetPerVertex, BSDF>(m, "MicrofacetBSDFPerVertex")
        .def(py::init<const Vector3fD&, const Vector3fD&, const Vector1fD&>())
        .def_rw("roughness", &MicrofacetPerVertex::m_roughness)
        .def_rw("diffuseReflectance", &MicrofacetPerVertex::m_diffuseReflectance)
        .def_rw("specularReflectance", &MicrofacetPerVertex::m_specularReflectance);

    // Shapes

    py::class_<Mesh, Object>(m, "Mesh")
        .def(py::init<>())
        .def("load_raw", &Mesh::load_raw, "v"_a, "f"_a, "uv"_a = Vector2fC(), "f_uv"_a = Vector3iC(), "verbose"_a = false)
        .def("load", &Mesh::load, "filename"_a, "verbose"_a = false)
        .def("configure", &Mesh::configure)
        .def("set_transform", &Mesh::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Mesh::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_position", py::overload_cast<const Vector2fC&, MaskC>(&Mesh::sample_position, py::const_), "sample2"_a, "active"_a = true)
        .def("sample_position", py::overload_cast<const Vector2fD&, MaskD>(&Mesh::sample_position, py::const_), "sample2"_a, "active"_a = true)
        .def_ro("num_vertices", &Mesh::m_num_vertices)
        .def_ro("num_faces", &Mesh::m_num_faces)
        .def_ro("bsdf", &Mesh::m_bsdf)
        .def_rw("to_world", &Mesh::m_to_world_raw)
        .def_rw("to_world_right", &Mesh::m_to_world_right)
        .def_rw("to_world_left", &Mesh::m_to_world_left)
        .def_rw("vertex_positions_T", &Mesh::m_vertex_positions, "Object-space vertex positions")
        .def_rw("vertex_positions", &Mesh::m_vertex_positions_raw, "Object-space vertex positions")
        .def_ro("vertex_normals", &Mesh::m_vertex_normals_raw, "Object-space vertex normals")
        .def_rw("vertex_uv", &Mesh::m_vertex_uv)
        .def_rw("face_indices", &Mesh::m_face_indices)
        .def_rw("face_uv_indices", &Mesh::m_face_uv_indices)
        .def_rw("valid_edge_indices", &Mesh::m_valid_edge_indices)
        .def_rw("use_face_normal", &Mesh::m_use_face_normals)
        .def_rw("enable_edges", &Mesh::m_enable_edges) // TODO
        .def("edge_indices", [](const Mesh &mesh) { return head<4>(mesh.m_edge_indices); })
        .def("dump", &Mesh::dump);

    py::class_<Emitter, Object>(m, "Emitter");

    py::class_<AreaLight, Emitter>(m, "AreaLight")
        .def(py::init<const ScalarVector3f&>())
        .def(py::init<const ScalarVector3f&, const Mesh*>())
        .def_rw("radiance", &AreaLight::m_radiance);

    py::class_<EnvironmentMap, Emitter>(m, "EnvironmentMap")
        .def(py::init<>())
        .def(py::init<const char *>())
        .def_ro("to_world", &EnvironmentMap::m_to_world_raw)
        .def("set_transform", &EnvironmentMap::set_transform)
        .def_rw("radiance", &EnvironmentMap::m_radiance)
        .def_rw("scale", &EnvironmentMap::m_scale);
        
    py::class_<Sensor, Object>(m, "Sensor")
        .def("set_transform", &Sensor::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Sensor::append_transform, "mat"_a, "append_left"_a = true)
        .def_rw("to_world", &Sensor::m_to_world_raw)
        .def_rw("to_world_left", &Sensor::m_to_world_left)
        .def_rw("to_world_right", &Sensor::m_to_world_right);

    py::class_<PerspectiveCamera, Sensor>(m, "PerspectiveCamera")
        .def(py::init<float, float, float>())
        .def(py::init<float, float, float, float, float, float>())
        .def("set_transform", &Sensor::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Sensor::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_direct", &Sensor::sample_direct, "vec"_a)
        .def_ro("world_to_sample", &PerspectiveCamera::m_world_to_sample)
        .def_rw("to_world", &PerspectiveCamera::m_to_world_raw)
        .def_rw("to_world_left", &Sensor::m_to_world_left)
        .def_rw("to_world_right", &Sensor::m_to_world_right);

    py::class_<OrthographicCamera, Sensor>(m, "OrthographicCamera")
        .def(py::init<float, float>())
        .def("set_transform", &Sensor::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Sensor::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_direct", &Sensor::sample_direct, "vec"_a)
        .def_ro("world_to_sample", &OrthographicCamera::m_world_to_sample)
        .def_rw("to_world", &OrthographicCamera::m_to_world_raw)
        .def_rw("to_world_left", &Sensor::m_to_world_left)
        .def_rw("to_world_right", &Sensor::m_to_world_right);

    py::class_<SensorDirectSample_<FloatC>, SampleRecord_<FloatC>>(m, "SensorDirectSample")
        .def_rw("q", &SensorDirectSample_<FloatC>::q)
        .def_rw("pixel_idx", &SensorDirectSample_<FloatC>::pixel_idx)
        .def_rw("sensor_val", &SensorDirectSample_<FloatC>::sensor_val)
        .def_rw("is_valid", &SensorDirectSample_<FloatC>::is_valid);

    py::class_<Scene, Object>(m, "Scene")
        .def(py::init<>())

        .def("add_Sensor", &Scene::add_Sensor, "Add Sensor")
        .def("add_EnvironmentMap", py::overload_cast<const char *, ScalarMatrix4f, float>(&Scene::add_EnvironmentMap), "Add EnvironmentMap")
        .def("add_EnvironmentMap", py::overload_cast<EnvironmentMap *>(&Scene::add_EnvironmentMap), "Add EnvironmentMap")
        .def("add_Mesh", py::overload_cast<const char *, Matrix4fC, const char *, Emitter *>(&Scene::add_Mesh), "Add Mesh", "filename"_a, "to_world"_a, "bsdf_id"_a, "emitter"_a.none())
        .def("add_Mesh", py::overload_cast<Mesh *, const char *, Emitter *>(&Scene::add_Mesh), "Add Mesh", "mesh"_a, "bsdf_id"_a, "emitter"_a = nullptr)
        .def("add_BSDF", &Scene::add_BSDF, "Add BSDf", "bsdf"_a, "name"_a, "twoSide"_a = false)
        .def("add_normalmap_BSDF", &Scene::add_normalmap_BSDF, "Add add_normalmap_BSDF", "bsdf1"_a, "bsdf2"_a, "name"_a, "twoSide"_a = false)

        .def("unit_ray_intersect", &Scene::unit_ray_intersect<false>, py::keep_alive<0, 1>())
        .def("unit_ray_intersectAD", &Scene::unit_ray_intersect<true>, py::keep_alive<0, 1>())

        .def("load_file", &Scene::load_file, "file_name"_a, "auto_configure"_a = true)
        .def("load_string", &Scene::load_string, "scene_xml"_a, "auto_configure"_a = true)
        .def("configure", &Scene::configure, "active_sensor"_a=std::vector<int>())
        .def_rw("opts", &Scene::m_opts, "Render options")
        .def_rw("seed", &Scene::seed, "Sample seed")
        .def_ro("num_sensors", &Scene::m_num_sensors)
        .def_ro("num_meshes", &Scene::m_num_meshes)
        .def("get_num_emitters", [](const Scene &scene){ return scene.m_emitters.size(); })
        .def_ro("param_map", &Scene::m_param_map, "Parameter map");

    py::class_<Integrator, Object>(m, "Integrator")
        .def("renderC", &Integrator::renderC, "scene"_a, "sensor_id"_a = 0, "seed"_a = -1, "batch_pix"_a = -1)
        .def("renderD", &Integrator::renderD, "scene"_a, "sensor_id"_a = 0, "seed"_a = -1, "batch_pix"_a = -1);

    py::class_<FieldExtractionIntegrator, Integrator>(m, "FieldExtractionIntegrator")
        .def(py::init<const char*>());

    py::class_<CollocatedIntegrator, Integrator>(m, "CollocatedIntegrator")
        .def_rw("m_intensity", &CollocatedIntegrator::m_intensity)
        .def(py::init<float>())
        .def(py::init<const FloatD &>());

    py::class_<PathTracer, Integrator>(m, "PathTracer")
        .def(py::init<int>(), "max_depth"_a = 1)
        .def("preprocess_secondary_edges", &PathTracer::preprocess_secondary_edges, "scene"_a, "sensor_id"_a, "resolution"_a, "nrounds"_a = 1, "seed"_a = 0)
        .def_rw("hide_emitters", &PathTracer::m_hide_emitters);

    py::class_<DirectIntegrator, Integrator>(m, "Direct")
        .def(py::init<int>(), "mis"_a = 2)
        .def("preprocess_secondary_edges", &DirectIntegrator::preprocess_secondary_edges, "scene"_a, "sensor_id"_a, "resolution"_a, "nrounds"_a = 1, "seed"_a = 0)
        .def_rw("hide_emitters", &DirectIntegrator::m_hide_emitters);

    // Intersection.shape exposes renderer-specific Dr.Jit pointer arrays.
    bind_mesh_array<MeshArrayC>(m, "MeshPtrC");
    bind_mesh_array<MeshArrayD>(m, "MeshPtrD");

}
