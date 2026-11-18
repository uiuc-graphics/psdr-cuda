#include <chrono>
#include <misc/Exception.h>
#include <drjit-core/jit.h>

#include <psdr/core/ray.h>
#include <psdr/core/transform.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/pmf.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/bsdf/diffuse.h>
#include <psdr/bsdf/microfacet.h>
#include <psdr/bsdf/microfacet_pv.h>
#include <psdr/bsdf/normalmap.h>
#include <psdr/bsdf/roughconductor.h>

#include <psdr/emitter/area.h>

#include <psdr/emitter/envmap.h>
#include <psdr/sensor/perspective.h>
#include <psdr/sensor/orthographic.h>
#include <psdr/shape/mesh.h>

#include <psdr/scene/scene_optix.h>
#include <psdr/scene/scene_loader.h>
#include <psdr/scene/scene.h>
#include <psdr/core/bitmap.h>

NAMESPACE_BEGIN(psdr_jit)

template <typename T>
void build_param_map(Scene::ParamMap &param_map, const std::vector<T*> &arr, const char *name) {
    for ( size_t i = 0; i < arr.size(); ++i ) {
        const T *obj = arr[i];

        std::stringstream oss1;
        std::operator<<(oss1, name); std::operator<<(oss1, "["); oss1 << i; std::operator<<(oss1, "]");
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *obj));

        if ( obj->m_id != "" ) {
            std::stringstream oss2;
            std::operator<<(oss2, name); std::operator<<(oss2, "[id="); std::operator<<(oss2, obj->m_id); std::operator<<(oss2, "]");

            bool is_new;
            std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *obj));
            PSDR_ASSERT_MSG(is_new, std::string("Duplicate id: ") + obj->m_id);
        }
    }
}

Scene::Scene() {
    m_loaded = true;
    m_samplers = new Sampler[3];
    m_optix = new Scene_OptiX();
    m_emitter_env = nullptr;
    m_emitters_distrb = new DiscreteDistribution();
    m_sec_edge_distrb = new DiscreteDistribution();
}


Scene::~Scene() {
    for ( Sensor*  s : m_sensors  ) delete s;
    for ( Emitter* e : m_emitters ) delete e;
    for ( BSDF*    b : m_bsdfs    ) delete b;
    for ( Mesh*    m : m_meshes   ) delete m;

    delete[]    m_samplers;
    delete      m_optix;
    delete      m_emitters_distrb;
    delete      m_sec_edge_distrb;

}


void Scene::load_file(const char *file_name, bool auto_configure) {
    SceneLoader::load_from_file(file_name, *this);
    if ( auto_configure ) configure();
}


void Scene::load_string(const char *scene_xml, bool auto_configure) {
    SceneLoader::load_from_string(scene_xml, *this);
    if ( auto_configure ) configure();
}

void Scene::add_EnvironmentMap(const char *fname, ScalarMatrix4f to_world, float scale) {
    PSDR_ASSERT_MSG(m_emitter_env == nullptr, "A scene is only allowed to have one envmap!");

    EnvironmentMap *emitter = new EnvironmentMap(fname);
    emitter->m_scale = scale;
    emitter->m_to_world_raw = Matrix4fD(to_world);
    m_emitters.push_back(emitter);
    m_emitter_env = emitter;

    build_param_map<Emitter>(m_param_map, m_emitters, "Emitter");
}

void Scene::add_EnvironmentMap(EnvironmentMap *emitter_) {
    PSDR_ASSERT_MSG(m_emitter_env == nullptr, "A scene is only allowed to have one envmap!");

    EnvironmentMap *emitter = new EnvironmentMap(*emitter_);
    m_emitters.push_back(emitter);
    m_emitter_env = emitter;

    build_param_map<Emitter>(m_param_map, m_emitters, "Emitter");
}

void Scene::add_Sensor(Sensor* sensor) {
    if ( m_opts.log_level > 0 ) std::cout << "add_Sensor: " << sensor->to_string() << std::endl;
    if (PerspectiveCamera *sensor_buff = dynamic_cast<PerspectiveCamera *>(sensor)) {
        Sensor *sensor_temp = new PerspectiveCamera(*sensor_buff);
        sensor_temp->m_to_world_raw = Matrix4fD(Matrix4fD(sensor_buff->m_to_world_raw));
        m_sensors.push_back(sensor_temp);
        build_param_map<Sensor >(m_param_map, m_sensors , "Sensor" );
        m_num_sensors = static_cast<int>(m_sensors.size());
    } 
    else if (OrthographicCamera *sensor_buff = dynamic_cast<OrthographicCamera *>(sensor)) {
        Sensor *sensor_temp = new OrthographicCamera(*sensor_buff);
        sensor_temp->m_to_world_raw = Matrix4fD(Matrix4fD(sensor_buff->m_to_world_raw));
        m_sensors.push_back(sensor_temp);
        build_param_map<Sensor >(m_param_map, m_sensors , "Sensor" );
        m_num_sensors = static_cast<int>(m_sensors.size());
    }
    else {
        PSDR_ASSERT_MSG(0, "Unknown sensor type!");
    }
}

void Scene::add_normalmap_BSDF(NormalMap* bsdf, Microfacet* bsdf_micro, const char *bsdf_id, bool twoSide) {
        if ( m_opts.log_level > 0 ) std::cout << "add_BSDF: " << "NormalMapBSDF" << " " << bsdf_id << std::endl;
        NormalMap *bsdf_temp = new NormalMap(bsdf->m_nmap);
        bsdf_temp->m_twoSide = twoSide;
        bsdf_temp->m_id = bsdf_id;
        Microfacet *nmap_b = new Microfacet(bsdf_micro->m_specularReflectance, bsdf_micro->m_diffuseReflectance, bsdf_micro->m_roughness);
        // std::cout << bsdf_micro->m_diffuseReflectance.m_data << std::endl;
        bsdf_temp->m_bsdf = nmap_b;
        m_bsdfs.push_back(bsdf_temp);
        Scene::ParamMap &param_map = m_param_map;
        std::stringstream oss1, oss2;
        std::operator<<(oss1, "BSDF["); oss1 << m_bsdfs.size() - 1; std::operator<<(oss1, "]");
        std::operator<<(oss2, "BSDF[id="); std::operator<<(oss2, bsdf_id); std::operator<<(oss2, "]");
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *bsdf_temp));
        bool is_new;
        std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *bsdf_temp));
        PSDR_ASSERT_MSG(is_new, std::string("Duplicate BSDF id: ") + bsdf_id);
}


void Scene::add_BSDF(BSDF* bsdf, const char *bsdf_id, bool twoSide) {
    if (Diffuse *bsdf_buff = dynamic_cast<Diffuse *>(bsdf)) {
        if ( m_opts.log_level > 0 ) std::cout << "add_BSDF: " << "Diffuse" << " " << bsdf_id << std::endl;
        Diffuse *bsdf_temp = new Diffuse(bsdf_buff->m_reflectance);
        bsdf_temp->m_twoSide = twoSide;

        bsdf_temp->m_id = bsdf_id;
        m_bsdfs.push_back(bsdf_temp);

        Scene::ParamMap &param_map = m_param_map;
        std::stringstream oss1, oss2;
        std::operator<<(oss1, "BSDF["); oss1 << m_bsdfs.size() - 1; std::operator<<(oss1, "]");
        std::operator<<(oss2, "BSDF[id="); std::operator<<(oss2, bsdf_id); std::operator<<(oss2, "]");
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *bsdf_temp));

        bool is_new;
        std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *bsdf_temp));
        PSDR_ASSERT_MSG(is_new, std::string("Duplicate BSDF id: ") + bsdf_id);

    } else if (Microfacet *bsdf_buff = dynamic_cast<Microfacet *>(bsdf)) {
        if ( m_opts.log_level > 0 ) std::cout << "add_BSDF: " << "Microfacet" << " " << bsdf_id << std::endl;
        Microfacet *bsdf_temp = new Microfacet(bsdf_buff->m_specularReflectance, bsdf_buff->m_diffuseReflectance, bsdf_buff->m_roughness);
        bsdf_temp->m_twoSide = twoSide;
        bsdf_temp->m_id = bsdf_id;
        m_bsdfs.push_back(bsdf_temp);

        Scene::ParamMap &param_map = m_param_map;
        std::stringstream oss1, oss2;
        std::operator<<(oss1, "BSDF["); oss1 << m_bsdfs.size() - 1; std::operator<<(oss1, "]");
        std::operator<<(oss2, "BSDF[id="); std::operator<<(oss2, bsdf_id); std::operator<<(oss2, "]");
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *bsdf_temp));

        bool is_new;
        std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *bsdf_temp));
        PSDR_ASSERT_MSG(is_new, std::string("Duplicate BSDF id: ") + bsdf_id);

    } else if (RoughConductor *bsdf_buff = dynamic_cast<RoughConductor *>(bsdf)) {
        if ( m_opts.log_level > 0 ) std::cout << "add_BSDF: " << "RoughConductor" << " " << bsdf_id << std::endl;
        RoughConductor *bsdf_temp = new RoughConductor(bsdf_buff->m_alpha_u, bsdf_buff->m_alpha_v, bsdf_buff->m_eta, bsdf_buff->m_k, bsdf_buff->m_specular_reflectance);
        bsdf_temp->m_twoSide = twoSide;
        bsdf_temp->m_id = bsdf_id;
        m_bsdfs.push_back(bsdf_temp);

        Scene::ParamMap &param_map = m_param_map;
        std::stringstream oss1, oss2;
        std::operator<<(oss1, "BSDF["); oss1 << m_bsdfs.size() - 1; std::operator<<(oss1, "]");
        std::operator<<(oss2, "BSDF[id="); std::operator<<(oss2, bsdf_id); std::operator<<(oss2, "]");
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *bsdf_temp));

        bool is_new;
        std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *bsdf_temp));
        PSDR_ASSERT_MSG(is_new, std::string("Duplicate BSDF id: ") + bsdf_id);

    } else if (MicrofacetPerVertex *bsdf_buff = dynamic_cast<MicrofacetPerVertex *>(bsdf)) {
        if ( m_opts.log_level > 0 ) std::cout << "add_BSDF: " << "MicrofacetPerVertex" << " " << bsdf_id << std::endl;
        MicrofacetPerVertex *bsdf_temp = new MicrofacetPerVertex(bsdf_buff->m_specularReflectance, bsdf_buff->m_diffuseReflectance, bsdf_buff->m_roughness);
        bsdf_temp->m_twoSide = twoSide;
        bsdf_temp->m_id = bsdf_id;
        m_bsdfs.push_back(bsdf_temp);

        Scene::ParamMap &param_map = m_param_map;
        std::stringstream oss1, oss2;
        std::operator<<(oss1, "BSDF["); oss1 << m_bsdfs.size() - 1; std::operator<<(oss1, "]");
        std::operator<<(oss2, "BSDF[id="); std::operator<<(oss2, bsdf_id); std::operator<<(oss2, "]");
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *bsdf_temp));

        bool is_new;
        std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *bsdf_temp));
        PSDR_ASSERT_MSG(is_new, std::string("Duplicate BSDF id: ") + bsdf_id);

    }
    else if (NormalMap *bsdf_buff = dynamic_cast<NormalMap *>(bsdf)) {
        if ( m_opts.log_level > 0 ) std::cout << "add_BSDF: " << "NormalMapBSDF" << " " << bsdf_id << std::endl;
        NormalMap *bsdf_temp = new NormalMap(ScalarVector3f(.499999f,.499999f,1.f));
        bsdf_temp->m_twoSide = twoSide;
        bsdf_temp->m_id = bsdf_id;

        Microfacet *nmap_b = new Microfacet();

        bsdf_temp->m_bsdf = nmap_b;


        m_bsdfs.push_back(bsdf_temp);



        Scene::ParamMap &param_map = m_param_map;
        std::stringstream oss1, oss2;
        std::operator<<(oss1, "BSDF["); oss1 << m_bsdfs.size() - 1; std::operator<<(oss1, "]");
        std::operator<<(oss2, "BSDF[id="); std::operator<<(oss2, bsdf_id); std::operator<<(oss2, "]");
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *bsdf_temp));

        bool is_new;
        std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *bsdf_temp));
        PSDR_ASSERT_MSG(is_new, std::string("Duplicate BSDF id: ") + bsdf_id);

    } else {
        PSDR_ASSERT_MSG(0, "Unknown BSDF type!");
    }
}

void Scene::add_Mesh(const char *fname, Matrix4fC transform, const char *bsdf_id, Emitter *emitter) {
    if ( m_opts.log_level > 0 ) std::cout << "add_Mesh: " << m_meshes.size() << std::endl;
    std::stringstream oss;
    std::operator<<(oss, "BSDF[id="); std::operator<<(oss, bsdf_id); std::operator<<(oss, "]");
    auto bsdf_info = m_param_map.find(oss.str());
    PSDR_ASSERT_MSG(bsdf_info != m_param_map.end(), std::string("Unknown BSDF id: ") + bsdf_id);

    Mesh *mesh = new Mesh();
    mesh->load(fname, false);
    mesh->m_bsdf = dynamic_cast<const BSDF*>(&bsdf_info->second);

    mesh->m_to_world_raw = Matrix4fD(transform);
    mesh->m_mesh_id = m_meshes.size();

    if ( emitter ) {
        if (AreaLight *emitter_buff = dynamic_cast<AreaLight *>(emitter)) {
            if ( m_opts.log_level > 0 ) std::cout << "add Area light" << std::endl;
            AreaLight *emitter_temp = new AreaLight(mesh);
            emitter_temp->m_radiance = emitter_buff->m_radiance;
            m_emitters.push_back(emitter_temp);
            mesh->m_emitter = emitter_temp;
        } else {
            PSDR_ASSERT_MSG(0, "Unknown emitter type!");
        }
        build_param_map<Emitter>(m_param_map, m_emitters, "Emitter");
    }

    m_meshes.push_back(mesh);
    build_param_map<Mesh>(m_param_map, m_meshes, "Mesh");
    m_num_meshes = static_cast<int>(m_meshes.size());
}

void Scene::add_Mesh(Mesh *mesh_, const char *bsdf_id, Emitter *emitter) {
    if ( m_opts.log_level > 0 ) std::cout << "add_Mesh: " << m_meshes.size() << std::endl;
    Mesh *mesh = new Mesh(*mesh_);

    mesh->m_mesh_id = m_meshes.size();

    std::stringstream oss;
    std::operator<<(oss, "BSDF[id="); std::operator<<(oss, bsdf_id); std::operator<<(oss, "]");
    auto bsdf_info = m_param_map.find(oss.str());
    PSDR_ASSERT_MSG(bsdf_info != m_param_map.end(), std::string("Unknown BSDF id: ") + bsdf_id);
    mesh->m_bsdf = dynamic_cast<const BSDF*>(&bsdf_info->second);

    if ( emitter ) {
        if (AreaLight *emitter_buff = dynamic_cast<AreaLight *>(emitter)) {
            if ( m_opts.log_level > 0 ) std::cout << "add Area light" << std::endl;
            AreaLight *emitter_temp = new AreaLight(mesh);
            emitter_temp->m_radiance = emitter_buff->m_radiance;
            m_emitters.push_back(emitter_temp);
            mesh->m_emitter = emitter_temp;
        } else {
            PSDR_ASSERT_MSG(0, "Unknown emitter type!");
        }
        build_param_map<Emitter>(m_param_map, m_emitters, "Emitter");
    }

    m_meshes.push_back(mesh);
    build_param_map<Mesh>(m_param_map, m_meshes, "Mesh");
    m_num_meshes = static_cast<int>(m_meshes.size());
}

void Scene::configure(std::vector<int> active_sensor) {
    for (int sensor_id : active_sensor) {
        PSDR_ASSERT_MSG(sensor_id >= 0 && static_cast<size_t>(sensor_id) < m_sensors.size(),
                        "Active sensor index out of range");
    }
    
    // Build the parameter map

    m_loaded = true;
    PSDR_ASSERT_MSG(m_loaded, "Scene not loaded yet!");

    if ( m_opts.log_level > 0 ) {
        std::cout << "[Scene] resolution: " << m_opts.height << " " << m_opts.width << std::endl;
    }
    PSDR_ASSERT(m_num_sensors == static_cast<int>(m_sensors.size()));
    PSDR_ASSERT(m_num_meshes == static_cast<int>(m_meshes.size()));

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();



    // Seed samplers
    if ( m_opts.spp  > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.height)*m_opts.width*m_opts.spp;
        if ( m_samplers[0].m_sample_count != sample_count )
            m_samplers[0].seed(arange<UInt64C>(sample_count)+seed);
    }
    if ( m_opts.sppe > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.height)*m_opts.width*m_opts.sppe;
        if ( m_samplers[1].m_sample_count != sample_count )
            m_samplers[1].seed(arange<UInt64C>(sample_count)+seed);
    }
    if ( m_opts.sppse > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.height)*m_opts.width*m_opts.sppse;
        if ( m_samplers[2].m_sample_count != sample_count )
            m_samplers[2].seed(arange<UInt64C>(sample_count)+seed);
    }

    // Preprocess meshes
    PSDR_ASSERT_MSG(!m_meshes.empty(), "Missing meshes!");
    std::vector<int> face_offset, edge_offset;
    face_offset.reserve(m_num_meshes + 1);
    face_offset.push_back(0);
    edge_offset.reserve(m_num_meshes + 1);
    edge_offset.push_back(0);


    m_lower = full<Vector3fC>(std::numeric_limits<float>::max());
    m_upper = full<Vector3fC>(std::numeric_limits<float>::lowest());
    for ( Mesh *mesh : m_meshes ) {
        if (mesh == m_bound_mesh) {
            face_offset.push_back(face_offset.back() + mesh->m_num_faces);
            edge_offset.push_back(edge_offset.back());
            continue;
        }
        mesh->configure();
        face_offset.push_back(face_offset.back() + mesh->m_num_faces);
        if ( m_opts.sppse > 0 && mesh->m_enable_edges ) {
            // std::cout << (mesh->m_sec_edge_info)->size() << std::endl;
            edge_offset.push_back(edge_offset.back() + static_cast<int>((mesh->m_sec_edge_info)->size()));
        } else {
            edge_offset.push_back(edge_offset.back());
        }
        for ( int i = 0; i < 3; ++i ) {
            m_lower[i] = minimum(m_lower[i], min(detach<false>(mesh->m_vertex_positions[i])));
            m_upper[i] = maximum(m_upper[i], max(detach<false>(mesh->m_vertex_positions[i])));
        }
    }


    


    // Preprocess sensors
    PSDR_ASSERT_MSG(!m_sensors.empty(), "Missing sensor!");
    std::vector<size_t> num_edges;
    if ( m_opts.sppe > 0 ) num_edges.reserve(m_sensors.size());

    for (int id = 0; id < m_num_sensors; ++id) {
        Sensor *sensor = m_sensors[id];
        const bool active = active_sensor.empty() ||
            std::find(active_sensor.begin(), active_sensor.end(), id) != active_sensor.end();
        sensor->m_enable_edges = false;
        if (active) {
            sensor->m_resolution = ScalarVector2i(m_opts.width, m_opts.height);
            sensor->m_scene = this;
            sensor->configure(true);
            if (m_opts.sppe > 0)
                num_edges.push_back(sensor->m_enable_edges ? sensor->m_edge_distrb.m_size : 0);
        }
        if (!sensor->m_enable_edges) {
            sensor->m_edge_info = {};
            sensor->m_edge_distrb = {};
        }
        if (auto *camera = dynamic_cast<PerspectiveCamera *>(sensor)) {
            // An inactive camera may never have been configured.
            Vector3fC position = active ? detach<false>(camera->m_camera_pos) :
                detach<false>(transform_pos(camera->m_to_world_left * camera->m_to_world_raw *
                                            camera->m_to_world_right, zeros<Vector3fD>()));
            m_lower = minimum(m_lower, position);
            m_upper = maximum(m_upper, position);
        }
    }

    if ( m_opts.log_level > 0 ) {
        std::stringstream oss;
        oss << "AABB: [lower = " << drjit::string(m_lower).c_str()
            << ", upper = " << drjit::string(m_upper).c_str() << "]";
        log(oss.str().c_str());

        if ( m_opts.sppe > 0 ) {
            std::stringstream oss;
            std::operator<<(oss, "("); oss << num_edges[0];
            for ( size_t i = 1; i < num_edges.size(); ++i ) { std::operator<<(oss, ", "); oss << num_edges[i]; }
            std::operator<<(oss, ") primary edges initialized.");
            log(oss.str().c_str());
        }
    }


    // Handling env. lighting
    if ( m_emitter_env != nullptr ) {
        FloatC margin = min((m_upper - m_lower)*0.05f);
        m_lower -= margin; m_upper += margin;

        m_emitter_env->m_lower = m_lower;
        m_emitter_env->m_upper = m_upper;

        if (!m_bound_mesh) {
            const int faces[3][12] = {
                {0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4},
                {1, 3, 5, 7, 3, 7, 5, 4, 2, 6, 7, 6},
                {3, 2, 7, 3, 7, 6, 1, 5, 6, 4, 5, 7}
            };
            m_bound_mesh = new Mesh();
            m_bound_mesh->m_num_vertices = 8;
            m_bound_mesh->m_num_faces = 12;
            m_bound_mesh->m_use_face_normals = true;
            m_bound_mesh->m_enable_edges = false;
            m_bound_mesh->m_emitter = m_emitter_env;
            for (int i = 0; i < 3; ++i)
                m_bound_mesh->m_face_indices[i] = load<IntD>(faces[i], 12);
            m_meshes.push_back(m_bound_mesh);
            face_offset.push_back(face_offset.back() + 12);
            edge_offset.push_back(edge_offset.back());
            ++m_num_meshes;
        }
        const IntC corners = arange<IntC>(8);
        for (int axis = 0; axis < 3; ++axis)
            m_bound_mesh->m_vertex_positions_raw[axis] = select(
                neq(corners & (1 << axis), 0), m_upper[axis], m_lower[axis]);
        m_bound_mesh->configure();
        if ( m_opts.log_level > 0 ) {
            log("Updated environment bounds.");
        }
    }


    // Preprocess emitters
    if ( !m_emitters.empty() ) {
        std::vector<float> weights;
        weights.reserve(m_emitters.size());

        double total_weight = 0.0;

        for ( Emitter *e : m_emitters ) {
            e->configure();
            total_weight += e->m_sampling_weight;
        }
        // std::cout << total_weight << std::endl;
        for ( Emitter *e : m_emitters ) {
            if (EnvironmentMap *enve = dynamic_cast<EnvironmentMap *>(e)) {
                e->m_sampling_weight = total_weight > 0.0 ? total_weight : 1.0;
            }
        }

        for ( Emitter *e : m_emitters ) {
            weights.push_back(e->m_sampling_weight);
        }
        m_emitters_distrb->init(load<FloatC>(weights.data(), weights.size()));

        float inv_total_weight = rcp(m_emitters_distrb->m_sum)[0];
        for ( Emitter *e : m_emitters ) {
            e->m_sampling_weight *= inv_total_weight;
        }
    }


    // PSDR_ASSERT(0);


    // Register every configured scene object, including newly added objects.
    // Destruction unregisters each object without affecting other live scenes.
    for (auto *object : m_meshes) object->register_instance();
    for (auto *object : m_bsdfs) object->register_instance();
    for (auto *object : m_emitters) object->register_instance();
    for (auto *object : m_sensors) object->register_instance();

    // Initialize CUDA arrays
    if ( !m_emitters.empty() ) {
        m_emitters_cuda = load<EmitterArrayD>(m_emitters.data(), m_emitters.size());

    }
    m_meshes_cuda = load<MeshArrayD>(m_meshes.data(), m_num_meshes);

    // Generate global triangle arrays
    m_triangle_info = empty<TriangleInfoD>(face_offset.back());
    m_triangle_uv = zeros<TriangleUVD>(face_offset.back());
    m_triangle_face_normals = empty<MaskD>(face_offset.back());
    for ( int i = 0; i < m_num_meshes; ++i ) {
        const Mesh &mesh = *m_meshes[i];
        // std::cout << mesh.m_num_faces << std::endl;
        // std::cout << face_offset[i] << std::endl;
        const IntD idx = arange<IntD>(mesh.m_num_faces) + face_offset[i];
        scatter(m_triangle_info, *mesh.m_triangle_info, idx);
        scatter(m_triangle_face_normals, MaskD(mesh.m_use_face_normals), idx);
        if ( mesh.m_has_uv ) {
            scatter(m_triangle_uv, *mesh.m_triangle_uv, idx);
        }
    }



    // Generate global sec. edge arrays
    if ( m_opts.sppse > 0 ) {
        m_sec_edge_info = empty<SecondaryEdgeInfo>(edge_offset.back());
        for ( int i = 0; i < m_num_meshes; ++i ) {
            const Mesh &mesh = *m_meshes[i];
            if ( mesh.m_enable_edges ) {
                PSDR_ASSERT(mesh.m_sec_edge_info != nullptr);
                const int m = static_cast<int>((mesh.m_sec_edge_info->size()));
                const IntD idx = arange<IntD>(m) + edge_offset[i];
                scatter(m_sec_edge_info, *mesh.m_sec_edge_info, idx);
            }
        }
        drjit::eval(m_sec_edge_info);
        FloatC edge_pmf = norm(detach<false>(m_sec_edge_info.e1));
        drjit::eval(edge_pmf);

        // std::cout << edge_pmf << std::endl;
        m_sec_edge_distrb->init(edge_pmf);
        if ( m_opts.log_level > 0 ) {
            std::stringstream oss;
            oss << edge_offset.back() << " secondary edges initialized.";
            log(oss.str().c_str());
        }
    } else {
        m_sec_edge_info = empty<SecondaryEdgeInfo>();
    }

    // Initialize OptiX

    m_optix->configure(m_meshes);

    // Cleanup
    for ( int i = 0; i < m_num_meshes; ++i ) {
        Mesh *mesh = m_meshes[i];

        if ( mesh->m_emitter == nullptr ) {
            // std::cout << "clean emitter" << std::endl;
            PSDR_ASSERT(mesh->m_triangle_info != nullptr);
            delete mesh->m_triangle_info;
            mesh->m_triangle_info = nullptr;

            if ( mesh->m_triangle_uv != nullptr ) {
                delete mesh->m_triangle_uv;
                mesh->m_triangle_uv = nullptr;
            }
        }
    }

    auto end_time = high_resolution_clock::now();
    if ( m_opts.log_level > 0 ) {
        std::stringstream oss;
        std::operator<<(oss, "Configured in "); oss << duration_cast<duration<double>>(end_time - start_time).count(); std::operator<<(oss, " seconds.");
        log(oss.str().c_str());
    }
    // PSDR_ASSERT(0);
}


bool Scene::is_ready() const {
    return (m_opts.spp   == 0 || m_samplers[0].is_ready()) &&
           (m_opts.sppe  == 0 || m_samplers[1].is_ready()) &&
           (m_opts.sppse == 0 || m_samplers[2].is_ready()) &&
           m_optix->is_ready();
}


template <bool ad, bool path_space>
Intersection<ad> Scene::ray_intersect(const Ray<ad> &ray, Mask<ad> active, TriangleInfoD *out_info) const {
    static_assert(ad || !path_space);
    Intersection<ad> its;

    Intersection_OptiX optix_its = m_optix->ray_intersect<ad>(ray, active);



    Vector2i<ad> idx(optix_its.triangle_id, optix_its.shape_id);
    // its.t = idx[1];
    // return its;

    TriangleUV<ad>      tri_uv_info;
    Mask<ad>            face_normal_mask;



    // TODO: FIX
    Vector3f<ad> tri_info_p0;
    Vector3f<ad> tri_info_e1; 
    Vector3f<ad> tri_info_e2; 
    Vector3f<ad> tri_info_n0; 
    Vector3f<ad> tri_info_n1; 
    Vector3f<ad> tri_info_n2; 
    Vector3f<ad> tri_info_face_normal; 
    Float<ad> tri_info_face_area; 
    Vector3i<ad> tri_info_face_indices;


    // std::cout << m_triangle_info.face_normal << std::endl;
    // std::cout << idx[1] << std::endl;

    // its.p = idx[1];
    // return its;

    if constexpr ( ad ) {
        its.n = gather<Vector3f<ad>>(m_triangle_info.face_normal, idx[1], active);
        tri_uv_info = gather<TriangleUVD>(m_triangle_uv, idx[1], active);
        face_normal_mask = gather<MaskD>(m_triangle_face_normals, idx[1], active);

        tri_info_p0 = gather<Vector3f<ad>>(m_triangle_info.p0, idx[1], active);
        tri_info_e1 = gather<Vector3f<ad>>(m_triangle_info.e1, idx[1], active);
        tri_info_e2 = gather<Vector3f<ad>>(m_triangle_info.e2, idx[1], active);
        tri_info_n0 = gather<Vector3f<ad>>(m_triangle_info.n0, idx[1], active);
        tri_info_n1 = gather<Vector3f<ad>>(m_triangle_info.n1, idx[1], active);
        tri_info_n2 = gather<Vector3f<ad>>(m_triangle_info.n2, idx[1], active);

        tri_info_face_normal = gather<Vector3f<ad>>(m_triangle_info.face_normal, idx[1], active);
        tri_info_face_area = gather<Float<ad>>(m_triangle_info.face_area, idx[1], active);

        tri_info_face_indices = gather<Vector3i<ad>>(m_triangle_info.face_indices, idx[1], active);

        TriangleInfo<ad>    tri_info = empty<TriangleInfo<ad>>(slices<Vector3f<ad>>(tri_info_p0));
        tri_info.p0 = tri_info_p0;
        tri_info.e1 = tri_info_e1;
        tri_info.e2 = tri_info_e2;
        tri_info.n0 = tri_info_n0;
        tri_info.n1 = tri_info_n1;
        tri_info.n2 = tri_info_n2;
        tri_info.face_normal = tri_info_face_normal;
        tri_info.face_area = tri_info_face_area;
        tri_info.face_indices = tri_info_face_indices;


        if ( out_info != nullptr ) *out_info = tri_info;

        if constexpr ( path_space ) {
            its.J = tri_info.face_area/detach<false>(tri_info.face_area);
        } else {
            its.J = 1.f;
        }

    } else {
        its.n = gather<Vector3f<ad>>(detach<false>(m_triangle_info.face_normal), idx[1], active);
        tri_uv_info = gather<TriangleUVC>(detach<false>(m_triangle_uv), idx[1], active);
        face_normal_mask = gather<MaskC>(detach<false>(m_triangle_face_normals), idx[1], active);

        tri_info_p0 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.p0), idx[1], active);
        tri_info_e1 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.e1), idx[1], active);
        tri_info_e2 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.e2), idx[1], active);
        tri_info_n0 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.n0), idx[1], active);
        tri_info_n1 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.n1), idx[1], active);
        tri_info_n2 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.n2), idx[1], active);
        tri_info_face_normal = gather<Vector3f<ad>>(detach<false>(m_triangle_info.face_normal), idx[1], active);
        tri_info_face_area = gather<Float<ad>>(detach<false>(m_triangle_info.face_area), idx[1], active);

        tri_info_face_indices = gather<Vector3i<ad>>(detach<false>(m_triangle_info.face_indices), idx[1], active);

        TriangleInfoD   tri_info = empty<TriangleInfoD>(slices<Vector3fC>(tri_info_p0));
        IntD idx_d(idx[1]);
        MaskD active_d(active);
        tri_info.p0 = gather<Vector3fD>((m_triangle_info.p0), idx_d, active_d);
        tri_info.e1 = gather<Vector3fD>((m_triangle_info.e1), idx_d, active_d);
        tri_info.e2 = gather<Vector3fD>((m_triangle_info.e2), idx_d, active_d);
        tri_info.n0 = gather<Vector3fD>((m_triangle_info.n0), idx_d, active_d);
        tri_info.n1 = gather<Vector3fD>((m_triangle_info.n1), idx_d, active_d);
        tri_info.n2 = gather<Vector3fD>((m_triangle_info.n2), idx_d, active_d);
        tri_info.face_normal = gather<Vector3fD>(m_triangle_info.face_normal, idx_d, active_d);
        tri_info.face_area = gather<FloatD>(m_triangle_info.face_area, idx_d, active_d);
        tri_info.face_indices = gather<Vector3iD>(m_triangle_info.face_indices, idx_d, active_d);


        if ( out_info != nullptr ) {
            *out_info = tri_info;
        }

        its.J = 1.f;

    }

    
    const Vector3f<ad> &vertex0 = tri_info_p0, &edge1 = tri_info_e1, &edge2 = tri_info_e2;

    // calculate dudp
    Vector3f<ad> dp0 = edge1,
                 dp1 = edge2;
    Vector2f<ad> uv0 = tri_uv_info[0],
                 uv1 = tri_uv_info[1],
                 uv2 = tri_uv_info[2];
    Vector2f<ad> duv0 = uv1 - uv0,
                 duv1 = uv2 - uv0;
    Float<ad> det     = fmsub(duv0.x(), duv1.y(), duv0.y() * duv1.x()),
              inv_det = rcp(det);
    Mask<ad>  valid_dp = neq(det, 0.f);

    if constexpr ( !ad || path_space ) {
        // Path-space formulation
        Vector2fC uv = optix_its.uv;
        Vector3f<ad> sh_n = normalize(bilinear<ad>(tri_info_n0,
                                                   tri_info_n1 - tri_info_n0,
                                                   tri_info_n2 - tri_info_n0,
                                                   uv));
        masked(sh_n, face_normal_mask) = its.n;

        if constexpr ( ad ) {
            its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        } else {
            its.shape = gather<MeshArrayC>(detach<false>(m_meshes_cuda), idx[0], active);
        }
        its.p = bilinear<ad>(vertex0, edge1, edge2, uv);

        Vector3f<ad> dir = its.p - ray.o;
        its.t = norm(dir);
        dir /= its.t;
        its.uv = bilinear2<ad>(tri_uv_info[0],
                               tri_uv_info[1] - tri_uv_info[0],
                               tri_uv_info[2] - tri_uv_info[0],
                               uv);

        its.dp_du = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_dv = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_du[valid_dp] = fmsub( duv1.y(), dp0, duv0.y() * dp1) * inv_det;
        its.dp_dv[valid_dp] = fnmadd(duv1.x(), dp0, duv0.x() * dp1) * inv_det;
        its.sh_frame = Frame<ad>(sh_n);
        its.sh_frame.s[valid_dp] = normalize(fnmadd(its.sh_frame.n, dot(its.sh_frame.n, its.dp_du), its.dp_du));
        its.sh_frame.t[valid_dp] = cross(its.sh_frame.n, its.sh_frame.s);
        its.wi = its.sh_frame.to_local(-dir);

        its.bc = uv;
        its.face_indices = tri_info_face_indices;

    } else {
        // Standard (solid-angle) formulation
        auto [uv, t] = ray_intersect_triangle<true>(vertex0, edge1, edge2, ray);
        Vector3fD sh_n = normalize(bilinear<true>(tri_info_n0,
                                                  tri_info_n1 - tri_info_n0,
                                                  tri_info_n2 - tri_info_n0,
                                                  uv));

        masked(sh_n, face_normal_mask) = its.n;

        its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        its.p = ray(t);        
        its.t = t;
        its.uv = bilinear2<true>(tri_uv_info[0],
                                 tri_uv_info[1] - tri_uv_info[0],
                                 tri_uv_info[2] - tri_uv_info[0],
                                 uv);
        // calculate dudp
        its.dp_du = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_dv = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_du[valid_dp] = fmsub( duv1.y(), dp0, duv0.y() * dp1) * inv_det;
        its.dp_dv[valid_dp] = fnmadd(duv1.x(), dp0, duv0.x() * dp1) * inv_det;
        its.sh_frame = Frame<ad>(sh_n);
        its.sh_frame.s[valid_dp] = normalize(fnmadd(its.sh_frame.n, dot(its.sh_frame.n, its.dp_du), its.dp_du));
        its.sh_frame.t[valid_dp] = cross(its.sh_frame.n, its.sh_frame.s);
        its.wi = its.sh_frame.to_local(-ray.d);
        
        its.bc = uv;
        its.face_indices = tri_info_face_indices;
    }


    // drjit::eval(its);
    return its;
}


template <bool ad, bool path_space>
Intersection<ad> Scene::unit_ray_intersect(const Ray<ad> &ray, Mask<ad> active) const {
    static_assert(ad || !path_space);
    Intersection<ad> its;

    Intersection_OptiX optix_its = m_optix->ray_intersect<ad>(ray, active);


    Vector2i<ad> idx(optix_its.triangle_id, optix_its.shape_id);

    TriangleUV<ad>      tri_uv_info;
    Mask<ad>            face_normal_mask;



    // TODO: FIX
    Vector3f<ad> tri_info_p0;
    Vector3f<ad> tri_info_e1; 
    Vector3f<ad> tri_info_e2; 
    Vector3f<ad> tri_info_n0; 
    Vector3f<ad> tri_info_n1; 
    Vector3f<ad> tri_info_n2; 
    Vector3f<ad> tri_info_face_normal; 
    Float<ad> tri_info_face_area; 
    Vector3i<ad> tri_info_face_indices; 

    if constexpr ( ad ) {
        its.n = gather<Vector3f<ad>>(m_triangle_info.face_normal, idx[1], active);
        tri_uv_info = gather<TriangleUVD>(m_triangle_uv, idx[1], active);
        face_normal_mask = gather<MaskD>(m_triangle_face_normals, idx[1], active);

        tri_info_p0 = gather<Vector3f<ad>>(m_triangle_info.p0, idx[1], active);
        tri_info_e1 = gather<Vector3f<ad>>(m_triangle_info.e1, idx[1], active);
        tri_info_e2 = gather<Vector3f<ad>>(m_triangle_info.e2, idx[1], active);
        tri_info_n0 = gather<Vector3f<ad>>(m_triangle_info.n0, idx[1], active);
        tri_info_n1 = gather<Vector3f<ad>>(m_triangle_info.n1, idx[1], active);
        tri_info_n2 = gather<Vector3f<ad>>(m_triangle_info.n2, idx[1], active);

        tri_info_face_normal = gather<Vector3f<ad>>(m_triangle_info.face_normal, idx[1], active);
        tri_info_face_area = gather<Float<ad>>(m_triangle_info.face_area, idx[1], active);
        tri_info_face_indices = gather<Vector3i<ad>>(m_triangle_info.face_indices, idx[1], active);

        TriangleInfo<ad>    tri_info = empty<TriangleInfo<ad>>(slices<Vector3f<ad>>(tri_info_p0));
        tri_info.p0 = tri_info_p0;
        tri_info.e1 = tri_info_e1;
        tri_info.e2 = tri_info_e2;
        tri_info.n0 = tri_info_n0;
        tri_info.n1 = tri_info_n1;
        tri_info.n2 = tri_info_n2;
        tri_info.face_normal = tri_info_face_normal;
        tri_info.face_area = tri_info_face_area;
        tri_info.face_indices = tri_info_face_indices;

        if constexpr ( path_space ) {
            its.J = tri_info.face_area/detach<false>(tri_info.face_area);
        } else {
            its.J = 1.f;
        }

    } else {
        its.n = gather<Vector3f<ad>>(detach<false>(m_triangle_info.face_normal), idx[1], active);
        tri_uv_info = gather<TriangleUVC>(detach<false>(m_triangle_uv), idx[1], active);
        face_normal_mask = gather<MaskC>(detach<false>(m_triangle_face_normals), idx[1], active);

        tri_info_p0 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.p0), idx[1], active);
        tri_info_e1 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.e1), idx[1], active);
        tri_info_e2 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.e2), idx[1], active);
        tri_info_n0 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.n0), idx[1], active);
        tri_info_n1 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.n1), idx[1], active);
        tri_info_n2 = gather<Vector3f<ad>>(detach<false>(m_triangle_info.n2), idx[1], active);
        tri_info_face_normal = gather<Vector3f<ad>>(detach<false>(m_triangle_info.face_normal), idx[1], active);
        tri_info_face_area = gather<Float<ad>>(detach<false>(m_triangle_info.face_area), idx[1], active);
        tri_info_face_indices = gather<Vector3i<ad>>(detach<false>(m_triangle_info.face_indices), idx[1], active);


        TriangleInfoD   tri_info = empty<TriangleInfoD>(slices<Vector3fC>(tri_info_p0));
        IntD idx_d2(idx[1]);
        MaskD active_d2(active);
        tri_info.p0 = gather<Vector3fD>((m_triangle_info.p0), idx_d2, active_d2);
        tri_info.e1 = gather<Vector3fD>((m_triangle_info.e1), idx_d2, active_d2);
        tri_info.e2 = gather<Vector3fD>((m_triangle_info.e2), idx_d2, active_d2);
        tri_info.n0 = gather<Vector3fD>((m_triangle_info.n0), idx_d2, active_d2);
        tri_info.n1 = gather<Vector3fD>((m_triangle_info.n1), idx_d2, active_d2);
        tri_info.n2 = gather<Vector3fD>((m_triangle_info.n2), idx_d2, active_d2);
        tri_info.face_normal = gather<Vector3fD>(m_triangle_info.face_normal, idx_d2, active_d2);
        tri_info.face_area = gather<FloatD>(m_triangle_info.face_area, idx_d2, active_d2);
        tri_info.face_indices = gather<Vector3iD>(m_triangle_info.face_indices, idx_d2, active_d2);

        its.J = 1.f;

    }

    
    const Vector3f<ad> &vertex0 = tri_info_p0, &edge1 = tri_info_e1, &edge2 = tri_info_e2;

    // calculate dudp
    Vector3f<ad> dp0 = edge1,
                 dp1 = edge2;
    Vector2f<ad> uv0 = tri_uv_info[0],
                 uv1 = tri_uv_info[1],
                 uv2 = tri_uv_info[2];
    Vector2f<ad> duv0 = uv1 - uv0,
                 duv1 = uv2 - uv0;
    Float<ad> det     = fmsub(duv0.x(), duv1.y(), duv0.y() * duv1.x()),
              inv_det = rcp(det);
    Mask<ad>  valid_dp = neq(det, 0.f);

    if constexpr ( !ad || path_space ) {
        // Path-space formulation
        Vector2fC uv = optix_its.uv;
        Vector3f<ad> sh_n = normalize(bilinear<ad>(tri_info_n0,
                                                   tri_info_n1 - tri_info_n0,
                                                   tri_info_n2 - tri_info_n0,
                                                   uv));
        masked(sh_n, face_normal_mask) = its.n;

        if constexpr ( ad ) {
            its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        } else {
            its.shape = gather<MeshArrayC>(detach<false>(m_meshes_cuda), idx[0], active);
        }
        its.p = bilinear<ad>(vertex0, edge1, edge2, uv);

        Vector3f<ad> dir = its.p - ray.o;
        its.t = norm(dir);
        dir /= its.t;
        its.uv = bilinear2<ad>(tri_uv_info[0],
                               tri_uv_info[1] - tri_uv_info[0],
                               tri_uv_info[2] - tri_uv_info[0],
                               uv);

        its.dp_du = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_dv = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_du[valid_dp] = fmsub( duv1.y(), dp0, duv0.y() * dp1) * inv_det;
        its.dp_dv[valid_dp] = fnmadd(duv1.x(), dp0, duv0.x() * dp1) * inv_det;
        its.sh_frame = Frame<ad>(sh_n);
        its.sh_frame.s[valid_dp] = normalize(fnmadd(its.sh_frame.n, dot(its.sh_frame.n, its.dp_du), its.dp_du));
        its.sh_frame.t[valid_dp] = cross(its.sh_frame.n, its.sh_frame.s);
        its.wi = its.sh_frame.to_local(-dir);

        its.bc = uv;
        its.face_indices = tri_info_face_indices;

    } else {
        // Standard (solid-angle) formulation
        auto [uv, t] = ray_intersect_triangle<true>(vertex0, edge1, edge2, ray);
        Vector3fD sh_n = normalize(bilinear<true>(tri_info_n0,
                                                  tri_info_n1 - tri_info_n0,
                                                  tri_info_n2 - tri_info_n0,
                                                  uv));

        masked(sh_n, face_normal_mask) = its.n;

        its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        its.p = ray(t);
        its.t = t;
        its.uv = bilinear2<true>(tri_uv_info[0],
                                 tri_uv_info[1] - tri_uv_info[0],
                                 tri_uv_info[2] - tri_uv_info[0],
                                 uv);
        // calculate dudp
        its.dp_du = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_dv = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_du[valid_dp] = fmsub( duv1.y(), dp0, duv0.y() * dp1) * inv_det;
        its.dp_dv[valid_dp] = fnmadd(duv1.x(), dp0, duv0.x() * dp1) * inv_det;
        its.sh_frame = Frame<ad>(sh_n);
        its.sh_frame.s[valid_dp] = normalize(fnmadd(its.sh_frame.n, dot(its.sh_frame.n, its.dp_du), its.dp_du));
        its.sh_frame.t[valid_dp] = cross(its.sh_frame.n, its.sh_frame.s);
        its.wi = its.sh_frame.to_local(-ray.d);

        its.bc = uv;
        its.face_indices = tri_info_face_indices;
    }


    // drjit::eval(its);
    return its;
}


template <bool ad>
PositionSample<ad> Scene::sample_emitter_position(const Vector3f<ad> &ref_p, const Vector2f<ad> &_sample2, Mask<ad> active) const {
    PSDR_ASSERT_MSG(!m_emitters.empty(), "No Emitter!");

    if ( m_emitters.size() == 1U ) {
        return m_emitters[0]->sample_position(ref_p, _sample2, active);
    } else {
        PositionSample<ad> result;
        Vector2f<ad> sample2 = _sample2;
        auto [emitter_index, emitter_pdf] = m_emitters_distrb->sample_reuse<ad>(sample2.y());

        EmitterArray<ad> emitter_arr;
        if constexpr ( ad ) {
            emitter_arr = gather<EmitterArrayD>(m_emitters_cuda, IntD(emitter_index), active);
            result = emitter_arr->sample_positionD(ref_p, sample2, active);
        } else {
            emitter_arr = gather<EmitterArrayC>(detach<false>(m_emitters_cuda), emitter_index, active);
            result = emitter_arr->sample_positionC(ref_p, sample2, active);
        }
        result.pdf *= emitter_pdf;

        return result;
    }

    
    
}


template <bool ad>
Float<ad> Scene::emitter_position_pdf(const Vector3f<ad> &ref_p, const Intersection<ad> &its, Mask<ad> active) const {
    if constexpr (ad) {
        return its.shape->emitter()->sample_position_pdfD(ref_p, its, active);
    } else {
        return its.shape->emitter()->sample_position_pdfC(ref_p, its, active);
    }
    
}


BoundarySegSampleDirect Scene::sample_boundary_segment_direct(const Vector3fC &sample3, MaskC active) const {
    BoundarySegSampleDirect result;
    // Sample a point p0 on a face edge
    FloatC sample1 = sample3.x();
    auto [edge_idx, pdf0] = m_sec_edge_distrb->sample_reuse<false>(sample1);

    // SecondaryEdgeInfo info = gather<SecondaryEdgeInfo>(m_sec_edge_info, IntD(edge_idx), active);

    Vector3fD info_e1 = gather<Vector3fD>(m_sec_edge_info.e1, IntD(edge_idx), active);
    Vector3fD info_p0 = gather<Vector3fD>(m_sec_edge_info.p0, IntD(edge_idx), active);
    Vector3fD info_p2 = gather<Vector3fD>(m_sec_edge_info.p2, IntD(edge_idx), active);
    Vector3fD info_n0 = gather<Vector3fD>(m_sec_edge_info.n0, IntD(edge_idx), active);
    Vector3fD info_n1 = gather<Vector3fD>(m_sec_edge_info.n1, IntD(edge_idx), active);
    MaskD info_is_boundary = gather<MaskD>(m_sec_edge_info.is_boundary, IntD(edge_idx), active);


    result.p0 = fmadd(info_e1, sample1, info_p0); // p0 = info.p0 + info.e1*sample1;
    result.edge = normalize(detach<false>(info_e1));
    result.edge2 = detach<false>(info_p2) - detach<false>(info_p0);
    const Vector3fC &p0 = detach<false>(result.p0);
    pdf0 /= norm(detach<false>(info_e1));

    // Sample a point ps2 on a emitter
    PositionSampleC ps2 = sample_emitter_position<false>(p0, tail<2>(sample3), active);
    result.p2 = ps2.p;
    result.n = ps2.n;

    // Construct the edge "ray" and check if it is valid
    Vector3fC e = result.p2 - p0;
    const FloatC distSqr = squared_norm(e);
    e /= safe_sqrt(distSqr);
    const FloatC cosTheta = dot(result.n, -e);

    IntC sgn0 = sign<false>(dot(detach<false>(info_n0), e), EdgeEpsilon),
         sgn1 = sign<false>(dot(detach<false>(info_n1), e), EdgeEpsilon);
    result.is_valid = active && (cosTheta > Epsilon) && (
        (detach<false>(info_is_boundary) && neq(sgn0, 0)) || (~detach<false>(info_is_boundary) && (sgn0*sgn1 < 0))
    );

    result.pdf = (pdf0*ps2.pdf*(distSqr/cosTheta)) & result.is_valid;
    return result;
}


// Explicit instantiations
template IntersectionC Scene::ray_intersect<false, false>(const RayC&, MaskC, TriangleInfoD*) const;
template IntersectionD Scene::ray_intersect<true , false>(const RayD&, MaskD, TriangleInfoD*) const;
template IntersectionD Scene::ray_intersect<true , true >(const RayD&, MaskD, TriangleInfoD*) const;

template IntersectionC Scene::unit_ray_intersect<false, false>(const RayC&, MaskC) const;
template IntersectionD Scene::unit_ray_intersect<true , false>(const RayD&, MaskD) const;
template IntersectionD Scene::unit_ray_intersect<true , true >(const RayD&, MaskD) const;

template PositionSampleC Scene::sample_emitter_position<false>(const Vector3fC&, const Vector2fC&, MaskC) const;
template PositionSampleD Scene::sample_emitter_position<true >(const Vector3fD&, const Vector2fD&, MaskD) const;

template FloatC Scene::emitter_position_pdf<false>(const Vector3fC&, const IntersectionC&, MaskC) const;
template FloatD Scene::emitter_position_pdf<true >(const Vector3fD&, const IntersectionD&, MaskD) const;

NAMESPACE_END(psdr_jit)
