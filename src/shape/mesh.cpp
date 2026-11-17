#include <vector>
#include <array>
#include <map>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

#include <misc/Exception.h>
#include <psdr/core/transform.h>
#include <psdr/core/pmf.h>
#include <psdr/core/warp.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/emitter.h>
#include <psdr/shape/mesh.h>

#include <chrono>
#include <numeric>

#include <drjit/while_loop.h>

NAMESPACE_BEGIN(psdr_jit)

template <bool ad>
static std::pair<TriangleInfo<ad>, Vector3f<ad>> process_mesh(const Vector3f<ad> &vertex_positions, const Vector3i<ad> &face_indices) {
    const int num_vertices = static_cast<int>(slices<Vector3f<ad>>(vertex_positions));

    TriangleInfo<ad> triangles;
    triangles.face_indices = face_indices; 
    triangles.p0 = gather<Vector3f<ad>>(vertex_positions, face_indices[0]);
    triangles.e1 = gather<Vector3f<ad>>(vertex_positions, face_indices[1]) - triangles.p0;
    triangles.e2 = gather<Vector3f<ad>>(vertex_positions, face_indices[2]) - triangles.p0;

    Vector3f<ad> &face_normals = triangles.face_normal;
    Float<ad>    &face_areas = triangles.face_area;

    face_normals = cross(triangles.e1, triangles.e2);
    face_areas   = norm(face_normals);

    Vector3f<ad> vertex_normals = zeros<Vector3f<ad>>(num_vertices);
    Float<ad>    vertex_weights = zeros<Float<ad>>(num_vertices);
    for ( int i = 0; i < 3; ++i ) {
        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, vertex_normals[j], face_normals[j], face_indices[i]);
        }
        scatter_reduce(ReduceOp::Add, vertex_weights, face_areas, face_indices[i]);
    }

    vertex_normals = normalize(vertex_normals/vertex_weights);

    triangles.n0 = gather<Vector3f<ad>>(vertex_normals, face_indices[0]);
    triangles.n1 = gather<Vector3f<ad>>(vertex_normals, face_indices[1]);
    triangles.n2 = gather<Vector3f<ad>>(vertex_normals, face_indices[2]);

    // Normalize the face normals
    face_normals /= face_areas;
    face_areas *= 0.5f;

    drjit::eval(triangles, vertex_normals);

    
    return { triangles, vertex_normals };
}



Mesh::Mesh(const Mesh &other) : Object(other) {
    // Copy authoring data, but rebuild the owned sampling/geometry caches.
    // Sharing these raw pointers caused double frees after add_Mesh().
    m_use_face_normals = other.m_use_face_normals;
    m_enable_edges = other.m_enable_edges;
    m_edge_sort = other.m_edge_sort;
    m_to_world_raw = other.m_to_world_raw;
    m_to_world_left = other.m_to_world_left;
    m_to_world_right = other.m_to_world_right;
    m_bsdf = other.m_bsdf;
    m_num_vertices = other.m_num_vertices;
    m_num_faces = other.m_num_faces;
    m_has_uv = other.m_has_uv;
    m_vertex_positions_raw = other.m_vertex_positions_raw;
    m_vertex_normals_raw = other.m_vertex_normals_raw;
    m_vertex_uv = other.m_vertex_uv;
    m_face_indices = other.m_face_indices;
    m_face_uv_indices = other.m_face_uv_indices;
    m_edge_indices = other.m_edge_indices;
    m_valid_edge_indices = other.m_valid_edge_indices;
    m_cut_position = other.m_cut_position;
}

Mesh::~Mesh() {
    if ( m_face_distrb ) delete m_face_distrb;
    if ( m_triangle_info ) delete m_triangle_info;
    if ( m_triangle_uv ) delete m_triangle_uv;
    if ( m_sec_edge_info ) delete m_sec_edge_info;
}


void Mesh::load_raw(const Vector3fC &new_vertex_positions, const Vector3iC &new_face_indices, 
                   const Vector2fC &new_vertex_uv, const Vector3iC &new_face_uv_indices, bool verbose) {
    using namespace std::chrono;
    
    m_num_vertices = slices(new_vertex_positions);
    // Loading vertex positions
    m_vertex_positions_raw = Vector3fD(new_vertex_positions);
    
    // Loading vertex uv coordinates if provided
    m_has_uv = slices(new_vertex_uv) > 0;
    if (m_has_uv) {
        m_vertex_uv = Vector2fD(new_vertex_uv);
        m_face_uv_indices = Vector3iD(new_face_uv_indices);
    } else {
        m_vertex_uv = Vector2fD();
        m_face_uv_indices = Vector3iD();
    }

    m_num_faces = slices(new_face_indices);
    // Loading face indices
    m_face_indices = Vector3iD(new_face_indices);

    std::array<std::vector<int>, 3> face_indices_cpu;
    copy_cuda_array(new_face_indices, face_indices_cpu);
    drjit::eval(); drjit::sync_thread();

    auto start_time = high_resolution_clock::now();

    // Constructing edge list
    int m_num_edges = 0;
    if ( m_enable_edges ) {
        std::vector<int> buffers[5];
        buffers[0].reserve(3*m_num_faces);
        buffers[1].reserve(3*m_num_faces);
        buffers[2].reserve(3*m_num_faces);
        buffers[3].reserve(3*m_num_faces);
        buffers[4].reserve(3*m_num_faces);

        std::map<std::pair<int, int>, std::vector<int>> edge_map;
        for ( size_t f = 0; f < m_num_faces; ++f ) {
            for ( int i = 0; i < 3; ++i ) {
                int k1 = i, k2 = (i + 1) % 3, k3 = (i + 2) % 3;
                int idx1 = face_indices_cpu[k1][f];
                int idx2 = face_indices_cpu[k2][f];
                int idx3 = face_indices_cpu[k3][f];
                auto key = idx1 < idx2 ?
                    std::make_pair(idx1, idx2) : std::make_pair(idx2, idx1);
                if (edge_map.find(key) == edge_map.end()) {
                    auto it = edge_map.insert(edge_map.end(), { key, std::vector<int>() });
                    it->second.push_back(idx3);
                }
                edge_map[key].push_back(static_cast<int>(f));
            }
        }

        for ( auto it: edge_map ) {
                buffers[0].push_back(it.first.first);
                buffers[1].push_back(it.first.second);
                if ( it.second.size() >= 3 ) {
                    buffers[2].push_back(it.second[1]);
                    buffers[3].push_back(it.second[2]);
                    buffers[4].push_back(it.second[0]);
                    ++m_num_edges;
                } else {
                    buffers[2].push_back(it.second[1]);
                    buffers[3].push_back(-1);
                    buffers[4].push_back(it.second[0]);
                    ++m_num_edges;
                }
        }

        m_edge_indices = Vectori<5, true>(drjit::load<IntD>(buffers[0].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[1].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[2].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[3].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[4].data(), m_num_edges));
    }

    auto end_time = high_resolution_clock::now();
    double seconds = duration_cast<duration<double>>(end_time - start_time).count();

    if ( verbose ) {
        std::cout << "Loaded " << m_num_vertices << " vertices, "
                               << m_num_faces    << " faces, "
                               << m_num_edges    << " edges in " << seconds << " seconds. " << std::endl;
    }
    drjit::eval(); drjit::sync_thread();
    m_ready = false;
}


void Mesh::load(const char *fname, bool verbose) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    PSDR_ASSERT_MSG(
        tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, fname),
        std::string("Failed to load OBJ from: ") + fname
    );

    assert(attrib.vertices.size() % 3 == 0);
    m_num_vertices = static_cast<int>(attrib.vertices.size())/3;

    // Loading vertex positions
    {
        std::vector<float> buffers[3];
        buffers[0].resize(m_num_vertices);
        buffers[1].resize(m_num_vertices);
        buffers[2].resize(m_num_vertices);

        for ( int i = 0; i < m_num_vertices; ++i )
            for ( int j = 0; j < 3; ++j )
                buffers[j][i] = attrib.vertices[3*i + j];

        m_vertex_positions_raw = Vector3fD(drjit::load<FloatD>(buffers[0].data(), m_num_vertices),
                                           drjit::load<FloatD>(buffers[1].data(), m_num_vertices),
                                           drjit::load<FloatD>(buffers[2].data(), m_num_vertices));
    }

    // Loading vertex uv coordinates
    if ( (m_has_uv = !attrib.texcoords.empty()) ) {
        int n = static_cast<int>(attrib.texcoords.size())/2;

        std::vector<float> buffers[2];
        buffers[0].resize(n);
        buffers[1].resize(n);

        for ( int i = 0; i < n; ++i )
            for ( int j = 0; j < 2; ++j )
                buffers[j][i] = attrib.texcoords[2*i + j];

        m_vertex_uv = Vector2fD(drjit::load<FloatD>(buffers[0].data(), n),
                                drjit::load<FloatD>(buffers[1].data(), n));
    }

    m_num_faces = 0;
    for ( size_t s = 0; s < shapes.size(); s++ )
        m_num_faces += static_cast<int>(shapes[s].mesh.num_face_vertices.size());

    std::vector<int> v[6];
    for ( int i = 0; i < 6; ++i ) v[i].reserve(m_num_faces);

    // Loading face indices
    for ( size_t s = 0; s < shapes.size(); ++s ) {
        for ( size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f ) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            assert(fv == 3);
            for ( int i = 0; i < fv; ++i ) {
                auto idx = shapes[s].mesh.indices[3*f + i];
                v[i].push_back(idx.vertex_index);
                if ( m_has_uv )
                    v[3 + i].push_back(idx.texcoord_index);
            }
        }
    }
    assert(static_cast<int>(v[0].size()) == m_num_faces);

    m_face_indices = Vector3iD(drjit::load<IntD>(v[0].data(), m_num_faces),
                               drjit::load<IntD>(v[1].data(), m_num_faces),
                               drjit::load<IntD>(v[2].data(), m_num_faces));


    if ( m_has_uv ) {
        m_face_uv_indices = Vector3iD(drjit::load<IntD>(v[3].data(), m_num_faces),
                                      drjit::load<IntD>(v[4].data(), m_num_faces),
                                      drjit::load<IntD>(v[5].data(), m_num_faces));
    }

    // Constructing edge list

    int m_num_edges = 0;
    if ( m_enable_edges ) {
        std::vector<int> buffers[5];
        buffers[0].reserve(3*m_num_faces);
        buffers[1].reserve(3*m_num_faces);
        buffers[2].reserve(3*m_num_faces);
        buffers[3].reserve(3*m_num_faces);
        buffers[4].reserve(3*m_num_faces);

        std::map<std::pair<int, int>, std::vector<int>> edge_map;
        for ( size_t s = 0; s < shapes.size(); ++s ) {
            for ( size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f ) {
                int fv = shapes[s].mesh.num_face_vertices[f];
                assert(fv == 3);
                for ( int i = 0; i < fv; ++i ) {
                    int k1 = i, k2 = (i + 1) % 3, k3 = (i + 2) % 3;
                    int idx1 = shapes[s].mesh.indices[3*f + k1].vertex_index;
                    int idx2 = shapes[s].mesh.indices[3*f + k2].vertex_index;
                    int idx3 = shapes[s].mesh.indices[3*f + k3].vertex_index;
                    auto key = idx1 < idx2 ?
                        std::make_pair(idx1, idx2) : std::make_pair(idx2, idx1);
                    if (edge_map.find(key) == edge_map.end()) {
                        auto it = edge_map.insert(edge_map.end(), { key, std::vector<int>() });
                        it->second.push_back(idx3);
                    }
                    edge_map[key].push_back(static_cast<int>(f));
                }
            }
        }

        for ( auto it: edge_map ) {
            // if ( it.second.size() > 3 ) std::cout << "WARN: Edge shared by more than 2 faces" << std::endl;
            // if ( it.second.size() > 3 ) {
            //     // Non-manifold mesh is not aled
            //     PSDR_ASSERT_MSG(false, std::string("Edge shared by more than 2 faces: ") + fname);
            // } else {
                buffers[0].push_back(it.first.first);
                buffers[1].push_back(it.first.second);
                if ( it.second.size() >= 3 ) {
                    // PSDR_ASSERT_MSG(it.second[1] != it.second[2], std::string("Duplicated faces: ") + fname);
                    buffers[2].push_back(it.second[1]);
                    buffers[3].push_back(it.second[2]);
                    buffers[4].push_back(it.second[0]);
                    ++m_num_edges;
                } else {
                    // PSDR_ASSERT_MSG(it.second.size() == 2, std::string("Edge should be boundary: ") + fname);
                    buffers[2].push_back(it.second[1]);
                    buffers[3].push_back(-1);
                    buffers[4].push_back(it.second[0]);
                    ++m_num_edges;
                }
            // }
        }

        m_edge_indices = Vectori<5, true>(drjit::load<IntD>(buffers[0].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[1].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[2].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[3].data(), m_num_edges),
                                          drjit::load<IntD>(buffers[4].data(), m_num_edges));
    }

    if ( verbose ) {
        std::cout << "Loaded " << m_num_vertices << " vertices, "
                               << m_num_faces    << " faces, "
                               << m_num_edges    << " edges. " << std::endl;
    }
    drjit::eval(); drjit::sync_thread();
    m_ready = false;
}


void Mesh::configure() {
    if ( m_bsdf != nullptr ) {
        PSDR_ASSERT(!m_bsdf->anisotropic() || !m_use_face_normals);
    }

    // Calculating the "raw" (i.e., object-space) vertex normals
    std::tie(std::ignore, m_vertex_normals_raw) = process_mesh<true>(m_vertex_positions_raw, m_face_indices);

    Matrix4fD to_world = m_to_world_left * m_to_world_raw * m_to_world_right;

    // Calculating the world-space vertex positions

    m_vertex_positions = transform_pos(to_world, m_vertex_positions_raw);

    delete m_triangle_info;
    m_triangle_info = nullptr;

    delete m_triangle_uv;
    m_triangle_uv = nullptr;

    m_triangle_info = new TriangleInfoD();
    TriangleInfoD &triangle_info = *m_triangle_info;
    std::tie(triangle_info, std::ignore) = process_mesh<true>(m_vertex_positions, m_face_indices);

    const FloatD &face_areas = triangle_info.face_area;
    m_total_area = sum(face_areas)[0];
    m_inv_total_area = 1.f/m_total_area;

    if ( m_has_uv ) {
        m_triangle_uv = new TriangleUVD();
        for ( int i = 0; i < 3; ++i )
            (*m_triangle_uv)[i] = gather<Vector2fD>(m_vertex_uv, m_face_uv_indices[i]);
    }

    if ( m_face_distrb == nullptr ) m_face_distrb = new DiscreteDistribution();
    m_face_distrb->init(detach<false>(face_areas));


    if ( m_enable_edges ) {
        if ( m_sec_edge_info == nullptr ) m_sec_edge_info = new SecondaryEdgeInfo();

        SecondaryEdgeInfo secEdgeInfo;
        secEdgeInfo.is_boundary = (m_edge_indices[3] < 0);
        secEdgeInfo.p0 = gather<Vector3fD>(m_vertex_positions, m_edge_indices[0]);
        secEdgeInfo.e1 = gather<Vector3fD>(m_vertex_positions, m_edge_indices[1]) - secEdgeInfo.p0;
        secEdgeInfo.n0 = gather<Vector3fD>(m_triangle_info->face_normal, m_edge_indices[2]);
        secEdgeInfo.n1 = gather<Vector3fD>(m_triangle_info->face_normal, m_edge_indices[3], ~secEdgeInfo.is_boundary);
        secEdgeInfo.p2 = gather<Vector3fD>(m_vertex_positions, m_edge_indices[4]);

        MaskD keep = (dot(secEdgeInfo.n0, secEdgeInfo.n1) < 1.f - EdgeEpsilon);
        // *m_sec_edge_info = compressD<SecondaryEdgeInfo>(secEdgeInfo, keep);
        // TODO: secondary edge compression
        *m_sec_edge_info = secEdgeInfo;
    } else {
        if ( m_sec_edge_info != nullptr ) {
            delete m_sec_edge_info;
            m_sec_edge_info = nullptr;
        }
    }

    m_ready = true;
    drjit::make_opaque(m_vertex_positions, m_vertex_normals_raw, m_to_world_left, m_to_world_raw, m_to_world_right, m_triangle_uv);

    prepare_optix_buffers();
    drjit::eval(m_vertex_buffer, m_face_buffer);
    drjit::sync_thread();
}


void Mesh::prepare_optix_buffers() {
    PSDR_ASSERT(m_ready);
    IntC idx;

    m_vertex_buffer = empty<FloatC>(m_num_vertices*3);
    idx = arange<IntC>(m_num_vertices)*3;
    for ( int i = 0; i < 3; ++i ) {
        scatter(m_vertex_buffer, detach<false>(m_vertex_positions[i]), idx + i);
    }

    m_face_buffer = empty<IntC>(m_num_faces*3);
    idx = arange<IntC>(m_num_faces)*3;
    for ( int i = 0; i < 3; ++i ) {
        scatter(m_face_buffer, detach<false>(m_face_indices[i]), idx + i);
    }
}


PositionSampleC Mesh::sample_position(const Vector2fC &sample2, MaskC active) const {
    return __sample_position<false>(sample2, active);
}


PositionSampleD Mesh::sample_position(const Vector2fD &sample2, MaskD active) const {
    return __sample_position<true>(sample2, active);
}


template <bool ad>
PositionSample<ad> Mesh::__sample_position(const Vector2f<ad> &_sample2, Mask<ad> active) const {
    PSDR_ASSERT(m_ready && m_emitter != nullptr);
    PSDR_ASSERT(m_triangle_info != nullptr);

    PositionSample<ad> result;
    Vector2f<ad> sample2 = _sample2;

    IntC idx;
    std::tie(idx, std::ignore) = m_face_distrb->sample_reuse<ad>(sample2.x());
    sample2 = warp::square_to_uniform_triangle<ad>(sample2);

    // TriangleInfo<ad> tri_info;
    if constexpr ( ad ) {
        TriangleInfoD tri_data = (*m_triangle_info);
        FloatD face_area = gather<FloatD>(tri_data.face_area, IntD(idx), active);
        Vector3fD p0 = gather<Vector3fD>(tri_data.p0, IntD(idx), active);
        Vector3fD e1 = gather<Vector3fD>(tri_data.e1, IntD(idx), active);
        Vector3fD e2 = gather<Vector3fD>(tri_data.e2, IntD(idx), active);
        Vector3fD face_normal = gather<Vector3fD>(tri_data.face_normal, IntD(idx), active);

        result.J = face_area/detach<false>(face_area);
        result.p = bilinear<ad>(p0, e1, e2, sample2);
        result.n = face_normal;

    } else {

        TriangleInfoC tri_data = detach<false>(*m_triangle_info);
        FloatC face_area = gather<FloatC>(tri_data.face_area, (idx), active);
        Vector3fC p0 = gather<Vector3fC>(tri_data.p0, (idx), active);
        Vector3fC e1 = gather<Vector3fC>(tri_data.e1, (idx), active);
        Vector3fC e2 = gather<Vector3fC>(tri_data.e2, (idx), active);
        Vector3fC face_normal = gather<Vector3fC>(tri_data.face_normal, (idx), active);

        result.J = 1.f;
        result.p = bilinear<ad>(p0, e1, e2, sample2);
        result.n = face_normal;
    }
    // Broadcast scalars to match the sample width
    result.pdf = Float<ad>(m_inv_total_area) & active;  // Creates array with proper width
    result.pdf = select(active, Float<ad>(m_inv_total_area), Float<ad>(0.f));
    result.is_valid = active;  // Use active mask as is_valid (already has correct width)
    return result;
}


FloatC Mesh::sample_position_pdf(const IntersectionC &its, MaskC active) const {
    active &= eq(its.shape, this);
    return FloatC(m_inv_total_area) & active;
}


FloatD Mesh::sample_position_pdfD(const IntersectionD &its, MaskD active) const {
    active &= eq(its.shape, this);
    return FloatD(m_inv_total_area) & active;
}


void Mesh::dump(const char *fname, bool raw) const {
    std::array<std::vector<float>, 3> vertex_positions, vertex_normals;
    {
        if (!raw) {
            const Vector3fC &vertex_positions_ = detach<false>(m_vertex_positions_raw);
            copy_cuda_array<float, 3>(vertex_positions_, vertex_positions);
            if ( !m_use_face_normals ) {
                Vector3fC vertex_normals_;
                std::tie(std::ignore, vertex_normals_) = process_mesh<false>(vertex_positions_, detach<false>(m_face_indices));
                copy_cuda_array<float, 3>(vertex_normals_, vertex_normals);
            }
        } else {
            const Vector3fC &vertex_positions_ = detach<false>(m_vertex_positions);
            copy_cuda_array<float, 3>(vertex_positions_, vertex_positions);
            if ( !m_use_face_normals ) {
                Vector3fC vertex_normals_;
                std::tie(std::ignore, vertex_normals_) = process_mesh<false>(vertex_positions_, detach<false>(m_face_indices));
                copy_cuda_array<float, 3>(vertex_normals_, vertex_normals);
            }
        }
        // PSDR_ASSERT(0);

    }

    FILE *fout = fopen(fname, "wt");
    for ( int i = 0; i < m_num_vertices; ++i ) {
        fprintf(fout, "v %.6e %.6e %.6e\n", vertex_positions[0][i], vertex_positions[1][i], vertex_positions[2][i]);
        if ( !m_use_face_normals ) {
            fprintf(fout, "vn %.6e %.6e %.6e\n", vertex_normals[0][i], vertex_normals[1][i], vertex_normals[2][i]);
        }
    }

    std::array<std::vector<int32_t>, 3> face_indices;
    copy_cuda_array<int32_t, 3>(detach<false>(m_face_indices), face_indices);
    if ( m_has_uv ) {
        std::array<std::vector<float>, 2> vertex_uv;
        std::array<std::vector<int32_t>, 3> face_uv_indices;
        copy_cuda_array<float, 2>(detach<false>(m_vertex_uv), vertex_uv);
        copy_cuda_array<int32_t, 3>(detach<false>(m_face_uv_indices), face_uv_indices);


        int m = static_cast<int>(slices<Vector2fD>(m_vertex_uv));
        for ( int i = 0; i < m; ++i )
            fprintf(fout, "vt %.6le %.6le\n", vertex_uv[0][i], vertex_uv[1][i]);

        for ( int i = 0; i < m_num_faces; ++i ) {
            int v0 = face_indices[0][i] + 1, v1 = face_indices[1][i] + 1, v2 = face_indices[2][i] + 1;
            if ( m_use_face_normals ) {
                fprintf(fout, "f %d/%d %d/%d %d/%d\n",
                        v0, face_uv_indices[0][i] + 1,
                        v1, face_uv_indices[1][i] + 1,
                        v2, face_uv_indices[2][i] + 1);

            } else {
                fprintf(fout, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                        v0, face_uv_indices[0][i] + 1, v0,
                        v1, face_uv_indices[1][i] + 1, v1,
                        v2, face_uv_indices[2][i] + 1, v2);
            }
        }
    } else {
        for ( int i = 0; i < m_num_faces; ++i ) {
            int v0 = face_indices[0][i] + 1, v1 = face_indices[1][i] + 1, v2 = face_indices[2][i] + 1;
            if ( m_use_face_normals ) {
                fprintf(fout, "f %d %d %d\n", v0, v1, v2);
            } else {
                fprintf(fout, "f %d//%d %d//%d %d//%d\n", v0, v0, v1, v1, v2, v2);
            }
        }
    }

    fclose(fout);
}


std::string Mesh::to_string() const {
    std::cout << "has to string" << std::endl;
    std::stringstream oss;
    std::operator<<(oss, "Mesh[nv="); oss << m_num_vertices; std::operator<<(oss, ", nf="); oss << m_num_faces;
    if ( m_id != "" ) { std::operator<<(oss, ", id="); std::operator<<(oss, m_id); }
    std::operator<<(oss, ", bsdf="); std::operator<<(oss, m_bsdf->to_string()); std::operator<<(oss, "]");
    return oss.str();
}


NAMESPACE_END(psdr_jit)
