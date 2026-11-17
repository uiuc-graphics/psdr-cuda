#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/transform.h>
#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/orthographic.h>

NAMESPACE_BEGIN(psdr_jit)

void OrthographicCamera::configure(bool cache=true) {
    Sensor::configure(cache);

    // std::cout << transform::perspective(m_fov_x, m_near_clip, m_far_clip) << std::endl;
    // std::cout << transform::orthographic(m_near_clip, m_far_clip) << std::endl;
    // PSDR_ASSERT(0);

    ScalarMatrix4f camera_to_sample =
        transform::scale(ScalarVector3f(-0.5f, -0.5f * m_aspect, 1.f)) *
        transform::translate(ScalarVector3f(-1.f, -1.f / m_aspect, 0.f)) *
        transform::orthographic(m_near_clip, m_far_clip);

    // ScalarMatrix4f camera_to_sample =
    //     transform::scale(ScalarVector3f(-0.5f, -0.5f * m_aspect, 1.f)) *
    //     transform::translate(ScalarVector3f(-1.f, -1.f / m_aspect, 0.f)) *
    //     transform::scale(ScalarVector3f(1.f, 1.f, 1.f/(m_far_clip-m_near_clip)) *
    //     transform::translate(ScalarVector3f(0.f, 0.f, -m_near_clip));

    m_camera_to_sample = Matrix4fD(camera_to_sample);
    m_sample_to_camera = Matrix4fD(inverse(camera_to_sample));

    Matrix4fD m_to_world = m_to_world_left * m_to_world_raw * m_to_world_right;

    m_world_to_sample = m_camera_to_sample*inverse(m_to_world);
    m_sample_to_world = m_to_world*m_sample_to_camera;

    m_camera_pos = transform_pos(m_to_world, zeros<Vector3fD>());
    m_camera_dir = transform_dir(m_to_world, Vector3fD(0.f, 0.f, 1.f));

    
    if (cache) drjit::make_opaque(m_camera_to_sample, m_sample_to_camera, m_world_to_sample, m_sample_to_world, m_camera_pos, m_camera_dir);

    Vector3fD v00 = transform_pos(m_sample_to_camera, Vector3fD(0.f, 0.f, 0.f)), //  bottom-left corner of the image at the near plane
              v10 = transform_pos(m_sample_to_camera, Vector3fD(1.f, 0.f, 0.f)), // bottom-right corner of the image at the near plane
              v11 = transform_pos(m_sample_to_camera, Vector3fD(1.f, 1.f, 0.f)), //    top-right corner of the image at the near plane
              vc  = transform_pos(m_sample_to_camera, Vector3fD(.5f, .5f, 0.f)); //              center of the image at the near plane
    m_inv_area = rcp(norm(v00 - v10)*norm(v11 - v10))*squared_norm(vc);

    // Construct list of primary edges

    PSDR_ASSERT_MSG(m_scene, "Missing scene data!");

    if ( m_scene->m_opts.sppe > 0 ) {
        const std::vector<Mesh*> &meshes = m_scene->m_meshes;
        std::vector<Vectori<5, true>> per_mesh_info(meshes.size());
        std::vector<int> offset(1, 0);

        for ( size_t i = 0; i < meshes.size(); ++i ) {
            const Mesh *mesh = meshes[i];
            if ( mesh->m_enable_edges ) {
                auto &info = per_mesh_info[i];

                PSDR_ASSERT(all(mesh->m_edge_indices[2] >= 0));
                MaskD valid = (mesh->m_edge_indices[3] >= 0);

                Vector3fD e0 = normalize(m_camera_pos - gather<Vector3fD>(mesh->m_triangle_info->p0, mesh->m_edge_indices[2])),
                          e1 = normalize(m_camera_pos - gather<Vector3fD>(mesh->m_triangle_info->p0, mesh->m_edge_indices[3], valid)),
                          n0 = gather<Vector3fD>(mesh->m_triangle_info->face_normal, mesh->m_edge_indices[2]),
                          n1 = gather<Vector3fD>(mesh->m_triangle_info->face_normal, mesh->m_edge_indices[3], valid);

                // add UV edge
                MaskD uv_mask;

                if (mesh->m_has_uv) {
                    Vector3iD fuv1_d = gather<Vector3iD>(mesh->m_face_uv_indices, mesh->m_edge_indices[2]);
                    Vector3iD fuv2_d = gather<Vector3iD>(mesh->m_face_uv_indices, mesh->m_edge_indices[3], neq(mesh->m_edge_indices[3], -1));

                    Vector3iC fuv1 = detach<false>(fuv1_d);
                    Vector3iC fuv2 = detach<false>(fuv2_d);

                    IntC uv_cut = zeros<IntC>(slices(valid));

                    uv_cut = select(  (eq(fuv1[0], fuv2[0]) || eq(fuv1[0], fuv2[1]) || eq(fuv1[0], fuv2[2])) , IntC(1), 0);
                    uv_cut = select( (eq(fuv1[1], fuv2[0]) || eq(fuv1[1], fuv2[1]) || eq(fuv1[1], fuv2[2])), uv_cut+1, uv_cut);
                    uv_cut = select( (eq(fuv1[2], fuv2[0]) || eq(fuv1[2], fuv2[1]) || eq(fuv1[2], fuv2[2])), uv_cut+1, uv_cut);

                    uv_mask = neq(uv_cut, 2);
                    PSDR_ASSERT(any(neq(uv_cut, 3)));
                }

                if ( mesh->m_use_face_normals ) {
                    MaskD skip = valid;
                    skip &= (dot(e0, n0) < Epsilon && dot(e1, n1) < Epsilon) ||
                            (dot(n0, n1) > 1.f - Epsilon);
                    if (mesh->m_has_uv) {
                        info = compressD<Vectori<5, true>>(mesh->m_edge_indices, ~skip || uv_mask);
                    } else {
                        info = compressD<Vectori<5, true>>(mesh->m_edge_indices, ~skip);
                    }
                } else {
                    MaskD active = ~valid;
                    active |= (dot(e0, n0) > Epsilon)^(dot(e1, n1) > Epsilon);
                    if (mesh->m_has_uv) {
                        info = compressD<Vectori<5, true>>(mesh->m_edge_indices, active || uv_mask);
                    } else {
                        info = compressD<Vectori<5, true>>(mesh->m_edge_indices, active);
                    }
                }
                PSDR_ASSERT(slices(info) > 0);
                offset.push_back(offset.back() + static_cast<int>(slices(info)));
            } else {
                offset.push_back(offset.back());
            }
        }

        if ( offset.back() > 0 ) {
            m_edge_info = empty<PrimaryEdgeInfo>(offset.back());
            for ( size_t i = 0; i < meshes.size(); ++i ) {
                const Mesh *mesh = meshes[i];
                if ( mesh->m_enable_edges ) {
                    const auto &info = per_mesh_info[i];
                    const int m = static_cast<int>(slices(info));
                    const IntD idx = arange<IntD>(m) + offset[i];

                    Vector3fD p0 = gather<Vector3fD>(mesh->m_vertex_positions, info[0]),
                              p1 = gather<Vector3fD>(mesh->m_vertex_positions, info[1]);

                    Vector3fD q0 = transform_pos(m_world_to_sample, p0),
                              q1 = transform_pos(m_world_to_sample, p1);

                    scatter(m_edge_info.p0, head<2>(q0), idx);
                    scatter(m_edge_info.p1, head<2>(q1), idx);

                    Vector2fD e = head<2>(detach<false>(q1) - detach<false>(q0));
                    FloatD len = norm(e);
                    e /= len;
                    scatter(m_edge_info.edge_normal, Vector2fD(-e.y(), e.x()), idx);
                    scatter(m_edge_info.edge_length, len, idx);
                }
            }
            m_edge_distrb.init(detach<false>(m_edge_info.edge_length));
            m_enable_edges = true;
        } else {
            m_enable_edges = false;
        }
    }
}


std::string OrthographicCamera::to_string() const {
    return "OrthographicCamera";
}


RayC OrthographicCamera::sample_primary_ray(const Vector2fC &samples) const {
    // Vector3fC d = normalize(transform_pos<FloatC>(detach(m_sample_to_camera), concat(samples, Vectorf<1, false>(0.f))));
    Matrix4fD m_to_world = m_to_world_left * m_to_world_raw * m_to_world_right;
    Matrix4fC to_world = detach<false>(m_to_world);
    Vector3fC near_p = transform_pos<FloatC>(detach<false>(m_sample_to_camera), Vector3fC(samples.x(), samples.y(), 0.f));
    return RayC(
        transform_pos<FloatC>(to_world, near_p),
        transform_dir<FloatC>(to_world, Vector3fC(0.f,0.f,1.f))
    );
}


RayD OrthographicCamera::sample_primary_ray(const Vector2fD &samples) const {
    Matrix4fD m_to_world = m_to_world_left * m_to_world_raw * m_to_world_right;
    Vector3fD near_p = transform_pos<FloatD>(m_sample_to_camera, Vector3fD(samples.x(), samples.y(), 0.f));
    return RayD(
        transform_pos<FloatD>(m_to_world, near_p),
        transform_dir<FloatD>(m_to_world, Vector3fD(0.f,0.f,1.f))
    );
}


SensorDirectSampleC OrthographicCamera::sample_direct(const Vector3fC &p) const {
    SensorDirectSampleC result;
    result.q = head<2>(transform_pos<FloatC>(detach<false>(m_world_to_sample), p));

    Vector2iC iq = floor2int<Vector2iC, Vector2fC>(result.q*m_resolution);
    result.is_valid = iq.x() >= 0 && iq.x() < m_resolution.x() &&
                      iq.y() >= 0 && iq.y() < m_resolution.y();
    result.pixel_idx = select(result.is_valid, iq.y()*m_resolution.x() + iq.x(), -1);

    Vector3fC dir = p - detach<false>(m_camera_pos);
    FloatC dist2 = squared_norm(dir);
    dir /= safe_sqrt(dist2);

    FloatC cosTheta = dot(detach<false>(m_camera_dir), dir);
    result.sensor_val = rcp(dist2)*pow(rcp(cosTheta), 3.f)*detach<false>(m_inv_area);
    return result;
}


PrimaryEdgeSample OrthographicCamera::sample_primary_edge(const FloatC &_sample1) const {
    FloatC sample1 = _sample1;

    PrimaryEdgeSample result;
    const int m = static_cast<int>(slices(sample1));

    IntC edge_idx;
    std::tie(edge_idx, result.pdf) = m_edge_distrb.sample_reuse<false>(sample1);
    result.pdf /= detach<false>(gather<FloatD>(m_edge_info.edge_length, IntD(edge_idx)));
    Vector2fC edge_normal = detach<false>(gather<Vector2fD>(m_edge_info.edge_normal, IntD(edge_idx)));
    Vector2fD p0 = gather<Vector2fD>(m_edge_info.p0, IntD(edge_idx)), p1 = gather<Vector2fD>(m_edge_info.p1, IntD(edge_idx));
    Vector2fD p_    = fmadd(p0, 1.0f - sample1, p1*sample1);
    Vector2fC p     = detach<false>(p_);
    result.x_dot_n  = dot(p_, edge_normal);

    Vector2iC ip    = floor2int<Vector2iC, Vector2fC>(p*m_resolution);
    MaskC valid     = ip.x() >= 0 && ip.x() < m_resolution.x() &&
                      ip.y() >= 0 && ip.y() < m_resolution.y();

    result.idx      = full<IntC>(-1, m);
    masked(result.idx, valid) = ip.y()*m_resolution.x() + ip.x();

    result.ray_p    = sample_primary_ray(p + EdgeEpsilon*edge_normal);
    result.ray_n    = sample_primary_ray(p - EdgeEpsilon*edge_normal);

    return result;
}

NAMESPACE_END(psdr_jit)
