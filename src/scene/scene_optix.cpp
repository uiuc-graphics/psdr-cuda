// #include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/fwd.h>

#include <psdr/scene/scene_optix.h>
#include <psdr/shape/mesh.h>

#include <drjit-core/jit.h>
#include <drjit/autodiff.h>
#include <drjit/idiv.h>
#include <drjit/while_loop.h>
#include <drjit/texture.h>


// #include <drjit-core/array.h>
#include <cstdio>
#include <cstring>
#include <psdr/jit_optix_test.h>
#include <iostream>

NAMESPACE_BEGIN(psdr_jit)

const char *miss_and_closesthit_ptx = R"(
.version 7.4
.target sm_52
.address_size 64
.entry __miss__ms() {
    .reg .b32  %r<5>;

    mov.b32 %r0, 0;
    mov.b32 %r1, 1;
    mov.b32 %r2, 2;
    mov.b32 %r3, 3;
    mov.b32 %r4, -1;


    call _optix_set_payload, (%r0, %r4);
    call _optix_set_payload, (%r1, %r4);

    ret;
}
.entry __closesthit__ch() {
    .reg .f32   %f<3>;
    .reg .b32   %r<4>;
    .reg .b32   %rr<8>;
    .reg .b64   %rd<2>;
    .reg .b32   %u<2>;
    .reg .b32   %uu<2>;

    mov.b32 %r0, 0;
    mov.b32 %r1, 1;
    mov.b32 %u0, 2;
    mov.b32 %u1, 3;
    call (%rd1), _optix_get_sbt_data_ptr_64, ();
    ld.u32  %r2, [%rd1];
    ld.u32  %r3, [%rd1+4];
    call (%rr4), _optix_read_primitive_idx, ();
    add.s32     %rr6, %r2, %rr4;

    call (%f1, %f2), _optix_get_triangle_barycentrics, ();

    mov.b32     %uu0, %f1;
    mov.b32     %uu1, %f2;
    call _optix_set_payload, (%r0, %rr6); // shape id
    call _optix_set_payload, (%r1, %r3); // triangle id
    call _optix_set_payload, (%u0, %uu0); // u
    call _optix_set_payload, (%u1, %uu1); // v
    ret;

})";



namespace dr = drjit;

// Per scene OptiX state data structure
struct PathTracerState {
    OptixModule mod;
    OptixDeviceContext             context                  = 0;
    OptixTraversableHandle         gas_handle               = 0;  // Traversable handle for triangle AS
    uint32_t pipeline_jit_index;
    uint32_t sbt_jit_index ;
    UInt64 handle;
    void* d_gas = nullptr;
    UInt32 pipeline_handle;
    UInt32 sbt_handle;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    std::vector<HitGroupSbtRecord> hg_sbts;
    OptixShaderBindingTable sbt {};
    OptixModuleCompileOptions module_compile_options { };
    OptixAccelEmitDesc build_property = {};
    OptixProgramGroupOptions pgo { };
    OptixProgramGroupDesc pgd[2] { };
    OptixProgramGroup pg[2];
    OptixAccelBuildOptions accel_options {};
    OptixAccelBufferSizes gas_buffer_sizes;
    void *d_gas_temp = nullptr;
    size_t  h_compacted_size = 0;
};

void Intersection_OptiX::reserve(int64_t size) {
    PSDR_ASSERT(size > 0);
    if ( size != m_size ) {
        m_size = size;
        triangle_id = empty<IntC>(size);
        shape_id = empty<IntC>(size);
        uv = empty<Vector2fC>(size);
    }
}


Scene_OptiX::Scene_OptiX() {
    m_accel = nullptr;
}


Scene_OptiX::~Scene_OptiX() {
    if ( m_accel != nullptr ) {
        drjit::eval();
        jit_sync_thread();
        jit_free((m_accel)->d_gas);
        delete m_accel;
    }
}


void Scene_OptiX::configure(const std::vector<Mesh *> &meshes) {

    PSDR_ASSERT(!meshes.empty());
    size_t num_meshes = meshes.size();
    std::vector<int> face_offset(num_meshes + 1);
    face_offset[0] = 0;
    for ( size_t i = 0; i < num_meshes; ++i ) {
        face_offset[i + 1] = face_offset[i] + meshes[i]->m_num_faces;
    }

    // Push CUDA context to ensure we're using the right context for OptiX operations
    struct ContextGuard {
        ContextGuard() { jit_cuda_push_context(jit_cuda_context()); }
        ~ContextGuard() { jit_cuda_pop_context(); }
    } context_guard;

    // SBT storage is sized to the mesh count. Recreate it when topology grows
    // or shrinks instead of writing past the previous allocation.
    if (m_accel && m_accel->hg_sbts.size() != num_meshes) {
        drjit::eval();
        jit_sync_thread();
        jit_free(m_accel->d_gas);
        delete m_accel;
        m_accel = nullptr;
    }

    if ( m_accel == nullptr ) {
        init_optix_api();
        


        m_accel = new PathTracerState();


        m_accel->context = jit_optix_context();
        

        // =====================================================
        // Configure options for OptiX pipeline
        // =====================================================

        m_accel->module_compile_options.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        m_accel->module_compile_options.optLevel      = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

        m_accel->pipeline_compile_options.usesMotionBlur = false;
        m_accel->pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; // <-- Use this when possible
        m_accel->pipeline_compile_options.numPayloadValues = 4;
        m_accel->pipeline_compile_options.numAttributeValues = 0;
        m_accel->pipeline_compile_options.exceptionFlags =
            OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
            OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        m_accel->pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        m_accel->pipeline_compile_options.usesPrimitiveTypeFlags =
            (unsigned) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // <-- Use this when possible

        // =====================================================
        // Create Optix module from supplemental PTX code
        // =====================================================

        char log[1024];
        size_t log_size = sizeof(log);

        
        int rv = optixModuleCreate(
            m_accel->context, &m_accel->module_compile_options, &m_accel->pipeline_compile_options,
            miss_and_closesthit_ptx, strlen(miss_and_closesthit_ptx), log,
            &log_size, &m_accel->mod);
        if (rv) {
            fputs(log, stderr);
            jit_optix_check(rv);
        }



        // =====================================================
        // Create program groups (raygen provided by Dr.Jit..)
        // =====================================================



        m_accel->pgd[0].kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
        m_accel->pgd[0].miss.module                  = m_accel->mod;
        m_accel->pgd[0].miss.entryFunctionName       = "__miss__ms";
        m_accel->pgd[1].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        m_accel->pgd[1].hitgroup.moduleCH            = m_accel->mod;
        m_accel->pgd[1].hitgroup.entryFunctionNameCH = "__closesthit__ch";

        log_size = sizeof(log);
        jit_optix_check(optixProgramGroupCreate(m_accel->context, m_accel->pgd, 2 /* two at once */,
                                                &m_accel->pgo, log, &log_size, m_accel->pg));

        // =====================================================
        // Shader binding table setup
        // =====================================================
        m_accel->sbt.missRecordBase =
            jit_malloc(JitBackend::None, OPTIX_SBT_RECORD_HEADER_SIZE);
        m_accel->sbt.missRecordStrideInBytes     = OPTIX_SBT_RECORD_HEADER_SIZE;
        m_accel->sbt.missRecordCount             = 1;

        m_accel->hg_sbts = std::vector<HitGroupSbtRecord>();
        for ( size_t i = 0; i < num_meshes; ++i ) {
            m_accel->hg_sbts.push_back(HitGroupSbtRecord());
            m_accel->hg_sbts.back().data.shape_offset = face_offset[i];
            m_accel->hg_sbts.back().data.shape_id = i;
            jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[1], &m_accel->hg_sbts.back()));

        }
        m_accel->sbt.hitgroupRecordBase =
            jit_malloc(JitBackend::None, num_meshes * sizeof(HitGroupSbtRecord));
        m_accel->sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        m_accel->sbt.hitgroupRecordCount         = num_meshes;
        jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[0], m_accel->sbt.missRecordBase));
        std::memcpy(m_accel->sbt.hitgroupRecordBase, m_accel->hg_sbts.data(),
                             num_meshes * sizeof(HitGroupSbtRecord));
                             

        m_accel->sbt.missRecordBase =
            jit_malloc_migrate(m_accel->sbt.missRecordBase, JitBackend::CUDA, 1);
        m_accel->sbt.hitgroupRecordBase =
            jit_malloc_migrate(m_accel->sbt.hitgroupRecordBase, JitBackend::CUDA, 1);


        // =====================================================
        // Let Dr.Jit know about all of this
        // =====================================================

        m_accel->pipeline_handle = UInt32::steal(jit_optix_configure_pipeline(
            &m_accel->pipeline_compile_options, // <-- these pointers must stay
            m_accel->mod,
            m_accel->pg, 2
        ));

        m_accel->sbt_handle = UInt32::steal(jit_optix_configure_sbt(
            &m_accel->sbt, //     alive while Dr.Jit runs
            m_accel->pipeline_handle.index()
        ));


    }


    for ( size_t i = 0; i < num_meshes; ++i ) {
        m_accel->hg_sbts[i].data.shape_offset = face_offset[i];
        m_accel->hg_sbts[i].data.shape_id = i;
        // std::cout << m_accel->hg_sbts[i].data.shape_id << std::endl;
        // std::cout << m_accel->hg_sbts[i].data.shape_offset << std::endl;

    }
    jit_memcpy_async(JitBackend::CUDA, m_accel->sbt.hitgroupRecordBase, m_accel->hg_sbts.data(),
                                num_meshes * sizeof(HitGroupSbtRecord));


    // Ensure mesh buffers are evaluated and synchronized
    for ( size_t i = 0; i < num_meshes; ++i ) {
        drjit::eval(meshes[i]->m_vertex_buffer, meshes[i]->m_face_buffer);
    }
    drjit::sync_thread();

    uint32_t triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    std::vector<void*> vertex_buffer_ptrs(num_meshes);
    std::vector<OptixBuildInput> build_inputs(num_meshes);
    for ( size_t i = 0; i < num_meshes; ++i ) {
        const Mesh *mesh = meshes[i];

        build_inputs[i].type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_inputs[i].triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        build_inputs[i].triangleArray.numVertices   = static_cast<uint32_t>(mesh->m_num_vertices);

        vertex_buffer_ptrs[i] = jit_malloc(JitBackend::CUDA, sizeof(float)*(mesh->m_num_vertices*3));
        jit_memcpy(JitBackend::CUDA, vertex_buffer_ptrs[i], mesh->m_vertex_buffer.data(), sizeof(float)*(mesh->m_num_vertices*3));

        build_inputs[i].triangleArray.vertexBuffers = &vertex_buffer_ptrs[i];
        build_inputs[i].triangleArray.flags         = triangle_input_flags;
        build_inputs[i].triangleArray.numSbtRecords = 1;
        build_inputs[i].triangleArray.indexFormat           = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        build_inputs[i].triangleArray.numIndexTriplets      = mesh->m_num_faces;
        build_inputs[i].triangleArray.indexBuffer           = (CUdeviceptr) mesh->m_face_buffer.data();
    }
    m_accel->accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    m_accel->accel_options.buildFlags =
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    
    jit_optix_check(optixAccelComputeMemoryUsage(
        m_accel->context, &m_accel->accel_options, build_inputs.data(), num_meshes, &m_accel->gas_buffer_sizes));

    m_accel->d_gas_temp =
        jit_malloc(JitBackend::CUDA, m_accel->gas_buffer_sizes.tempSizeInBytes);

    if ((m_accel)->d_gas != nullptr) {
        jit_free((m_accel)->d_gas);
    }
    

    m_accel->d_gas = jit_malloc(JitBackend::CUDA, m_accel->gas_buffer_sizes.outputSizeInBytes);

    size_t *d_compacted_size = (size_t *) jit_malloc(JitBackend::CUDA, sizeof(size_t));
    
    
    m_accel->build_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    m_accel->build_property.result = d_compacted_size;

    // Sync CUDA operations before OptiX acceleration structure build
    jit_sync_thread();

    jit_optix_check(optixAccelBuild(
        m_accel->context, jit_cuda_stream(), &m_accel->accel_options, build_inputs.data(), num_meshes, m_accel->d_gas_temp,
        m_accel->gas_buffer_sizes.tempSizeInBytes, m_accel->d_gas,
        m_accel->gas_buffer_sizes.outputSizeInBytes, &m_accel->gas_handle, &m_accel->build_property, 1));


    jit_free(m_accel->d_gas_temp);

    for ( size_t i = 0; i < num_meshes; ++i )
        jit_free(vertex_buffer_ptrs[i]);

    jit_memcpy(JitBackend::CUDA, &m_accel->h_compacted_size, d_compacted_size, sizeof(size_t));

    if (m_accel->h_compacted_size < m_accel->gas_buffer_sizes.outputSizeInBytes) {
        void *d_gas_compact = jit_malloc(JitBackend::CUDA, m_accel->h_compacted_size);
        jit_optix_check(optixAccelCompact(m_accel->context, jit_cuda_stream(),
                                            m_accel->gas_handle, d_gas_compact,
                                            m_accel->h_compacted_size, &m_accel->gas_handle));
        jit_free(m_accel->d_gas);
        m_accel->d_gas = d_gas_compact;
    }

    jit_free(d_compacted_size);

}



bool Scene_OptiX::is_ready() const {
    return m_accel != nullptr;
}


template <bool ad>
Intersection_OptiX Scene_OptiX::ray_intersect(const Ray<ad> &ray, Mask<ad> &active) const {
    drjit::sync_thread();
    const int m = static_cast<int>(slices<Vector3f<ad>>(ray.o));

    active &= ~drjit::isnan(ray.o.x());
    active &= ~drjit::isnan(ray.o.y());
    active &= ~drjit::isnan(ray.o.z());
    active &= ~drjit::isnan(ray.d.x());
    active &= ~drjit::isnan(ray.d.y());
    active &= ~drjit::isnan(ray.d.z());

    Intersection_OptiX m_its;
    m_its.reserve(m);
    FloatC ox,oy,oz,dx,dy,dz;
    if constexpr(ad) {
           ox = detach<false>(ray.o.x()),
           oy = detach<false>(ray.o.y()),
           oz = detach<false>(ray.o.z());

           dx = detach<false>(ray.d.x()),
           dy = detach<false>(ray.d.y()),
           dz = detach<false>(ray.d.z());
    } else {
           ox = ray.o.x(),
           oy = ray.o.y(), //268.342, 540.3, 278.433 
           oz = ray.o.z();

           dx = ray.d.x(),
           dy = ray.d.y(),
           dz = ray.d.z();

    }
    FloatC mint = RayEpsilon, maxt = 100000000.f, time = 0.f;
    UInt32 ray_mask(255), ray_flags(0), sbt_offset(0), sbt_stride(1),
        miss_sbt_index(0);
    UInt32 payload_0(0);
    UInt32 payload_1(0);
    UInt32 payload_u(0);
    UInt32 payload_v(0);
    // MaskC mask = true;

    m_accel->handle = dr::opaque<UInt64>(m_accel->gas_handle);

    uint32_t trace_args[] {
        m_accel->handle.index(),
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        mint.index(), maxt.index(), time.index(),
        ray_mask.index(), ray_flags.index(),
        sbt_offset.index(), sbt_stride.index(),
        miss_sbt_index.index(), payload_0.index(), payload_1.index(), payload_u.index(), payload_v.index()
    };

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args,
                        0, nullptr, nullptr,  // no hit objects
                        0, 0, 0,              // no reordering
                        1,                    // invoke
                        active.index(), m_accel->pipeline_handle.index(), m_accel->sbt_handle.index());
    using Single = drjit::float32_array_t<FloatC>;
    m_its.shape_id = UInt32::steal(trace_args[15]); // slow
    m_its.triangle_id = UInt32::steal(trace_args[16]);
    m_its.uv[0] = drjit::reinterpret_array<Single, UInt32>(UInt32::steal(trace_args[17]));
    m_its.uv[1] = drjit::reinterpret_array<Single, UInt32>(UInt32::steal(trace_args[18]));
    // m_its.uv = Vector2fC(0.5f); // very fast
    // m_its.shape_id = IntC(0);
    // m_its.triangle_id = IntC(0);
    active &= (m_its.shape_id >= 0) && (m_its.triangle_id >= 0);
    return m_its;

}


// Explicit instantiations
template Intersection_OptiX Scene_OptiX::ray_intersect<false>(const RayC &ray, MaskC &active) const;
template Intersection_OptiX Scene_OptiX::ray_intersect<true>(const RayD &ray, MaskD &active) const;

NAMESPACE_END(psdr_jit)
