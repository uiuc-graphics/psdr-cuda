#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>
#include <psdr/jit_optix_test.h>
#include <iostream>
#include <psdr/optix_stubs.h>
#include <vector>

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

    call _optix_set_payload, (%r0, %rr6); // shape id
    call _optix_set_payload, (%r1, %r3); // triangle id
    ret;



})";
namespace dr = drjit;

using Float = dr::CUDAArray<float>;
using UInt32 = dr::CUDAArray<uint32_t>;
using UInt64 = dr::CUDAArray<uint64_t>;
using Mask = dr::CUDAArray<bool>;

void demo() {
    // Push CUDA context for OptiX operations
    jit_cuda_push_context(jit_cuda_context());

    // =====================================================
    // Copy vertex data to device
    // =====================================================
    float h_vertices[18] = {  0.4f, 0.4f, -0.15f,
                             -0.4f, 0.4f, -0.15f,
                             0.0f, -0.4f, -0.15f,
                             -0.4f, -0.4f, -0.2f,
                             0.4f, -0.4f, -0.2f,
                             0.0f, 0.4f, -0.2f   };

    UInt32 m_faces(0,1,2,3,4,5);


    float h_vertices1[18] = {0.8f, 0.8f, 0.0f,
                             -0.8f, 0.8f, 0.0f,
                             0.0f, -0.8f, 0.0f,
                             -0.8f, -0.8f, 0.1f,
                             0.8f, -0.8f, 0.1f,
                             0.0f, 0.8f, 0.1f   };

    UInt32 m_faces1(0,1,2,3,4,5);


    int m_vertex_count = 6;
    int m_face_count = 2;

    int num_meshes = 2;

    


    uint32_t triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    std::vector<void *> vertex_buffer_ptrs(num_meshes);
    std::vector<OptixBuildInput> build_inputs(num_meshes);


        
        void *d_vertices = jit_malloc(JitBackend::CUDA, sizeof(h_vertices));
        jit_memcpy(JitBackend::CUDA, d_vertices, h_vertices, sizeof(h_vertices));

        // =====================================================
        // Build geometric acceleration data structure
        // =====================================================

        build_inputs[0].type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_inputs[0].triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        build_inputs[0].triangleArray.numVertices   = m_vertex_count;
        build_inputs[0].triangleArray.vertexBuffers = &d_vertices;
        build_inputs[0].triangleArray.flags         = triangle_input_flags;
        build_inputs[0].triangleArray.numSbtRecords = 1;
        build_inputs[0].triangleArray.indexFormat           = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        build_inputs[0].triangleArray.numIndexTriplets      = m_face_count;
        build_inputs[0].triangleArray.indexBuffer           = (CUdeviceptr) m_faces.data();

        void *d_vertices1 = jit_malloc(JitBackend::CUDA, sizeof(h_vertices1));
        jit_memcpy(JitBackend::CUDA, d_vertices1, h_vertices1, sizeof(h_vertices1));

        build_inputs[1].type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_inputs[1].triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        build_inputs[1].triangleArray.numVertices   = m_vertex_count;
        build_inputs[1].triangleArray.vertexBuffers = &d_vertices1;
        build_inputs[1].triangleArray.flags         = triangle_input_flags;
        build_inputs[1].triangleArray.numSbtRecords = 1;
        build_inputs[1].triangleArray.indexFormat           = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        build_inputs[1].triangleArray.numIndexTriplets      = m_face_count;
        build_inputs[1].triangleArray.indexBuffer           = (CUdeviceptr) m_faces1.data();




    OptixDeviceContext context = jit_optix_context();
    OptixAccelBuildOptions accel_options {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags =
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    OptixAccelBufferSizes gas_buffer_sizes;
    jit_optix_check(optixAccelComputeMemoryUsage(
        context, &accel_options, build_inputs.data(), num_meshes, &gas_buffer_sizes));

    void *d_gas_temp =
        jit_malloc(JitBackend::CUDA, gas_buffer_sizes.tempSizeInBytes);

    void *d_gas =
        jit_malloc(JitBackend::CUDA, gas_buffer_sizes.outputSizeInBytes);

    size_t *d_compacted_size = (size_t *) jit_malloc(JitBackend::CUDA, sizeof(size_t)),
            h_compacted_size = 0;
    OptixAccelEmitDesc build_property = {};
    build_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    build_property.result = d_compacted_size;

    OptixTraversableHandle gas_handle;
    jit_optix_check(optixAccelBuild(
        context, 0, &accel_options, build_inputs.data(), num_meshes, d_gas_temp,
        gas_buffer_sizes.tempSizeInBytes, d_gas,
        gas_buffer_sizes.outputSizeInBytes, &gas_handle, &build_property, 1));

    jit_free(d_gas_temp);
    jit_free(d_vertices);
    jit_free(d_vertices1);

    // =====================================================
    // Compactify geometric acceleration data structure
    // =====================================================

    jit_memcpy(JitBackend::CUDA, &h_compacted_size, d_compacted_size, sizeof(size_t));

    if (h_compacted_size < gas_buffer_sizes.outputSizeInBytes) {
        void *d_gas_compact = jit_malloc(JitBackend::CUDA, h_compacted_size);
        jit_optix_check(optixAccelCompact(context, jit_cuda_stream(),
                                           gas_handle, d_gas_compact,
                                           h_compacted_size, &gas_handle));
        jit_free(d_gas);
        d_gas = d_gas_compact;
    }

    jit_free(d_compacted_size);

    // =====================================================
    // Configure options for OptiX pipeline
    // =====================================================

    OptixModuleCompileOptions module_compile_options { };
    module_compile_options.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    module_compile_options.optLevel      = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

    OptixPipelineCompileOptions pipeline_compile_options { };
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; // <-- Use this when possible
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 0;
    // Use the same exception checks as Scene_OptiX and Dr.Jit.
    pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags =
        (unsigned) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // <-- Use this when possible

    // =====================================================
    // Create Optix module from supplemental PTX code
    // =====================================================

    char log[1024];
    size_t log_size = sizeof(log);

    OptixModule mod;
    int rv = optixModuleCreate(
        context, &module_compile_options, &pipeline_compile_options,
        miss_and_closesthit_ptx, strlen(miss_and_closesthit_ptx), log,
        &log_size, &mod);
    if (rv) {
        fputs(log, stderr);
        jit_optix_check(rv);
    }

    // =====================================================
    // Create program groups (raygen provided by Dr.Jit..)
    // =====================================================

    OptixProgramGroupOptions pgo { };
    OptixProgramGroupDesc pgd[2] { };
    OptixProgramGroup pg[2];

    pgd[0].kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgd[0].miss.module                  = mod;
    pgd[0].miss.entryFunctionName       = "__miss__ms";
    pgd[1].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgd[1].hitgroup.moduleCH            = mod;
    pgd[1].hitgroup.entryFunctionNameCH = "__closesthit__ch";

    log_size = sizeof(log);
    jit_optix_check(optixProgramGroupCreate(context, pgd, 2 /* two at once */,
                                            &pgo, log, &log_size, pg));

    // =====================================================
    // Shader binding table setup
    // =====================================================
    OptixShaderBindingTable sbt {};

    sbt.missRecordBase =
        jit_malloc(JitBackend::None, OPTIX_SBT_RECORD_HEADER_SIZE);
    sbt.missRecordStrideInBytes     = OPTIX_SBT_RECORD_HEADER_SIZE;
    sbt.missRecordCount             = 1;


    std::vector<HitGroupSbtRecord> hg_sbts;

    hg_sbts.push_back(HitGroupSbtRecord());

    hg_sbts.back().data.shape_offset = 0;
    hg_sbts.back().data.shape_id = 1;
    jit_optix_check(optixSbtRecordPackHeader(pg[1], &hg_sbts.back()));

    hg_sbts.push_back(HitGroupSbtRecord());
    hg_sbts.back().data.shape_offset = 2;
    hg_sbts.back().data.shape_id = 2;
    jit_optix_check(optixSbtRecordPackHeader(pg[1], &hg_sbts.back()));


    

    sbt.hitgroupRecordBase =
        jit_malloc(JitBackend::None, num_meshes * sizeof(HitGroupSbtRecord));
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount         = num_meshes;

    jit_optix_check(optixSbtRecordPackHeader(pg[0], sbt.missRecordBase));
    
    jit_memcpy_async(JitBackend::CUDA, sbt.hitgroupRecordBase, hg_sbts.data(),
                         num_meshes * sizeof(HitGroupSbtRecord));

    sbt.missRecordBase =
        jit_malloc_migrate(sbt.missRecordBase, JitBackend::CUDA, 1);
    sbt.hitgroupRecordBase =
        jit_malloc_migrate(sbt.hitgroupRecordBase, JitBackend::CUDA, 1);

    // =====================================================
    // Let Dr.Jit know about all of this
    // =====================================================

    UInt32 pipeline_handle = UInt32::steal(jit_optix_configure_pipeline(
        &pipeline_compile_options, // <-- these pointers must stay
        mod,
        pg, 2
    ));

    UInt32 sbt_handle = UInt32::steal(jit_optix_configure_sbt(
        &sbt, //     alive while Dr.Jit runs
        pipeline_handle.index()
    ));

    // Do four times to verify caching, with mask in it. 3 + 4
    for (int i = 0; i < 2; ++i) {
        // =====================================================
        // Generate a camera ray
        // =====================================================

        int res = 16;
        UInt32 index = dr::arange<UInt32>(res * res),
               x     = index % res,
               y     = index / res;

        Float ox = Float(x) * (2.0f / res) - 2.0f,
              oy = Float(y) * (2.0f / res) - 2.0f,
              oz = -1.f;

        Float dx = 0.f, dy = 0.f, dz = 1.f;

        Float mint = 0.f, maxt = 100.f, time = 0.f;

        UInt64 handle = dr::opaque<UInt64>(gas_handle);
        UInt32 ray_mask(255), ray_flags(0), sbt_offset(0), sbt_stride(1),
            miss_sbt_index(0);

        UInt32 payload_0(0);
        UInt32 payload_1(0);
        Mask mask = true;

        // =====================================================
        // Launch a ray tracing call
        // =====================================================

        uint32_t trace_args[] {
            handle.index(),
            ox.index(), oy.index(), oz.index(),
            dx.index(), dy.index(), dz.index(),
            mint.index(), maxt.index(), time.index(),
            ray_mask.index(), ray_flags.index(),
            sbt_offset.index(), sbt_stride.index(),
            miss_sbt_index.index(), payload_0.index(), payload_1.index()
        };

        jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args,
                            0, nullptr, nullptr,  // no hit objects
                            0, 0, 0,              // no reordering
                            1,                    // invoke
                            mask.index(), pipeline_handle.index(), sbt_handle.index());

        payload_0 = UInt32::steal(trace_args[15]);
        jit_var_eval(payload_0.index());

        UInt32 payload_0_host =
            UInt32::steal(jit_var_migrate(payload_0.index(), JitBackend::None));
        jit_sync_thread();

        for (int k = 0; k < res; ++k) {
            for (int j = 0; j < res; ++j)
                printf("%i ", payload_0_host.data()[k*res + j]);
            printf("\n");
        }
        printf("\n");


        payload_1 = UInt32::steal(trace_args[16]);
        jit_var_eval(payload_1.index());

        UInt32 payload_1_host =
            UInt32::steal(jit_var_migrate(payload_1.index(), JitBackend::None));
        jit_sync_thread();

        for (int k = 0; k < res; ++k) {
            for (int j = 0; j < res; ++j)
                printf("%i ", payload_1_host.data()[k*res + j]);
            printf("\n");
        }
        printf("\n");

    }

    for (int i = 0; i < 2; ++i) {
        // =====================================================
        // Generate a camera ray
        // =====================================================

        int res = 16;
        UInt32 index = dr::arange<UInt32>(res * res),
               x     = index % res,
               y     = index / res;

        Float ox = Float(x) * (2.0f / res) - 1.0f,
              oy = Float(y) * (2.0f / res) - 1.0f,
              oz = -1.f;

        Float dx = 0.f, dy = 0.f, dz = 1.f;

        Float mint = 0.f, maxt = 100.f, time = 0.f;

        UInt64 handle = dr::opaque<UInt64>(gas_handle);
        UInt32 ray_mask(255), ray_flags(0), sbt_offset(0), sbt_stride(1),
            miss_sbt_index(0);

        UInt32 payload_0(0);
        UInt32 payload_1(0);
        Mask mask = true;

        // =====================================================
        // Launch a ray tracing call
        // =====================================================

        uint32_t trace_args[] {
            handle.index(),
            ox.index(), oy.index(), oz.index(),
            dx.index(), dy.index(), dz.index(),
            mint.index(), maxt.index(), time.index(),
            ray_mask.index(), ray_flags.index(),
            sbt_offset.index(), sbt_stride.index(),
            miss_sbt_index.index(), payload_0.index(), payload_1.index()
        };

        jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args,
                            0, nullptr, nullptr,  // no hit objects
                            0, 0, 0,              // no reordering
                            1,                    // invoke
                            mask.index(), pipeline_handle.index(), sbt_handle.index());

        payload_0 = UInt32::steal(trace_args[15]);
        jit_var_eval(payload_0.index());

        UInt32 payload_0_host =
            UInt32::steal(jit_var_migrate(payload_0.index(), JitBackend::None));
        jit_sync_thread();

        for (int k = 0; k < res; ++k) {
            for (int j = 0; j < res; ++j)
                printf("%i ", payload_0_host.data()[k*res + j]);
            printf("\n");
        }
        printf("\n");


        payload_1 = UInt32::steal(trace_args[16]);
        jit_var_eval(payload_1.index());

        UInt32 payload_1_host =
            UInt32::steal(jit_var_migrate(payload_1.index(), JitBackend::None));
        jit_sync_thread();

        for (int k = 0; k < res; ++k) {
            for (int j = 0; j < res; ++j)
                printf("%i ", payload_1_host.data()[k*res + j]);
            printf("\n");
        }
        printf("\n");

    }


    // =====================================================
    // Cleanup
    // =====================================================

    jit_free(d_gas);

    // Pop CUDA context
    jit_cuda_pop_context();
}


void jit_test::trace_ray() {

    jit_init((int) JitBackend::CUDA);
    // jit_set_log_level_stderr(LogLevel::Trace);
    init_optix_api();
    demo();

    // Shut down Dr.Jit (will destroy OptiX device context)
    // The Python Dr.Jit module owns the runtime; do not shut it down here.

}
