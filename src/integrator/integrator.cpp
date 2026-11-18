#include <chrono>
#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/sensor.h>
#include <psdr/integrator/integrator.h>

NAMESPACE_BEGIN(psdr_jit)

// An all-minus-one array selects the full image; explicit batches contain pixel indices.
static bool is_pixel_batch(const IntC &pixels, const RenderOption &opts) {
    if (all(pixels == -1))
        return false;
    const int64_t count = static_cast<int64_t>(opts.width) * opts.height;
    PSDR_ASSERT_MSG(count > 0 && count <= std::numeric_limits<int>::max(), "Invalid film size!");
    PSDR_ASSERT_MSG(all(pixels >= 0 && pixels < static_cast<int>(count)), "Batch pixel out of range!");
    return true;
}

SpectrumC Integrator::renderC(const Scene &scene, int sensor_id, int seed, IntC pix_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    const RenderOption &opts = scene.m_opts;

    const bool batch = is_pixel_batch(detach<false>(pix_id), scene.m_opts);
    if (batch && seed == -1) {
        PSDR_ASSERT_MSG(false, "While using batch rendering, seed must be set!");
    }

    if (!batch && opts.spp > 0 && seed != -1) {
        const int64_t count = static_cast<int64_t>(opts.height) * opts.width * opts.spp;
        scene.m_samplers[0].seed(arange<UInt64C>(count) + seed);
    }

    SpectrumC result;
    if (!batch) {
        result = __render<false>(scene, sensor_id);
    } else {
        result = __render_batch<false>(scene, sensor_id, pix_id, seed);
    }
    auto end_time = high_resolution_clock::now();
    if ( opts.log_level ) {
        std::stringstream oss;
        std::operator<<(oss, "Rendered in ");
        oss << duration_cast<duration<double>>(end_time - start_time).count();
        std::operator<<(oss, " seconds.");
        log(oss.str().c_str());
    }
    drjit::eval(result);
    return result;
}


SpectrumD Integrator::renderD(const Scene &scene, int sensor_id, int seed, IntD pix_id) const {
    const bool batch = is_pixel_batch(detach<false>(pix_id), scene.m_opts);
    if (batch && seed == -1) {
        PSDR_ASSERT_MSG(false, "While using batch rendering, seed must be set!");
    }
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    const RenderOption &opts = scene.m_opts;

    int64_t num_pixels = static_cast<int64_t>(opts.height)*opts.width;
    if (!batch && opts.spp > 0 && seed != -1)
        scene.m_samplers[0].seed(arange<UInt64C>(num_pixels * opts.spp) + seed);
    if (opts.sppe > 0 && seed != -1) {
        scene.m_samplers[1].seed(arange<UInt64C>(num_pixels * opts.sppe) + seed);
    }
    if (opts.sppse > 0 && seed != -1) {
        scene.m_samplers[2].seed(arange<UInt64C>(num_pixels * opts.sppse) + seed);
    }

    // Interior integral
    SpectrumD result;
    if (!batch) {
        result = __render<true>(scene, sensor_id);
    } else {
        result = __render_batch<true>(scene, sensor_id, pix_id, seed);
    }

    // Edge samplers use full-image pixel indices, including during a batch.
    // Accumulate into a full-sized buffer before gathering requested pixels.
    if (opts.sppe > 0 || opts.sppse > 0) {
        SpectrumD edges;
        if (batch)
            edges = zeros<SpectrumD>(num_pixels);
        SpectrumD &target = batch ? edges : result;
        if (opts.sppe > 0)
            render_primary_edges(scene, sensor_id, target);
        if (opts.sppse > 0)
            render_secondary_edges(scene, sensor_id, target);
        if (batch)
            result += gather<SpectrumD>(edges, pix_id);
    }

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        std::operator<<(oss, "Rendered in ");
        oss << duration_cast<duration<double>>(end_time - start_time).count();
        std::operator<<(oss, " seconds.");
        log(oss.str().c_str());
    }
    drjit::eval(result);
    return result;
}


template <bool ad>
Spectrum<ad> Integrator::__render(const Scene &scene, int sensor_id) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");
    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.width*opts.height;

    Spectrum<ad> result = zeros<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.spp > 1) ) idx /= opts.spp;

        auto [dx, dy] = meshgrid(arange<Float<ad>>(opts.width),arange<Float<ad>>(opts.height));

        Vector2f<ad> samples_base = gather<Vector2f<ad>>(Vector2f<ad>(dx, dy), idx);

        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                /ScalarVector2f(opts.width, opts.height);
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray);
        masked(value, ~drjit::isfinite<Spectrum<ad>>(value) || drjit::isnan<Spectrum<ad>>(value)) = 0.f;
        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, result[j], value[j], idx);
        }
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
    }
    
    return result;
}


template <bool ad>
Spectrum<ad> Integrator::__render_batch(const Scene &scene, int sensor_id, Int<ad> pix_id, int seed) const {
    // std::cout << "using batch renderer" << std::endl;
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");
    const RenderOption &opts = scene.m_opts;
    const int num_pixels = pix_id.size();


    Spectrum<ad> result = zeros<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

        // A batch has fewer RNG lanes than the scene's full-image sampler.
        Sampler sampler;
        sampler.m_base_seed = scene.m_samplers[0].m_base_seed;
        UInt64C indices = repeat(UInt64C(detach<false>(pix_id)), opts.spp) * opts.spp
            + arange<UInt64C>(num_samples) % opts.spp;
        sampler.seed_indices(indices + seed, indices);

        Int<ad> repeat_idx = arange<Int<ad>>(num_samples) / opts.spp;
        Vector2f<ad> sampled_pixels(pix_id % opts.width, pix_id / opts.width);

        Vector2f<ad> samples_base = gather<Vector2f<ad>>(sampled_pixels, repeat_idx);
        // std::cout << samples_base << std::endl;
        // std::cout << sampler.next_2d<ad>() << std::endl;
        Vector2f<ad> samples = (samples_base + sampler.next_2d<ad>())
                                /ScalarVector2f(opts.width, opts.height);

        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);


        Spectrum<ad> value = Li(scene, sampler, camera_ray);
        masked(value, ~drjit::isfinite<Spectrum<ad>>(value) || drjit::isnan<Spectrum<ad>>(value)) = 0.f;
        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, result[j], value[j], repeat_idx);
        }
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
    }
    
    return result;
}


void Integrator::render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const RenderOption &opts = scene.m_opts;
    const Sensor *sensor = scene.m_sensors[sensor_id];
    if ( sensor->m_enable_edges ) {
        PrimaryEdgeSample edge_samples = sensor->sample_primary_edge(scene.m_samplers[1].next_1d<false>());
        MaskC valid = (edge_samples.idx >= 0);
        SpectrumC delta_L = Li(scene, scene.m_samplers[1], edge_samples.ray_n, valid) -
                            Li(scene, scene.m_samplers[1], edge_samples.ray_p, valid);
        SpectrumD value = edge_samples.x_dot_n*SpectrumD(delta_L/edge_samples.pdf);
        masked(value, ~drjit::isfinite<SpectrumD>(value)) = 0.f;
        if ( likely(opts.sppe > 1) ) {
            value /= static_cast<float>(opts.sppe);
        }
        value -= detach<false>(value);

        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, result[j], value[j], IntD(edge_samples.idx), MaskD(valid));
        }
    }
}

NAMESPACE_END(psdr_jit)
