// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include "metal_device.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
fpm::Image raster(int width, int height, unsigned seed, int mode = 0)
{
    fpm::Image image(width, height); std::mt19937 generator(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        image.row(y)[x] = mode == 1 ? uint8_t(((x+y)&1)*255) : mode == 2 ? uint8_t(131) : uint8_t(generator());
    return image;
}
void compare(fpm_coarse_detail::Backend& backend, fpm::ImageView source,
    const fpm::TemplateLevel& templ, const std::vector<fpm::detail::FineRequest>& poses)
{
    double rotation = -1;
    const auto maps = backend.computeFineBatch(source, templ, poses, &rotation);
    require(std::isfinite(rotation) && rotation >= 0, "invalid fine rotation timing");
    require(maps.size() == poses.size(), "fine map count differs");
    for (size_t i = 0; i < poses.size(); ++i) {
        const auto roi = fpm::detail::rotatedRoi(source, templ.image.width(), templ.image.height(), poses[i].anchor, poses[i].angle);
        const auto expected = fpm::detail::ncc(roi.view(), templ, fpm::Backend::Scalar);
        if (maps[i].values != expected.values) {
            std::cerr << "Map " << i << ", anchor " << poses[i].anchor.x << ',' << poses[i].anchor.y
                << ", angle " << poses[i].angle << ", template " << templ.image.width() << 'x' << templ.image.height() << '\n';
            for (size_t j = 0; j < expected.values.size(); ++j)
                if (maps[i].values[j] != expected.values[j]) { std::cerr << "First unequal window " << j << '\n'; break; }
            throw std::runtime_error("guarded Metal warp/NCC differs from CPU double warp and integer NCC");
        }
        require(maps[i].width == expected.width && maps[i].height == expected.height, "fine map dimensions differ");
    }
    const auto m = backend.fineMetrics();
    require(m.version == 1 && m.struct_bytes == sizeof(m), "invalid fine metrics metadata");
    require(m.maps >= poses.size() && m.repair_pixels <= m.roi_pixels &&
        m.peak_batch_maps <= backend.fineBatchSize(), "fine batch/repair counters invalid");
    require(m.retained_gpu_bytes == m.source_buffer_bytes+m.roi_buffer_bytes+m.guard_buffer_bytes+
        m.descriptor_buffer_bytes+m.statistics_buffer_bytes+m.template_buffer_bytes, "fine retained memory counts differ");
}
}
int main(int argc, char** argv)
{
    if (!fpm_test::metalDeviceAvailable()) { std::cout << "SKIP: no Metal device\n"; return 77; }
    try {
        auto backend = fpm_coarse_detail::makeMetalBackend();
        int mode = 0;
        const bool fused = argc == 3 && std::string(argv[2]) == "--fused";
        if (argc == 2 || fused) {
            const std::string argument = argv[1];
            if (argument == "--original") mode = 0;
            else if (argument == "--shared") mode = 1;
            else if (argument == "--strip32") mode = 2;
            else if (argument == "--strip16") mode = 3;
            else throw std::invalid_argument("unknown fine NCC test mode");
        } else if (argc != 1) throw std::invalid_argument("expected at most one fine NCC test mode");
        const bool shared = mode == 1;
        backend->setFineNccMode(mode);
        backend->setFineFused(fused);
        backend->setFineBatchSize(7);
        FpmCoarseExecutionConfigV1 config{}; fpm_coarse_execution_default_config(&config);
        std::mt19937 generator(81731); std::uniform_real_distribution<double> angle(-180, 180);
        bool repaired = false, used64 = false;
        for (int pyramidMode : {0, 1, 2}) {
            config.pyramid_mode = pyramidMode; backend->setExecution(config);
            for (int texture : {0, 1, 2}) {
                std::vector<fpm::Image> levels; levels.push_back(raster(513, 389, 117, texture));
                backend->invalidateFineSource(); backend->resetMetrics();
                if (!backend->buildPyramid(levels, 2))
                    for (int i = 0; i < 2; ++i) levels.push_back(fpm::detail::pyrDown(levels.back().view()));
                backend->beginFineMatch(levels);
                for (int layer : {0, 2, 1, 0}) {
                    auto& image = levels[layer];
                    const auto templ = fpm::detail::learnLevel(raster(19, 17, 481));
                    std::vector<fpm::detail::FineRequest> requests;
                    for (double a : {0., 90., 180., -90., 45., -179.999999999, 0.000000001})
                        for (double offset : {-1.000000001, -.5, 0., .5})
                            requests.push_back({{offset, offset}, a});
                    for (int i = 0; i < 21; ++i)
                        requests.push_back({{double(generator()%image.width())+.5,
                                             double(generator()%image.height())+.125}, angle(generator)});
                    requests.push_back({{double(image.width())-.000000001, double(image.height())+.5}, 0.});
                    compare(*backend, image.view(), templ, requests);
                    repaired = repaired || backend->fineMetrics().repair_pixels > 0;
                }
                if (pyramidMode == 0 && texture == 0) {
                    // Explicit invalidation refreshes pixels at identical addresses.
                    auto* address = levels[0].row(0);
                    for (int y = 0; y < levels[0].height(); ++y) for (int x = 0; x < levels[0].width(); ++x)
                        levels[0].row(y)[x] = uint8_t(255-levels[0].row(y)[x]);
                    backend->invalidateFineSource(); backend->beginFineMatch(levels);
                    require(address == levels[0].row(0), "test did not reuse source allocation");
                    const auto templ = fpm::detail::learnLevel(raster(258, 258, 139));
                    compare(*backend, levels[0].view(), templ, {{{64.5, 32.5}, 13.75}, {{-4.5, 9.125}, -37.}});
                    used64 = true;
                    const auto flat = fpm::detail::learnLevel(raster(3, 3, 1, 2));
                    compare(*backend, levels[0].view(), flat, {{{12.5, 19.5}, 0.}});
                }
            }
        }
        // Narrow dimensions exercise zero-border interpolation independently of pyramid reflection.
        for (auto size : {std::pair<int,int>{1, 1}, {1, 17}, {23, 1}}) {
            std::vector<fpm::Image> levels; levels.push_back(raster(size.first, size.second, 817));
            backend->invalidateFineSource(); backend->beginFineMatch(levels);
            const auto templ = fpm::detail::learnLevel(raster(2, 3, 341));
            compare(*backend, levels[0].view(), templ, {{{-.5, -.5}, 0.}, {{.5, .5}, 45.}});
        }
        // Exercise strip chunk tails on both sides of the 16/32-pixel
        // boundaries, a one-row template and odd heights. Fifteen maps with
        // batch size seven also leave a one-map final batch.
        for (int width : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
            for (int height : {1, 3, 17}) {
                const auto source = raster(109, 83, 913 + width + height);
                const auto templ = fpm::detail::learnLevel(raster(width, height, 517 + width));
                std::vector<fpm::detail::FineRequest> requests;
                for (int i = 0; i < 15; ++i)
                    requests.push_back({{double((i*23)%115)-3.5, double((i*17)%89)-2.125},
                        i == 0 ? 0. : i == 1 ? 90. : double(i*29%360)-180.});
                backend->invalidateFineSource(); backend->resetMetrics();
                compare(*backend, source.view(), templ, requests);
                const auto counters = backend->fineMetrics();
                const auto reuse = backend->fineReuseMetrics();
                require(counters.maps == requests.size() && counters.peak_batch_maps == 7 &&
                    counters.command_buffers == (fused ? 3 : 6) && reuse.statistics_dispatches == 3,
                    "fine strip boundary test omitted batch tail or changed submissions");
                require(shared ? reuse.row_dispatches == 3 : reuse.row_dispatches == 0,
                    "fine strip boundary test dispatched wrong statistics route");
            }
        }
        // Large, nearly saturated values force actual square/product sums
        // above UINT32_MAX, not just selection of the uint64 kernel branch.
        {
            auto source = raster(269, 271, 1, 2), patch = raster(258, 258, 1, 2);
            for (int y = 0; y < source.height(); ++y) for (int x = 0; x < source.width(); ++x)
                source.row(y)[x] = uint8_t(255-((x+3*y)%131 == 0));
            for (int y = 0; y < patch.height(); ++y) for (int x = 0; x < patch.width(); ++x)
                patch.row(y)[x] = uint8_t(255-((x+3*y)%131 == 0));
            const auto templ = fpm::detail::learnLevel(std::move(patch));
            backend->invalidateFineSource(); backend->resetMetrics();
            compare(*backend, source.view(), templ, {{{3., 3.}, 0.}, {{3.5, 3.5}, 0.}});
            compare(*backend, source.view(), templ, {{{std::numeric_limits<double>::max(), 0.}, 0.}});
            if (fused) require(backend->fineFusionMetrics().fallback_batches == 1, "repairAll did not fall back");
            require(backend->fineMetrics().repair_all_maps == 1, "out-of-range pose repair was not counted");
        }
        if (fused) {
            const auto source = raster(47, 43, 99, 1);
            const auto templ = fpm::detail::learnLevel(raster(17, 19, 104));
            backend->invalidateFineSource(); backend->resetMetrics();
            compare(*backend, source.view(), templ, {{{6.5, 7.}, 0.}});
            require(backend->fineFusionMetrics().dense_maps == 0 && backend->fineFusionMetrics().changed_pixels == 0,
                "unchanged half-integer pixels unnecessarily recomputed");
            backend->resetMetrics();
            compare(*backend, source.view(), templ, {{{6.5-1e-10, 7.}, 0.}});
            const auto d = backend->fineFusionMetrics();
            require(d.dense_maps > 0 && d.fused_batches == 1 && d.fallback_batches == 0,
                "dense uncertainty fallback omitted");
            require(backend->fineMetrics().command_buffers == 1, "dense repair added GPU wait");
            fpm::Image impulse(47,43);
            impulse.row(15)[14] = 255;
            backend->invalidateFineSource(); backend->resetMetrics();
            compare(*backend, impulse.view(), templ, {{{6.5-1e-10,7.},0.}});
            const auto sparse = backend->fineFusionMetrics();
            require(sparse.changed_pixels > 0 && sparse.corrected_windows > 0 && sparse.dense_maps == 0,
                "sparse actual-change correction omitted");
            backend->resetMetrics();
            compare(*backend, impulse.view(), templ, {{{6.5-1e-10,7.},0.},
                {{std::numeric_limits<double>::max(),0.},0.}});
            require(backend->fineFusionMetrics().fallback_batches == 1 &&
                backend->fineFusionMetrics().fused_batches == 0 && backend->fineMetrics().command_buffers == 2,
                "mixed repairAll batch did not use exact legacy fallback");
            backend->setFineFused(false); backend->resetMetrics();
            compare(*backend, source.view(), templ, {{{6.5, 7.}, 0.}});
            require(!backend->fineFusionMetrics().enabled && backend->fineMetrics().command_buffers == 2,
                "fusion mode switch failed");
            backend->setFineFused(true);
        }
        require(repaired && used64, "test omitted uncertainty repair or uint64 NCC coverage");
        // Adjacent template sizes straddle the exact uint32 square bound.
        for (int height : {257,258}) {
            auto source = raster(269,271,31), patch = raster(257,height,32);
            for (int y=0;y<271;++y) for (int x=0;x<269;++x) source.row(y)[x]=255-((x+3*y)%131==0);
            for (int y=0;y<height;++y) for (int x=0;x<257;++x) patch.row(y)[x]=255-((x+3*y)%131==0);
            backend->invalidateFineSource(); backend->resetMetrics();
            compare(*backend, source.view(), fpm::detail::learnLevel(std::move(patch)), {{{3.,3.},0.},{{3.5,3.5},0.}});
        }
        const auto reuse = backend->fineReuseMetrics();
        require(reuse.version == 1 && reuse.struct_bytes == sizeof(reuse) && reuse.ncc_mode == mode, "invalid fine reuse metadata");
        require(reuse.statistics_dispatches > 0 && backend->fineMetrics().command_buffers == (fused ? 1 : 2)*reuse.statistics_dispatches,
            "fine reuse introduced extra command submissions");
        require(shared ? (reuse.row_buffer_bytes > 0 && reuse.row_dispatches == reuse.statistics_dispatches) :
            (!reuse.row_buffer_bytes && !reuse.row_dispatches), "fine row dispatch/memory accounting incorrect");
        for (int invalid : {-1, 4}) {
            bool rejected = false;
            try { backend->setFineNccMode(invalid); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && backend->fineReuseMetrics().ncc_mode == mode, "invalid mode changed state");
        }
        backend->resetMetrics();
        const auto reset = backend->fineReuseMetrics();
        require(reset.ncc_mode == mode && reset.row_buffer_bytes == reuse.row_buffer_bytes,
            "fine metrics reset changed mode or retained row allocation");
        require(!reset.row_dispatches && !reset.statistics_dispatches && reset.stats_device_ms == (fused ? -1 : 0),
            "fine metrics reset retained dispatch counts or timing");
        if (shared) {
            // A narrow ROI stays below the 64 MiB source arena while its
            // uint64 row statistics exceed their independent 16 MiB limit.
            constexpr size_t rowArenaBytes = 16 * 1024 * 1024;
            const int height = int(rowArenaBytes / (7 * 2 * sizeof(uint64_t))) - 6 + 1;
            const auto tall = fpm::detail::learnLevel(raster(1, height, 719));
            const auto source = raster(1, 1, 817);
            bool rejected = false;
            try { backend->computeFineBatch(source.view(), tall, {}, nullptr); }
            catch (const std::invalid_argument& e) {
                rejected = std::string(e.what()).find("row statistics exceed") != std::string::npos;
            }
            require(rejected, "oversized fine row arena was not rejected");
            const auto after = backend->fineReuseMetrics();
            require(after.row_buffer_bytes == reset.row_buffer_bytes && !after.row_dispatches &&
                !after.statistics_dispatches, "oversized fine row arena allocated or dispatched work");
        }
        for (int stripMode : {2, 3}) {
            const auto source = raster(109, 83, 712);
            const auto templ = fpm::detail::learnLevel(raster(33, 17, 713));
            const std::vector<fpm::detail::FineRequest> requests = {{{12.5, 19.125}, 13.}, {{-.5, -.5}, -91.}};
            backend->setFineNccMode(1); backend->resetMetrics();
            compare(*backend, source.view(), templ, requests);
            require(backend->fineReuseMetrics().row_buffer_bytes > 0, "mode switch test omitted row allocation");
            backend->setFineNccMode(stripMode);
            require(backend->fineReuseMetrics().row_buffer_bytes == 0, "strip mode retains unused row buffer");
            backend->resetMetrics();
            compare(*backend, source.view(), templ, requests);
            const auto strip = backend->fineReuseMetrics();
            require(strip.ncc_mode == stripMode && !strip.row_buffer_bytes && !strip.row_dispatches &&
                strip.statistics_dispatches == 1 && backend->fineMetrics().command_buffers == (fused ? 1 : 2),
                "switch from row reuse to strip mode changed dispatch/memory contract");
        }
        backend->setFineNccMode(0);
        require(backend->fineReuseMetrics().row_buffer_bytes == 0, "disabled fine reuse retains row buffer");
        for (int device : {0,1}) {
            FpmFrameConfigV1 cfg{};fpm_frame_default_config(&cfg);
            cfg.backend=device;cfg.pyramid_mode=0;cfg.batch_angles=1;cfg.fine_batch_size=device?7:0;
            fpm_frame_handle* frame=nullptr;char error[1024]{};
            require(fpm_frame_create(&cfg,sizeof(cfg),&frame,error,sizeof(error))==FPM_OK,"fine API frame creation failed");
            FpmFineReuseMetricsV1 metrics{};
            for (int apiMode : {1,2,3}) {
                require(fpm_frame_set_fine_ncc_mode(frame,apiMode,error,sizeof(error))==(device?FPM_OK:FPM_ERROR),"fine API backend gate failed");
                require(fpm_frame_get_fine_reuse_metrics(frame,&metrics,sizeof(metrics),error,sizeof(error))==FPM_OK && metrics.ncc_mode==(device?apiMode:0),"fine API metrics mode differs");
            }
            require(fpm_frame_get_fine_reuse_metrics(frame,&metrics,sizeof(metrics)-1,error,sizeof(error))==FPM_ERROR,"short metrics accepted");
            require(fpm_frame_get_fine_reuse_metrics(frame,nullptr,sizeof(metrics),error,sizeof(error))==FPM_ERROR,"null metrics accepted");
            for (int invalid : {-1,4}) {
                require(fpm_frame_set_fine_ncc_mode(frame,invalid,error,sizeof(error))==FPM_ERROR,"bad mode accepted");
                require(fpm_frame_get_fine_reuse_metrics(frame,&metrics,sizeof(metrics),error,sizeof(error))==FPM_OK && metrics.ncc_mode==(device?3:0),"bad mode changed API state");
            }
            require(fpm_frame_set_fine_ncc_mode(frame,0,error,sizeof(error))==FPM_OK,"mode reset failed");
            fpm_frame_destroy(frame);
        }
        require(fpm_frame_set_fine_ncc_mode(nullptr,0,nullptr,0)==FPM_ERROR,"null frame accepted");
        std::cout << "Guarded Q28 Metal warp and batched NCC equal CPU at borders, half pixels, large/flat templates and changed frames\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
