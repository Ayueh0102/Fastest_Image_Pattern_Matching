// SPDX-License-Identifier: BSD-2-Clause
#include "coarse_internal.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using fpm::detail::ScoreMap;
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
template<class Function> void rejects(Function function)
{
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid asynchronous lifecycle operation was accepted");
}
fpm::Image noise(int width, int height, unsigned seed)
{
    fpm::Image image(width, height); std::mt19937 rng(seed);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) image.row(y)[x] = uint8_t(rng());
    return image;
}
void paste(fpm::Image& source, const fpm::Image& templ, int x, int y)
{
    for (int row = 0; row < templ.height(); ++row)
        std::copy_n(templ.row(row), templ.width(), source.row(y+row)+x);
}
struct Raster {
    int width, height; size_t stride; std::vector<uint8_t> pixels;
    Raster(int w, int h, unsigned seed) : width(w), height(h), stride(size_t(w)+7), pixels(stride*h) {
        std::mt19937 rng(seed); for (auto& pixel : pixels) pixel = uint8_t(rng());
    }
    fpm::ImageView view() const { return {pixels.data(), width, height, stride}; }
};
std::vector<ScoreMap> reference(const std::vector<fpm::ImageView>& sources, const fpm::TemplateLevel& templ)
{
    std::vector<ScoreMap> result;
    for (const auto source : sources) result.push_back(fpm::detail::ncc(source, templ, fpm::Backend::Scalar));
    return result;
}
void equal(const std::vector<ScoreMap>& actual, const std::vector<ScoreMap>& expected)
{
    require(actual.size() == expected.size(), "asynchronous batch lost or reordered maps");
    for (size_t i = 0; i < actual.size(); ++i)
        require(actual[i].width == expected[i].width && actual[i].height == expected[i].height &&
            actual[i].values == expected[i].values, "asynchronous NCC changed exact map values or order");
}
void bounded(const fpm_coarse_detail::Backend& backend)
{
    const auto m = backend.coarseOverlapMetrics();
    require(m.version == 1 && m.struct_bytes == sizeof(m), "overlap metrics ABI metadata incorrect");
    require(m.max_pending <= 2 && m.slot0_source_bytes <= 64u*1024*1024 &&
        m.slot1_source_bytes <= 64u*1024*1024 && m.slot0_statistics_bytes <= 16u*1024*1024 &&
        m.slot1_statistics_bytes <= 16u*1024*1024, "two-slot allocation or pending-work cap exceeded");
    require(m.retained_gpu_bytes == m.slot0_source_bytes+m.slot0_statistics_bytes+m.slot0_template_bytes+
        m.slot1_source_bytes+m.slot1_statistics_bytes+m.slot1_template_bytes,
        "overlap retained capacities disagree with slot breakdown");
    require(backend.metrics().gpu_bytes >= m.retained_gpu_bytes,
        "legacy GPU memory omitted overlap allocations");
}
std::unique_ptr<fpm_coarse_detail::Backend> makeBackend()
{
    auto backend = fpm_coarse_detail::makeMetalBackend();
    require(!backend->coarseOverlapEnabled() && backend->coarseOverlapMetrics().enabled == 0,
        "coarse overlap unexpectedly enabled by default");
    FpmCoarseExecutionConfigV1 config{}; fpm_coarse_execution_default_config(&config);
    config.batch_angles = 4; backend->setExecution(config); backend->setCoarseOverlap(true);
    return backend;
}
void inputOwnershipAndReuse()
{
    auto backend = makeBackend();
    // Include uint32 and uint64 statistics; each map has a distinct shape and
    // padded row stride. Replacing the template preserves its object address.
    for (int extent : {5, 258}) {
        std::vector<Raster> owners;
        for (int i = 0; i < 4; ++i) owners.emplace_back(extent+5+i, extent+7+i, 561+i);
        std::vector<fpm::ImageView> first{owners[0].view(), owners[1].view()};
        std::vector<fpm::ImageView> second{owners[2].view(), owners[3].view()};
        auto templ = fpm::detail::learnLevel(noise(extent, extent, 881));
        uint64_t retained = 0;
        for (int pass = 0; pass < 3; ++pass) {
            backend->resetMetrics();
            const auto expectedFirst = reference(first, templ);
            backend->beginCoarseBatch(0, first, templ);
            // The two pending slots must also own different template contents.
            templ = fpm::detail::learnLevel(noise(extent, extent, 992+pass));
            const auto expectedSecond = reference(second, templ);
            backend->beginCoarseBatch(1, second, templ);
            auto pending = backend->coarseOverlapMetrics();
            require(pending.submitted_batches == 2 && pending.submitted_maps == 4 &&
                pending.max_pending == 2 && pending.submissions_with_pending == 1,
                "test did not submit two unconsumed GPU batches");
            rejects([&] { backend->beginCoarseBatch(0, first, templ); });
            rejects([&] { backend->beginCoarseBatch(2, first, templ); });
            rejects([&] { backend->finishCoarseBatch(1); }); // No out-of-order consumption.
            rejects([&] { backend->setCoarseOverlap(false); });
            // begin must have copied every caller-owned input needed by GPU
            // work and later CPU normalization, including template mean/norm.
            for (auto& owner : owners) for (auto& pixel : owner.pixels) pixel = uint8_t(255-pixel);
            templ = fpm::detail::learnLevel(noise(extent, extent, 1151+pass));
            equal(backend->finishCoarseBatch(0), expectedFirst);
            equal(backend->finishCoarseBatch(1), expectedSecond);
            rejects([&] { backend->finishCoarseBatch(0); });
            rejects([&] { backend->finishCoarseBatch(2); });
            const auto completed = backend->coarseOverlapMetrics();
            require(completed.completed_batches == 2 && completed.fallback_batches == 0,
                "completed asynchronous batches or fallback counters incorrect");
            require(backend->metrics().score_maps == 4 &&
                (extent == 5 ? backend->metrics().integer32_maps : backend->metrics().integer64_maps) == 4,
                "asynchronous statistics width/map counters incorrect");
            bounded(*backend);
            if (pass) require(completed.retained_gpu_bytes == retained,
                "same-shape repeated batches continually grow GPU buffers");
            retained = completed.retained_gpu_bytes;
        }
    }
}
void cancellationAndDestruction()
{
    auto backend = makeBackend();
    auto templ = fpm::detail::learnLevel(noise(7, 9, 207));
    auto source = noise(43, 47, 208);
    const std::vector<fpm::ImageView> sources{source.view(), source.view()};
    const auto expected = reference(sources, templ);
    backend->beginCoarseBatch(0, sources, templ);
    backend->beginCoarseBatch(1, sources, templ);
    backend->drainCoarseBatches();
    require(backend->coarseOverlapMetrics().drained_batches == 2, "cancellation did not drain both commands");
    rejects([&] { backend->finishCoarseBatch(0); });
    rejects([&] { backend->finishCoarseBatch(1); });
    backend->drainCoarseBatches(); // Idempotent cleanup on exceptional exits.
    backend->beginCoarseBatch(0, sources, templ);
    auto invalid = source.view(); invalid.data = nullptr;
    rejects([&] { backend->beginCoarseBatch(1, {invalid, source.view()}, templ); });
    equal(backend->finishCoarseBatch(0), expected);
    // A rejected submission must leave its slot reusable.
    backend->beginCoarseBatch(1, sources, templ);
    equal(backend->finishCoarseBatch(1), expected);
    bounded(*backend);
    // Caller inputs die first. Backend destruction must drain pending work,
    // including during stack unwinding, without accessing those inputs.
    try {
        auto unfinished = makeBackend();
        {
            auto temporarySource = noise(53, 61, 209);
            auto temporaryTemplate = fpm::detail::learnLevel(noise(11, 13, 210));
            const std::vector<fpm::ImageView> temporary{temporarySource.view(), temporarySource.view()};
            unfinished->beginCoarseBatch(0, temporary, temporaryTemplate);
            unfinished->beginCoarseBatch(1, temporary, temporaryTemplate);
        }
        throw std::runtime_error("test stack unwinding");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "test stack unwinding", "pending-work setup failed before destructor test");
    }
    backend->beginCoarseBatch(0, sources, templ);
    equal(backend->finishCoarseBatch(0), expected);
}
void synchronousFallbacks()
{
    auto backend = makeBackend();
    for (int kind = 0; kind < 3; ++kind) {
        auto source = noise(kind == 2 ? 1200 : 39, kind == 2 ? 1200 : 31, 223+kind);
        auto image = noise(3, 3, 236);
        if (kind == 1) std::fill_n(image.row(0), 9, uint8_t(91));
        const auto templ = fpm::detail::learnLevel(std::move(image));
        const std::vector<fpm::ImageView> sources(size_t(kind == 0 ? 1 : 2), source.view());
        const auto expected = reference(sources, templ);
        backend->resetMetrics();
        backend->beginCoarseBatch(0, sources, templ);
        equal(backend->finishCoarseBatch(0), expected);
        const auto m = backend->coarseOverlapMetrics();
        require(m.submitted_batches == 0 && m.fallback_batches == 1 && m.fallback_maps == sources.size(),
            "fallback silently discarded work or counted synchronous work as asynchronous");
        require((kind == 0 ? m.fallback_single : kind == 1 ? m.fallback_degenerate : m.fallback_oversized) == 1,
            "fallback reason was not recorded");
        bounded(*backend);
    }
    // A synchronous tail must not overwrite a pending async slot's template,
    // source pixels or statistics, even when the tail uses another template.
    auto source = noise(43, 47, 311);
    const auto asyncTemplate = fpm::detail::learnLevel(noise(7, 9, 312));
    const auto tailTemplate = fpm::detail::learnLevel(noise(11, 13, 313));
    const std::vector<fpm::ImageView> pair{source.view(), source.view()}, tail{source.view()};
    const auto expectedPair = reference(pair, asyncTemplate), expectedTail = reference(tail, tailTemplate);
    backend->resetMetrics();
    backend->beginCoarseBatch(0, pair, asyncTemplate);
    backend->beginCoarseBatch(1, tail, tailTemplate);
    equal(backend->finishCoarseBatch(0), expectedPair);
    equal(backend->finishCoarseBatch(1), expectedTail);
    require(backend->coarseOverlapMetrics().submitted_batches == 1 &&
        backend->coarseOverlapMetrics().fallback_single == 1, "mixed asynchronous/synchronous tail was not exercised");
    bounded(*backend);
}
void sameWork(const fpm::detail::MatchStageProfile& a, const fpm::detail::MatchStageProfile& b)
{
#define SAME(field) require(a.field == b.field, "overlap changed matcher work: " #field)
    SAME(coarse_angles); SAME(coarse_maps); SAME(coarse_windows); SAME(coarse_candidates);
    SAME(fine_layer_visits); SAME(fine_passes); SAME(fine_rotations); SAME(fine_maps); SAME(fine_windows);
    SAME(refined_candidates); SAME(output_matches);
#undef SAME
    const double times[] = {a.source_copy_invert_ms, a.pyramid_ms, a.coarse_rotation_ms, a.coarse_ncc_ms,
        a.peak_selection_ms, a.fine_rotation_ms, a.fine_ncc_ms, a.fine_control_ms, a.nms_ms, a.unclassified_ms};
    double total = 0;
    for (double value : times) {
        require(std::isfinite(value) && value >= 0, "overlap produced an invalid exclusive profile interval");
        total += value;
    }
    require(std::abs(total-a.core_wall_ms) <= 1e-6*std::max(1., a.core_wall_ms),
        "overlap profile intervals no longer sum to core wall time");
}
void matchingOrder()
{
    auto backend = makeBackend();
    for (int variant = 0; variant < 4; ++variant) {
        auto templ = noise(19+2*variant, 17+variant, 890+variant);
        auto source = noise(173, 151, 780+variant);
        paste(source, templ, 9, 13); paste(source, templ, 113, 89);
        auto turned = fpm::detail::rotateCanvas(templ.view(), templ.width(), templ.height(), 17.25, 0);
        paste(source, turned.image, 59, 53);
        fpm::Pattern pattern(templ.view(), 64);
        fpm::MatchConfig config; config.maxTargets = 8; config.minScore = .5; config.maxOverlap = .25;
        config.toleranceAngle = 90; config.subpixel = true; config.stopLayer = variant%2;
        if (variant == 2) config.angleRanges = {{-90, -5}, {0, 0}, {7, 90}};
        if (variant == 3) config.toleranceAngle = 0; // Singleton/tail scheduler fallback.
        fpm::detail::MatchStageProfile expectedProfile{}, actualProfile{};
        backend->setCoarseOverlap(false); backend->resetMetrics();
        const auto expected = fpm::detail::matchWithCoarseNcc(source.view(), pattern, config, backend.get(), &expectedProfile);
        backend->setCoarseOverlap(true); backend->resetMetrics();
        const auto actual = fpm::detail::matchWithCoarseNcc(source.view(), pattern, config, backend.get(), &actualProfile);
        require(!expected.empty() && actual.size() == expected.size() &&
            std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(fpm::Match)) == 0,
            "overlap changed ordered final matches, scores or geometry");
        sameWork(actualProfile, expectedProfile); bounded(*backend);
        const auto m = backend->coarseOverlapMetrics();
        if (variant < 3) require(m.submitted_batches >= 2 && m.max_pending == 2,
            "full matcher failed to pipeline more than one coarse angle batch");
        require(m.submitted_batches == m.completed_batches && m.drained_batches == 0,
            "successful matcher left work pending or discarded a batch");
    }
}
struct Frame {
    fpm_frame_handle* handle = nullptr;
    char error[1024]{};
    explicit Frame(bool metal) {
        FpmFrameConfigV1 config{}; fpm_frame_default_config(&config);
        if (!metal) { config.backend = 0; config.pyramid_mode = 0; config.batch_angles = 1; }
        require(fpm_frame_create(&config, sizeof(config), &handle, error, sizeof(error)) == FPM_OK, error);
    }
    ~Frame() { fpm_frame_destroy(handle); }
    int set(const FpmCoarseOverlapConfigV1& config) {
        return fpm_frame_set_coarse_overlap(handle, &config, sizeof(config), error, sizeof(error));
    }
};
void configurationErrors()
{
    static_assert(sizeof(FpmCoarseOverlapConfigV1) == 32 && sizeof(FpmCoarseOverlapMetricsV1) == 240,
        "coarse overlap ABI changed");
    for (bool metal : {false, true}) {
        Frame frame(metal); FpmCoarseOverlapMetricsV1 metrics{};
        require(fpm_frame_coarse_overlap_metrics(frame.handle, &metrics, sizeof(metrics), frame.error, sizeof(frame.error)) == FPM_OK &&
            metrics.version == 1 && metrics.struct_bytes == sizeof(metrics) && metrics.enabled == 0 &&
            metrics.submitted_batches == 0 && metrics.retained_gpu_bytes == 0, "frame overlap defaults are not idle/off");
        FpmCoarseOverlapConfigV1 valid{1, sizeof(FpmCoarseOverlapConfigV1), 0, 0, 0, 0};
        require(frame.set(valid) == FPM_OK, frame.error);
        for (int kind = 0; kind < 7; ++kind) {
            auto invalid = valid;
            if (kind == 0) invalid.version = 2;
            if (kind == 1) invalid.struct_bytes = sizeof(invalid)-1;
            if (kind == 2) invalid.enabled = -1;
            if (kind == 3) invalid.enabled = 2;
            if (kind == 4) invalid.reserved0 = 1;
            if (kind == 5) invalid.reserved1 = 1;
            if (kind == 6) invalid.reserved2 = 1;
            require(frame.set(invalid) == FPM_ERROR, "invalid overlap config accepted");
        }
        require(fpm_frame_set_coarse_overlap(frame.handle, nullptr, sizeof(valid), nullptr, 0) == FPM_ERROR &&
            fpm_frame_set_coarse_overlap(frame.handle, &valid, sizeof(valid)-1, nullptr, 0) == FPM_ERROR &&
            fpm_frame_set_coarse_overlap(nullptr, &valid, sizeof(valid), nullptr, 0) == FPM_ERROR,
            "invalid overlap setter pointers/size accepted");
        metrics.version = 719;
        require(fpm_frame_coarse_overlap_metrics(frame.handle, &metrics, sizeof(metrics)-1, nullptr, 0) == FPM_ERROR &&
            metrics.version == 719 &&
            fpm_frame_coarse_overlap_metrics(frame.handle, nullptr, sizeof(metrics), nullptr, 0) == FPM_ERROR &&
            fpm_frame_coarse_overlap_metrics(nullptr, &metrics, sizeof(metrics), nullptr, 0) == FPM_ERROR,
            "invalid overlap metrics pointers/size accepted or output overwritten");
        valid.enabled = 1;
        require(frame.set(valid) == (metal ? FPM_OK : FPM_ERROR), "CPU accepted overlap or Metal rejected it");
        require(fpm_frame_coarse_overlap_metrics(frame.handle, &metrics, sizeof(metrics), frame.error, sizeof(frame.error)) == FPM_OK &&
            metrics.enabled == int(metal), "overlap option was not retained");
        valid.enabled = 0; require(frame.set(valid) == FPM_OK, frame.error);
    }
}
}
int main()
{
    try {
        configurationErrors(); inputOwnershipAndReuse(); cancellationAndDestruction();
        synchronousFallbacks(); matchingOrder();
        std::cout << "Coarse overlap: two pending batches, owned inputs, exact ordering, drain/reuse, fallback and bounded memory passed\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string message(error.what());
        if (message.find("is unavailable") != std::string::npos) { std::cout << "SKIP: " << message << '\n'; return 77; }
        std::cerr << message << '\n'; return 1;
    }
}
