// SPDX-License-Identifier: BSD-2-Clause
// Times one fixture through the research frame API on one backend, in its own
// process.
//
// usage: fpm_frame_bench BACKEND PYRAMID FINE_BATCH NCC_MODE MODE REPEAT WARMUP_MS
//            SRC.pgm TMPL.pgm MAX_TARGETS MIN_SCORE TOLERANCE MAX_OVERLAP MIN_AREA [TRANSFER]
// MODE is exhaustive, fast (stop_layer 1) or subpixel. TRANSFER is the frame's
// transfer mode (fpm_frame_set_transfer_mode, default 0). Image decoding, frame
// creation (which creates any GPU context) and pattern learning stay outside the
// timed calls. A call is fpm_frame_prepare (source copy and pyramid) followed by
// fpm_frame_match, each timed separately. The first call is reported on its own
// (cold); untimed calls then repeat until WARMUP_MS has passed (a GPU left idle
// runs at low clocks), and REPEAT timed calls follow. Prints one JSON object with
// those times, the last call's backend metrics, the process peak RSS, and the
// matches as the hex bits of their IEEE-754 doubles.
#include "coarse_ncc.h"
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

struct Gray { int width = 0, height = 0; std::vector<std::uint8_t> pixels; };
Gray readPgm(const char* path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::string data((std::istreambuf_iterator<char>(in)), {});
    std::size_t pos = 0;
    std::string fields[4];
    for (auto& field : fields) {
        while (pos < data.size() && (data[pos] == '#' || std::strchr(" \t\r\n", data[pos])))
            pos = data[pos] == '#' ? data.find('\n', pos) + 1 : pos + 1;
        const std::size_t end = data.find_first_of(" \t\r\n", pos);
        field = data.substr(pos, end - pos);
        pos = end;
    }
    if (fields[0] != "P5" || fields[3] != "255") throw std::runtime_error(std::string("expected an 8-bit P5 PGM: ") + path);
    Gray g{std::stoi(fields[1]), std::stoi(fields[2]), {}};
    if (data.size() < pos + 1 + std::size_t(g.width) * g.height) throw std::runtime_error("truncated PGM");
    g.pixels.assign(data.begin() + std::ptrdiff_t(pos + 1), data.begin() + std::ptrdiff_t(pos + 1 + std::size_t(g.width) * g.height));
    return g;
}
fpm_image view(const Gray& g) { return {g.pixels.data(), g.width, g.height, std::size_t(g.width)}; }

double peakRssMiB()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) return -1;
    return double(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) return -1;
#if defined(__APPLE__)
    return double(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return double(usage.ru_maxrss) / 1024.0;
#endif
#endif
}

void bits(std::string& out, double value)
{
    std::uint64_t u;
    std::memcpy(&u, &value, sizeof(u));
    char text[24];
    std::snprintf(text, sizeof text, "\"%016" PRIx64 "\"", u);
    out += text;
}
std::string list(const std::vector<double>& values)
{
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        char text[32];
        std::snprintf(text, sizeof text, "%s%.4f", i ? "," : "", values[i]);
        out += text;
    }
    return out + "]";
}
}

int main(int argc, char** argv)
{
    if (argc != 15 && argc != 16) {
        std::fprintf(stderr, "usage: fpm_frame_bench BACKEND PYRAMID FINE_BATCH NCC_MODE MODE REPEAT WARMUP_MS "
            "SRC.pgm TMPL.pgm MAX_TARGETS MIN_SCORE TOLERANCE MAX_OVERLAP MIN_AREA [TRANSFER]\n");
        return 2;
    }
    char error[1024]{};
    try {
        const int backend = std::atoi(argv[1]), pyramid = std::atoi(argv[2]), fineBatch = std::atoi(argv[3]);
        const int nccMode = std::atoi(argv[4]), repeat = std::atoi(argv[6]);
        const double warmupMs = std::atof(argv[7]);
        const std::string mode = argv[5];
        const Gray source = readPgm(argv[8]), templ = readPgm(argv[9]);
        fpm_config config{};
        config.max_targets = std::atoi(argv[10]); config.min_score = std::atof(argv[11]);
        config.tolerance_angle = std::atof(argv[12]); config.max_overlap = std::atof(argv[13]);
        config.subpixel = mode == "subpixel"; config.stop_layer = mode == "fast";
        if (mode != "exhaustive" && mode != "fast" && mode != "subpixel") throw std::invalid_argument("unknown mode " + mode);
        const int minArea = std::atoi(argv[14]);
        const int transfer = argc == 16 ? std::atoi(argv[15]) : 0;

        fpm_coarse_handle* pattern = nullptr;
        FpmCoarseMetrics metrics{};
        if (fpm_coarse_create_backend(view(templ), minArea, 0, &pattern, &metrics, error, sizeof error) != FPM_OK)
            throw std::runtime_error(error);
        FpmFrameConfigV1 frameConfig{};
        fpm_frame_default_config(&frameConfig);
        frameConfig.backend = backend; frameConfig.pyramid_mode = pyramid;
        frameConfig.batch_angles = 1; frameConfig.fine_batch_size = fineBatch;
        fpm_frame_handle* frame = nullptr;
        if (fpm_frame_create(&frameConfig, sizeof frameConfig, &frame, error, sizeof error) != FPM_OK) throw std::runtime_error(error);
        if (nccMode) {
            FpmCoarseOptimizationConfigV1 optimization{1, sizeof(FpmCoarseOptimizationConfigV1), 0, nccMode, 0, 0};
            if (fpm_frame_set_coarse_optimization(frame, &optimization, sizeof optimization, error, sizeof error) != FPM_OK)
                throw std::runtime_error(error);
        }
        if (transfer && fpm_frame_set_transfer_mode(frame, transfer, error, sizeof error) != FPM_OK)
            throw std::runtime_error(error);
        std::vector<double> prepareMs, matchMs;
        std::vector<fpm_result> results(1024);
        std::size_t count = 0;
        FpmFrameMetricsV1 frameMetrics{};
        FpmCoarseExecutionMetricsV1 execution{};
        FpmFineMetricsV1 fine{};
        auto call = [&] {
            auto start = Clock::now();
            if (fpm_frame_prepare(frame, view(source), &pattern, 1, 0, &frameMetrics, error, sizeof error) != FPM_OK)
                throw std::runtime_error(error);
            const double prepare = elapsed(start);
            start = Clock::now();
            if (fpm_frame_match(frame, pattern, &config, results.data(), results.size(), &count, &metrics, nullptr, 0,
                &execution, sizeof execution, &fine, sizeof fine, error, sizeof error) != FPM_OK)
                throw std::runtime_error(error);
            return std::pair<double, double>(prepare, elapsed(start));
        };
        const auto first = call();
        int warmupCalls = 0;
        for (const auto warmup = Clock::now(); elapsed(warmup) < warmupMs; ++warmupCalls) call();
        for (int r = 0; r < repeat; ++r) {
            const auto timed = call();
            prepareMs.push_back(timed.first); matchMs.push_back(timed.second);
        }
        std::string matches = "[";
        for (std::size_t i = 0; i < count; ++i) {
            const auto& m = results[i];
            matches += i ? ",[" : "[";
            const double values[] = {m.corners[0].x, m.corners[0].y, m.corners[1].x, m.corners[1].y, m.corners[2].x,
                m.corners[2].y, m.corners[3].x, m.corners[3].y, m.center.x, m.center.y, m.angle, m.score};
            for (std::size_t k = 0; k < 12; ++k) { if (k) matches += ","; bits(matches, values[k]); }
            matches += "]";
        }
        matches += "]";
        FpmTransferMetricsV1 transferMetrics{};
        if (fpm_frame_get_transfer_metrics(frame, &transferMetrics, sizeof transferMetrics, error, sizeof error) != FPM_OK)
            throw std::runtime_error(error);
        std::printf("{\"backend\":%d,\"pyramid\":%d,\"fine_batch\":%d,\"ncc_mode\":%d,\"mode\":\"%s\","
            "\"first_prepare_ms\":%.4f,\"first_match_ms\":%.4f,\"warmup_ms\":%.1f,\"warmup_calls\":%d,"
            "\"prepare_ms\":%s,\"match_ms\":%s,\"peak_rss_mib\":%.2f,"
            "\"coarse\":{\"wall_ms\":%.4f,\"device_ms\":%.4f,\"wait_ms\":%.4f,\"select_ms\":%.4f,\"maps\":%" PRIu64
            ",\"windows\":%" PRIu64 ",\"syncs\":%" PRIu64 ",\"gpu_bytes\":%" PRIu64 "},"
            "\"fine\":{\"wall_ms\":%.4f,\"device_ms\":%.4f,\"wait_ms\":%.4f,\"calls\":%" PRIu64 ",\"maps\":%" PRIu64
            ",\"source_uploads\":%" PRIu64 ",\"repair_pixels\":%" PRIu64 ",\"repair_all_maps\":%" PRIu64
            ",\"roi_pixels\":%" PRIu64 ",\"retained_gpu_bytes\":%" PRIu64 "},"
            "\"pyramid\":{\"prepare_wall_ms\":%.4f,\"pyramid_ms\":%.4f,\"source_copy_ms\":%.4f},"
            "\"transfer\":{\"mode\":%d,\"source_registered\":%d,\"prepare_copy_ms\":%.4f,\"prepare_enqueue_ms\":%.4f,"
            "\"signal_wait_ms\":%.4f,\"lean_prepares\":%" PRIu64 ",\"upload_chunks\":%" PRIu64 ",\"registered_bytes\":%" PRIu64
            ",\"host_levels_skipped_bytes\":%" PRIu64 ",\"host_level_rebuilds\":%" PRIu64 ",\"signal_waits\":%" PRIu64
            ",\"signal_queries\":%" PRIu64 "},"
            "\"count\":%zu,\"matches\":%s}\n",
            backend, pyramid, fineBatch, nccMode, mode.c_str(), first.first, first.second, warmupMs, warmupCalls,
            list(prepareMs).c_str(), list(matchMs).c_str(), peakRssMiB(),
            metrics.wall_ms, metrics.device_ms, metrics.wait_ms, metrics.normalization_ms, metrics.score_maps,
            metrics.windows, metrics.command_buffers, metrics.gpu_bytes,
            fine.wall_ms, fine.device_ms, fine.wait_ms, fine.calls, fine.maps, fine.source_uploads, fine.repair_pixels, fine.repair_all_maps, fine.roi_pixels,
            fine.retained_gpu_bytes,
            frameMetrics.prepare_wall_ms, frameMetrics.pyramid_ms, frameMetrics.source_copy_ms,
            transferMetrics.mode, transferMetrics.source_registered, transferMetrics.prepare_copy_ms,
            transferMetrics.prepare_enqueue_ms, transferMetrics.signal_wait_ms, transferMetrics.lean_prepares,
            transferMetrics.upload_chunks, transferMetrics.registered_bytes, transferMetrics.host_levels_skipped_bytes,
            transferMetrics.host_level_rebuilds, transferMetrics.signal_waits, transferMetrics.signal_queries,
            count, matches.c_str());
        fpm_frame_destroy(frame);
        fpm_coarse_destroy(pattern);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fpm_frame_bench: %s\n", e.what());
        return 1;
    }
    return 0;
}
