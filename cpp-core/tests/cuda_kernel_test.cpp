// SPDX-License-Identifier: BSD-2-Clause
// Kernel-level checks of the CUDA backend against fpm_core in the same build.
// Fine maps must equal ncc(rotatedRoi()) for random, border, out-of-image and
// repairAll poses, for poses found here on the host where exact Q28 sampling
// rounds a pixel differently from binary64 (the guard must send those pixels to
// FP64, and each such pixel must lie inside the guard), and, with every pixel
// sampled in FP64, for poses where re-associating or contracting warpPixel()'s
// coordinate arithmetic would change a pixel. Coarse peaks must equal
// selectPeaks(ncc(rotateCanvas())) for random angle sets, borders, counts,
// overlaps and thresholds, fused and unfused; dense maps must equal ncc(); the
// device pyramid must equal pyrDown(). Exit 77 when no CUDA device is available.
#include "coarse_internal.hpp"
#include "fine_sampling.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fpm;
using namespace fpm::detail;

namespace {
void require(bool value, const std::string& text) { if (!value) throw std::runtime_error(text); }
using Rng = std::mt19937;

Image noise(int width, int height, Rng& rng, bool contrast = false)
{
    Image image(width, height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) image.row(y)[x] = std::uint8_t(contrast ? (rng() % 2) * 255 : rng() % 256);
    return image;
}
// Smooth texture, so that NCC scores spread over (-1, 1) instead of near 0.
Image texture(int width, int height, Rng& rng)
{
    Image image(width, height);
    const double fx = 0.05 + (rng() % 100) / 400.0, fy = 0.05 + (rng() % 100) / 400.0, phase = rng() % 628 / 100.0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.row(y)[x] = std::uint8_t(std::clamp(128 + 90 * std::sin(fx * x + phase) * std::cos(fy * y) + int(rng() % 31) - 15, 0.0, 255.0));
    return image;
}
bool sameBits(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// The device fast path on the host: exact Q28 bilinear numerator over zero borders.
std::uint8_t q28Pixel(ImageView s, const fpm_coarse_detail::FineWarpDescriptor& d, int x, int y, std::uint64_t* distance)
{
    constexpr long long unit = 1ll << 28;
    const long long sx = d.anchorX + d.cosine * x - d.sine * y, sy = d.anchorY + d.sine * x + d.cosine * y;
    auto pixel = [&](long long px, long long py) -> std::uint64_t {
        return px < 0 || py < 0 || px >= s.width || py >= s.height ? 0 : s.data[std::size_t(py) * s.stride + std::size_t(px)];
    };
    std::uint64_t numerator = 0;
    if (sx >= -unit && sy >= -unit && sx < (long long)s.width * unit && sy < (long long)s.height * unit) {
        const long long ix = sx >> 28, iy = sy >> 28;
        const std::uint64_t dx = std::uint64_t(sx & (unit - 1)), dy = std::uint64_t(sy & (unit - 1)), one = unit;
        const std::uint64_t top = pixel(ix, iy) * (one - dx) + pixel(ix + 1, iy) * dx;
        const std::uint64_t bottom = pixel(ix, iy + 1) * (one - dx) + pixel(ix + 1, iy + 1) * dx;
        numerator = top * (one - dy) + bottom * dy;
    }
    constexpr std::uint64_t half = std::uint64_t(1) << 55;
    const std::uint64_t remainder = numerator & ((std::uint64_t(1) << 56) - 1);
    *distance = remainder > half ? remainder - half : half - remainder;
    return std::uint8_t((numerator + half) >> 56);
}

struct Gpu {
    std::unique_ptr<fpm_coarse_detail::Backend> value;
    Gpu() : value(fpm_coarse_detail::makeCudaBackend()) {}
};

void fineCase(fpm_coarse_detail::Backend& gpu, const Image& source, const TemplateLevel& templ,
    const std::vector<FineRequest>& requests, const std::string& what)
{
    gpu.invalidateFineSource();
    gpu.beginFineMatch({});
    for (std::size_t first = 0; first < requests.size(); first += 64) {
        const std::vector<FineRequest> batch(requests.begin() + std::ptrdiff_t(first),
            requests.begin() + std::ptrdiff_t(std::min(requests.size(), first + 64)));
        const auto maps = gpu.computeFineBatch(source.view(), templ, batch, nullptr);
        require(maps.size() == batch.size(), what + ": map count");
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const auto roi = rotatedRoi(source.view(), templ.image.width(), templ.image.height(), batch[i].anchor, batch[i].angle);
            const auto expected = ncc(roi.view(), templ, fpm::Backend::Auto);
            require(maps[i].width == expected.width && maps[i].height == expected.height && sameBits(maps[i].values, expected.values),
                what + ": fine map " + std::to_string(first + i) + " differs");
        }
    }
}

// Poses whose ROI contains a pixel where Q28 and binary64 round differently.
std::vector<FineRequest> disagreeingPoses(const Image& source, int tw, int th, Rng& rng, std::size_t wanted, std::size_t& inspected)
{
    std::vector<FineRequest> found;
    std::uniform_real_distribution<double> u(0, 1);
    inspected = 0;
    for (int trial = 0; trial < 400000 && found.size() < wanted; ++trial) {
        const FineRequest request{{u(rng) * source.width(), u(rng) * source.height()}, (u(rng) * 2 - 1) * 180};
        const auto map = roiWarp(tw, th, request.anchor, request.angle);
        fpm_coarse_detail::FineWarpPlan plan{map.anchor, map.c, map.s, {}};
        fpm_coarse_detail::describeFineWarp(plan, map.width, map.height);
        if (plan.descriptor.repairAll) continue;
        bool differs = false;
        for (int y = 0; y < map.height; ++y)
            for (int x = 0; x < map.width; ++x) {
                ++inspected;
                std::uint64_t distance = 0;
                const auto fast = q28Pixel(source.view(), plan.descriptor, x, y, &distance);
                if (fast != fpm_coarse_detail::sampleFineLegacyPixel(source.view(), plan, x, y)) {
                    require(distance <= plan.descriptor.roundingGuard, "Q28 differs from binary64 outside the rounding guard");
                    differs = true;
                }
            }
        if (differs) found.push_back(request);
    }
    return found;
}

// The sampled value of warpPixel() at (sx, sy), as in sampleFineLegacyPixel().
std::uint8_t bilinear(ImageView s, double sx, double sy)
{
    if (sx < -1 || sy < -1 || sx >= s.width || sy >= s.height) return 0;
    const int ix = int(std::floor(sx)), iy = int(std::floor(sy));
    const double dx = sx - ix, dy = sy - iy;
    auto pixel = [&](int px, int py) -> double {
        return px < 0 || py < 0 || px >= s.width || py >= s.height ? 0 : s.data[std::size_t(py) * s.stride + std::size_t(px)];
    };
    const double top = pixel(ix, iy) * (1 - dx) + pixel(ix + 1, iy) * dx;
    const double bottom = pixel(ix, iy + 1) * (1 - dx) + pixel(ix + 1, iy + 1) * dx;
    return std::uint8_t(std::clamp(std::floor(top * (1 - dy) + bottom * dy + 0.5), 0.0, 255.0));
}

// Poses whose ROI has a pixel that changes when warpPixel()'s coordinates are
// re-associated or contracted into an FMA: only an exact FP64 port reproduces them.
std::vector<FineRequest> arithmeticPoses(const Image& source, int tw, int th, Rng& rng, std::size_t wanted, std::size_t& inspected)
{
    // Exact ties need structured poses: special angles (tiny or zero products) and
    // anchors on half, third and seventh grids, as in core_test's exact-reference.
    std::vector<FineRequest> candidates;
    const double angles[] = {0, 90, -90, 180, -180, 1e-9, 1e-12, -1e-12, 45, 30, 60, -37.25, 120};
    for (double angle : angles)
        for (int i = -12; i < 2 * source.width(); i += 1)
            for (int j = -12; j < 2 * source.height(); j += 3)
                for (double denominator : {2.0, 3.0, 7.0}) candidates.push_back({{i / denominator, j / denominator}, angle});
    std::shuffle(candidates.begin(), candidates.end(), rng);
    std::vector<FineRequest> found;
    inspected = 0;
    for (const auto& request : candidates) {
        if (found.size() >= wanted) break;
        const auto m = roiWarp(tw, th, request.anchor, request.angle);
        fpm_coarse_detail::FineWarpPlan plan{m.anchor, m.c, m.s, {}};
        bool differs = false;
        for (int y = 0; y < m.height && !differs; ++y)
            for (int x = 0; x < m.width && !differs; ++x) {
                ++inspected;
                const auto expected = fpm_coarse_detail::sampleFineLegacyPixel(source.view(), plan, x, y);
                const double cx = m.c * x, sy0 = m.s * y, sx1 = m.s * x, cy = m.c * y;
                const double sx = (m.anchor.x + cx) - sy0, sy = (m.anchor.y + sx1) + cy;
                const double variants[][2] = {{m.anchor.x + (cx - sy0), sy}, {sx, std::fma(m.c, double(y), m.anchor.y + sx1)},
                    {std::fma(m.c, double(x), m.anchor.x) - sy0, sy}, {sx, std::fma(m.s, double(x), m.anchor.y) + cy}};
                for (const auto& v : variants) differs |= bilinear(source.view(), v[0], v[1]) != expected;
            }
        if (differs) found.push_back(request);
    }
    return found;
}

void fine()
{
    Gpu backend;
    auto& gpu = *backend.value;
    gpu.setFineBatchSize(64);
    Rng rng(20261101);
    std::uniform_real_distribution<double> u(0, 1);
    // 1. Poses where the fast path alone would be wrong.
    {
        const auto source = noise(64, 48, rng, true);
        const auto templ = learnLevel(texture(5, 4, rng));
        std::size_t inspected = 0;
        const auto poses = disagreeingPoses(source, 5, 4, rng, 24, inspected);
        require(poses.size() >= 4, "found only " + std::to_string(poses.size()) + " Q28/binary64 disagreements in " +
            std::to_string(inspected) + " pixels");
        std::cout << "Q28/binary64 disagreements: " << poses.size() << " poses in " << inspected << " pixels\n";
        fineCase(gpu, source, templ, poses, "disagreeing poses");
    }
    // 2. FP64 sampling of every pixel (fine NCC mode 1) at poses where re-associated
    //    or contracted coordinate arithmetic changes a pixel.
    {
        const auto source = noise(64, 48, rng, true);
        const auto templ = learnLevel(texture(5, 4, rng));
        std::size_t inspected = 0;
        const auto poses = arithmeticPoses(source, 5, 4, rng, 24, inspected);
        require(poses.size() >= 4, "found only " + std::to_string(poses.size()) + " arithmetic-sensitive poses in " +
            std::to_string(inspected) + " pixels");
        std::cout << "arithmetic-sensitive poses: " << poses.size() << " in " << inspected << " pixels\n";
        gpu.setFineNccMode(1);
        fineCase(gpu, source, templ, poses, "arithmetic-sensitive poses, FP64 sampling");
        gpu.setFineNccMode(0);
    }
    // 3. Random, border, out-of-image, special-angle and repairAll poses.
    for (int trial = 0; trial < 8; ++trial) {
        const int sw = 20 + int(rng() % 300), sh = 20 + int(rng() % 200);
        const int tw = 1 + int(rng() % 90), th = 1 + int(rng() % 70);
        const auto source = texture(sw, sh, rng);
        const auto templ = learnLevel(texture(tw, th, rng));
        std::vector<FineRequest> requests;
        for (int i = 0; i < 40; ++i) requests.push_back({{u(rng) * sw, u(rng) * sh}, (u(rng) * 2 - 1) * 180});
        for (double angle : {0.0, 90.0, -90.0, 180.0, -180.0, 1e-12, 45.0})
            requests.push_back({{std::floor(u(rng) * sw), std::floor(u(rng) * sh)}, angle});
        requests.push_back({{-tw - 20.0, -th - 20.0}, 30});
        requests.push_back({{sw + 5.0, sh * 0.5}, -60});
        requests.push_back({{3.5e9, 12.0}, 10});
        requests.push_back({{-4.0e9, -4.0e9}, 0});
        fineCase(gpu, source, templ, requests, "fine trial " + std::to_string(trial));
    }
    // 4. A flat template (every score 1) and a template wider than one 32-row stripe is tall.
    {
        const auto source = texture(400, 120, rng);
        Image flat(17, 9);
        for (int y = 0; y < 9; ++y) std::memset(flat.row(y), 77, 17);
        fineCase(gpu, source, learnLevel(flat), {{{100.5, 50.25}, 12}, {{3, 4}, 0}}, "flat template");
        fineCase(gpu, source, learnLevel(texture(300, 70, rng)), {{{150, 60}, 3}, {{140.25, 55.5}, -7}}, "wide template");
    }
    const auto metrics = gpu.fineMetrics();
    require(metrics.repair_pixels > 0 && metrics.repair_all_maps > 0, "FP64 fallback paths were not exercised");
    std::cout << "fine: " << metrics.maps << " maps, " << metrics.repair_pixels << " FP64 pixels, "
              << metrics.repair_all_maps << " FP64 maps\n";
}

// Peaks of one coarse request on the device against selectPeaks(ncc(rotateCanvas())).
void sameCoarse(fpm_coarse_detail::Backend& gpu, const Image& source, const TemplateLevel& templ,
    const std::vector<double>& angles, std::uint8_t border, int count, double overlapRatio, double threshold,
    const std::string& name)
{
    const int tw = templ.image.width(), th = templ.image.height();
    gpu.invalidateFineSource();
    const auto got = gpu.computeCoarseDevice(source.view(), templ, angles, border, count, overlapRatio, threshold);
    require(got.size() == angles.size(), name + ": coarse angle count");
    for (std::size_t i = 0; i < angles.size(); ++i) {
        const auto canvas = rotateCanvas(source.view(), tw, th, angles[i], border);
        const std::string what = name + " angle " + std::to_string(i);
        require(got[i].canvasWidth == canvas.image.width() && got[i].canvasHeight == canvas.image.height() &&
            std::memcmp(&got[i].translation, &canvas.translation, sizeof(Point)) == 0, what + ": canvas geometry");
        if (canvas.image.width() < tw || canvas.image.height() < th) {
            require(got[i].peaks.empty(), what + ": skipped canvas has peaks");
            continue;
        }
        const auto expected = selectPeaks(ncc(canvas.image.view(), templ, fpm::Backend::Auto), tw, th,
            count, overlapRatio, threshold, false);
        require(got[i].peaks.size() == expected.size(), what + ": peak count");
        for (std::size_t k = 0; k < expected.size(); ++k)
            require(got[i].peaks[k].x == expected[k].x && got[i].peaks[k].y == expected[k].y &&
                std::memcmp(&got[i].peaks[k].score, &expected[k].score, sizeof(double)) == 0, what + ": peak differs");
    }
}

void coarse()
{
    Rng rng(4242);
    std::uniform_real_distribution<double> u(0, 1);
    // Scores exactly at the threshold are kept: a flat template scores 1 everywhere,
    // a flat source scores 0 everywhere (zero variance). Their own generator keeps
    // the random trials below unchanged.
    Rng flatRng(99);
    for (int mode = 0; mode < 2; ++mode) {
        Gpu backend;
        backend.value->setCoarseOptimization(0, mode);
        Image flat(9, 7), flatSource(60, 40);
        for (int y = 0; y < 7; ++y) std::memset(flat.row(y), 50, 9);
        for (int y = 0; y < 40; ++y) std::memset(flatSource.row(y), 120, 60);
        const auto textured = texture(80, 50, flatRng);
        sameCoarse(*backend.value, textured, learnLevel(flat), {0, 30, -90}, 0, 6, 0.3, 1.0, "flat template, threshold 1");
        sameCoarse(*backend.value, flatSource, learnLevel(texture(9, 7, flatRng)), {0, 45}, 120, 6, 0.3, 0.0, "flat source, threshold 0");
    }
    for (int mode = 0; mode < 2; ++mode) {
        Gpu backend;
        auto& gpu = *backend.value;
        gpu.setCoarseOptimization(0, mode);
        for (int trial = 0; trial < 30; ++trial) {
            const int sw = 8 + int(rng() % 200), sh = 8 + int(rng() % 160);
            const int tw = 1 + int(rng() % std::min(40, sw)), th = 1 + int(rng() % std::min(40, sh));
            const auto source = texture(sw, sh, rng);
            Image templImage = trial == 7 ? Image(tw, th) : texture(tw, th, rng);
            if (trial == 7) for (int y = 0; y < th; ++y) std::memset(templImage.row(y), 50, std::size_t(tw));
            const auto templ = learnLevel(templImage);
            std::vector<double> angles;
            for (int i = 0, n = 1 + int(rng() % 40); i < n; ++i) angles.push_back(normalizeAngle((u(rng) * 2 - 1) * 180));
            for (double a : {0.0, 90.0, 180.0, -90.0}) if (rng() % 2) angles.push_back(a);
            const std::uint8_t border = std::uint8_t(rng() % 256);
            const int count = int(rng() % 30);
            const double overlapRatio = (rng() % 9) / 8.0, threshold = (rng() % 15) / 10.0 - 0.5;
            gpu.invalidateFineSource();
            const auto got = gpu.computeCoarseDevice(source.view(), templ, angles, border, count, overlapRatio, threshold);
            require(got.size() == angles.size(), "coarse angle count");
            for (std::size_t i = 0; i < angles.size(); ++i) {
                const auto canvas = rotateCanvas(source.view(), tw, th, angles[i], border);
                const std::string what = "coarse mode " + std::to_string(mode) + " trial " + std::to_string(trial) +
                    " angle " + std::to_string(i);
                require(got[i].canvasWidth == canvas.image.width() && got[i].canvasHeight == canvas.image.height() &&
                    std::memcmp(&got[i].translation, &canvas.translation, sizeof(Point)) == 0, what + ": canvas geometry");
                if (canvas.image.width() < tw || canvas.image.height() < th) {
                    require(got[i].peaks.empty(), what + ": skipped canvas has peaks");
                    continue;
                }
                const auto expected = selectPeaks(ncc(canvas.image.view(), templ, fpm::Backend::Auto), tw, th,
                    count, overlapRatio, threshold, false);
                require(got[i].peaks.size() == expected.size(), what + ": peak count");
                for (std::size_t k = 0; k < expected.size(); ++k)
                    require(got[i].peaks[k].x == expected[k].x && got[i].peaks[k].y == expected[k].y &&
                        std::memcmp(&got[i].peaks[k].score, &expected[k].score, sizeof(double)) == 0, what + ": peak differs");
            }
        }
    }
}

void dense()
{
    Gpu backend;
    Rng rng(77);
    for (int trial = 0; trial < 6; ++trial) {
        const int sw = 10 + int(rng() % 150), sh = 10 + int(rng() % 120);
        const auto source = texture(sw, sh, rng);
        const auto templ = learnLevel(texture(1 + int(rng() % 9), 1 + int(rng() % 9), rng));
        require(sameBits(backend.value->compute(source.view(), templ).values, ncc(source.view(), templ, fpm::Backend::Auto).values),
            "dense map differs");
    }
    // Bright, nearly flat windows: 10*FLT_EPSILON*square exceeds 0.5, so the 0.5 cap
    // of nccDenominator()'s variance guard decides whether a window scores 0.
    Image bright(120, 90);
    for (int y = 0; y < 90; ++y)
        for (int x = 0; x < 120; ++x) bright.row(y)[x] = std::uint8_t(rng() % 23 == 0 ? 201 : 200);
    const auto templ = learnLevel(texture(12, 10, rng));
    require(sameBits(backend.value->compute(bright.view(), templ).values, ncc(bright.view(), templ, fpm::Backend::Auto).values),
        "dense map of a bright, nearly flat image differs");
}

void pyramid()
{
    Gpu backend;
    const FpmCoarseExecutionConfigV1 execution{1, sizeof(FpmCoarseExecutionConfigV1), 1, 1, 0, 0};
    backend.value->setExecution(execution);
    Rng rng(9);
    for (auto size : std::vector<std::pair<int, int>>{{1, 1}, {2, 3}, {5, 1}, {37, 23}, {513, 257}, {1000, 3}}) {
        std::vector<Image> levels{noise(size.first, size.second, rng)};
        std::vector<Image> expected{levels[0]};
        for (int i = 0; i < 5; ++i) expected.push_back(pyrDown(expected.back().view()));
        backend.value->invalidateFineSource();
        require(backend.value->buildPyramid(levels, 5) && levels.size() == expected.size(), "pyramid level count");
        for (std::size_t i = 0; i < levels.size(); ++i) {
            require(levels[i].width() == expected[i].width() && levels[i].height() == expected[i].height(), "pyramid size");
            for (int y = 0; y < levels[i].height(); ++y)
                require(std::memcmp(levels[i].row(y), expected[i].row(y), std::size_t(levels[i].width())) == 0, "pyramid pixels differ");
        }
    }
}
}

int main()
{
    try {
        Gpu probe;
    } catch (const std::exception& e) {
        if (std::strstr(e.what(), "no CUDA device")) { std::cout << "SKIP: " << e.what() << "\n"; return 77; }
        std::cerr << "FAIL backend creation: " << e.what() << "\n";
        return 1;
    }
    const std::pair<const char*, void (*)()> tests[] = {{"fine", fine}, {"coarse", coarse}, {"dense", dense}, {"pyramid", pyramid}};
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << "\n";
        } catch (const std::exception& e) {
            std::cerr << "FAIL " << test.first << ": " << e.what() << "\n";
            return 1;
        }
    }
    return 0;
}
