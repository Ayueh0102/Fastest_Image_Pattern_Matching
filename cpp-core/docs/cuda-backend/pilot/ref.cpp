// SPDX-License-Identifier: BSD-2-Clause
// CPU references: the same fpm_core functions the matcher calls, compiled by cl.
#include "pilot.hpp"
#include "internal.hpp"
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

using namespace fpm;
using namespace fpm::detail;

namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

Image readPgm(const char* path)
{
    std::ifstream in(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(in)), {});
    std::size_t pos = 0;
    std::string fields[4];
    for (int i = 0; i < 4; ++i) {
        while (pos < data.size() && (std::isspace((unsigned char)data[pos]) || data[pos] == '#'))
            pos = data[pos] == '#' ? data.find('\n', pos) + 1 : pos + 1;
        const std::size_t end = data.find_first_of(" \t\r\n", pos);
        fields[i] = data.substr(pos, end - pos);
        pos = end;
    }
    if (fields[0] != "P5" || fields[3] != "255") throw std::runtime_error(std::string("not an 8-bit P5 PGM: ") + path);
    const int w = std::stoi(fields[1]), h = std::stoi(fields[2]);
    ++pos;
    if (data.size() < pos + std::size_t(w) * h) throw std::runtime_error("truncated PGM");
    Image image(w, h);
    for (int y = 0; y < h; ++y)
        std::copy_n(data.data() + pos + std::size_t(y) * w, w, image.row(y));
    return image;
}

Plane plane(const Image& image)
{
    Plane p{image.width(), image.height(), {}};
    p.pixels.resize(std::size_t(p.width) * p.height);
    for (int y = 0; y < p.height; ++y) std::copy_n(image.row(y), p.width, p.pixels.data() + std::size_t(y) * p.width);
    return p;
}
ImageView view(const Plane& p) { return {p.pixels.data(), p.width, p.height, std::size_t(p.width)}; }

// Same as matcher.cpp coarseAngles() for an empty angleRanges list.
void appendAngle(std::vector<double>& angles, double angle)
{
    angle = normalizeAngle(angle);
    for (double existing : angles)
        if (std::abs(normalizeAngle(existing - angle)) < 1e-9) return;
    angles.push_back(angle);
}
}

Workload loadWorkload(const char* source, const char* templ, int minArea, double tolerance)
{
    const Image src = readPgm(source), tmpl = readPgm(templ);
    const Pattern pattern(tmpl.view(), minArea);
    Workload w;
    w.border = pattern.borderColor();
    for (const auto& level : pattern.levels()) w.templ.push_back({plane(level.image), level.mean, level.norm});
    w.source.push_back(plane(src));
    for (std::size_t i = 1; i < pattern.levels().size(); ++i)
        w.source.push_back(plane(pyrDown(view(w.source.back()))));
    const auto& top = pattern.levels().back().image;
    const double step = std::atan(2.0 / std::max(top.width(), top.height())) / D2R;
    appendAngle(w.angles, 0);
    for (double a = step; a < tolerance; a += step) appendAngle(w.angles, a);
    appendAngle(w.angles, tolerance);
    for (double a = -step; a > -tolerance; a -= step) appendAngle(w.angles, a);
    appendAngle(w.angles, -tolerance);
    return w;
}

WarpJob canvasJob(const Plane& top, int tw, int th, double angle, std::uint8_t border, Plane* reference, double* cpuMs)
{
    // Geometry copied from rotateCanvas(); the pilot compares the GPU canvas with
    // rotateCanvas() itself, so any drift in this copy shows up as a mismatch.
    const Point center{(top.width - 1) * 0.5, (top.height - 1) * 0.5};
    const double ca = std::abs(std::cos(angle * D2R)), sa = std::abs(std::sin(angle * D2R));
    double w = 2 * std::ceil(((top.width - 1) * ca + (top.height - 1) * sa) * 0.5 - th * sa * ca);
    double h = 2 * std::ceil(((top.width - 1) * sa + (top.height - 1) * ca) * 0.5 - tw * sa * ca);
    if (sa < 1e-12) { w = top.width; h = top.height; }
    else if (ca < 1e-12) { w = top.height; h = top.width; }
    if (w < tw || h < th) {
        w = std::ceil(top.width * ca + top.height * sa);
        h = std::ceil(top.width * sa + top.height * ca);
    }
    const Point translation{(w - 1) * 0.5 - center.x, (h - 1) * 0.5 - center.y};
    const Point anchor = rotate({-translation.x, -translation.y}, center, -angle);
    WarpJob job{anchor.x, anchor.y, std::cos(angle * D2R), std::sin(angle * D2R), int(w), int(h), 0};
    if (reference) {
        const auto t = Clock::now();
        auto canvas = rotateCanvas(view(top), tw, th, angle, border);
        if (cpuMs) *cpuMs += since(t);
        *reference = plane(canvas.image);
    }
    return job;
}

WarpJob roiJob(const Plane& level, int tw, int th, double ax, double ay, double angle, Plane* reference, double* cpuMs)
{
    const Point origin = rotate({-PADDING, -PADDING}, {0, 0}, -angle);
    WarpJob job{ax + origin.x, ay + origin.y, std::cos(angle * D2R), std::sin(angle * D2R),
        tw + 2 * PADDING, th + 2 * PADDING, 0};
    if (reference) {
        const auto t = Clock::now();
        auto roi = rotatedRoi(view(level), tw, th, {ax, ay}, angle);
        if (cpuMs) *cpuMs += since(t);
        *reference = plane(roi);
    }
    return job;
}

std::vector<double> referenceNcc(const Plane& image, const TemplatePlane& templ, double* cpuMs)
{
    TemplateLevel level{Image(view(templ.image)), templ.mean, templ.norm};
    const auto t = Clock::now();
    auto map = ncc(view(image), level, Backend::Auto);
    if (cpuMs) *cpuMs += since(t);
    return map.values;
}

Plane referencePyrDown(const Plane& image, double* cpuMs)
{
    const auto t = Clock::now();
    auto out = pyrDown(view(image));
    if (cpuMs) *cpuMs += since(t);
    return plane(out);
}

const char* coreBackend() { return compiledBackend(); }
