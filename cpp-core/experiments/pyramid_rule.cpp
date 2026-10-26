// SPDX-License-Identifier: BSD-2-Clause
#include "pyramid_rule.hpp"
#include "internal.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace fpm::research {
namespace {
using fpm::detail::D2R;
// Search-grid angles near 20/65/110/155 degrees; the residual off the grid is
// a quarter coarse step and the center sits a quarter top-level pixel off the
// integer grid, with the signs alternating over the four diagonals.
struct Pose { double gridDegrees, residualSteps, shiftX, shiftY; };
constexpr Pose POSES[PYRAMID_RULE_POSES]={
    {20,.25,.25,.25},{20,-.25,-.25,.25},{65,.25,-.25,-.25},{65,-.25,.25,-.25},
    {110,.25,.25,-.25},{110,-.25,-.25,-.25},{155,.25,-.25,.25},{155,-.25,.25,.25}};
constexpr int MARGIN = 4; // Top-level pixels of background around the object's circumcircle.

// Template pixel q, bilinear, with the flat background for every tap outside it.
double tap(ImageView t, int x, int y, double background)
{
    return x < 0 || y < 0 || x >= t.width || y >= t.height ? background : t.data[std::size_t(y) * t.stride + x];
}
// Object at `center`, rotated by `angle` (output convention: scene = center +
// R(angle)(q - template center), R = [[c, s], [-s, c]]) on a flat canvas.
Image render(ImageView t, int side, Point center, double angle, std::uint8_t background)
{
    Image canvas(side, side);
    const double c = std::cos(angle * D2R), s = std::sin(angle * D2R);
    const double tx = (t.width - 1) * 0.5, ty = (t.height - 1) * 0.5;
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            const double dx = x - center.x, dy = y - center.y;
            const double u = c * dx - s * dy + tx, v = s * dx + c * dy + ty;
            if (u <= -1 || v <= -1 || u >= t.width || v >= t.height) { canvas.row(y)[x] = background; continue; }
            const int iu = int(std::floor(u)), iv = int(std::floor(v));
            const double fu = u - iu, fv = v - iv;
            const double top = tap(t, iu, iv, background) * (1 - fu) + tap(t, iu + 1, iv, background) * fu;
            const double bottom = tap(t, iu, iv + 1, background) * (1 - fu) + tap(t, iu + 1, iv + 1, background) * fu;
            canvas.row(y)[x] = std::uint8_t(std::clamp(std::floor(top * (1 - fv) + bottom * fv + 0.5), 0.0, 255.0));
        }
    return canvas;
}
// NCC of the learned level against the window at (left, top); 0 when either is flat.
double windowNcc(const TemplateLevel& level, ImageView image, int left, int top)
{
    const int w = level.image.width(), h = level.image.height();
    if (left < 0 || top < 0 || left + w > image.width || top + h > image.height) return -2;
    double sum = 0, squares = 0, cross = 0;
    for (int y = 0; y < h; ++y) {
        const auto* row = image.data + std::size_t(top + y) * image.stride + left;
        const auto* templ = level.image.row(y);
        for (int x = 0; x < w; ++x) { sum += row[x]; squares += double(row[x]) * row[x]; cross += double(row[x]) * templ[x]; }
    }
    const double n = double(w) * h, variance = squares - sum * sum / n;
    if (level.norm <= 0 || variance <= 0) return 0;
    return std::clamp((cross - sum * level.mean) / (level.norm * std::sqrt(variance)), -1.0, 1.0);
}
} // namespace

int minAreaForTop(int width, int height, int top)
{
    if (width < 1 || height < 1 || top < 0) throw std::invalid_argument("minAreaForTop needs a positive size and level");
    std::uint64_t area = std::uint64_t(width) * height;
    for (int level = 0; level < top; ++level) area /= 4;
    // Pattern stops at the first area <= floor(sqrt(minArea))^2; ceil(sqrt(a))^2
    // is >= a and below 4a, so every earlier (>= 4a) area continues.
    std::uint64_t side = std::uint64_t(std::sqrt(double(std::max<std::uint64_t>(area, 1))));
    while (side * side < area) ++side;
    while (side > 1 && (side - 1) * (side - 1) >= area) --side;
    if (side * side > std::uint64_t(INT_MAX)) throw std::invalid_argument("template area too large for minReducedArea");
    return int(std::max<std::uint64_t>(side * side, 1));
}

double predictedTopScore(const Pattern& pattern, int level, std::uint8_t background)
{
    const auto& levels = pattern.levels();
    if (level < 0 || level >= int(levels.size())) throw std::invalid_argument("predicted level outside the pattern");
    const auto& top = levels[level];
    if (top.norm <= 0) return 0;
    const ImageView templ = levels.front().image.view();
    const int topW = top.image.width(), topH = top.image.height();
    const double factor = std::ldexp(1.0, level);
    const double step = std::atan(2.0 / std::max(topW, topH)) / D2R;
    const int sideTop = int(std::ceil(std::hypot(double(topW), double(topH)))) + 2 * MARGIN;
    const int side = int(sideTop * factor);
    double total = 0;
    for (const auto& pose : POSES) {
        // Search angle a shows an object of output angle -a upright.
        const double grid = std::round(pose.gridDegrees / step) * step;
        const double angle = -(grid + pose.residualSteps * step);
        const Point center{(side - 1) * 0.5 + pose.shiftX * factor, (side - 1) * 0.5 + pose.shiftY * factor};
        Image reduced = render(templ, side, center, angle, background);
        for (int k = 0; k < level; ++k) reduced = detail::pyrDown(reduced.view());
        const auto canvas = detail::rotateCanvas(reduced.view(), topW, topH, grid, background);
        // Level-L pixel i is level-0 pixel 2^L i; the matcher maps a canvas pixel
        // q back to rotate(q - translation, center, -grid).
        const Point source{(reduced.width() - 1) * 0.5, (reduced.height() - 1) * 0.5};
        const Point object = detail::rotate({center.x / factor, center.y / factor}, source, grid);
        const double left = object.x + canvas.translation.x - (templ.width - 1) * 0.5 / factor;
        const double upper = object.y + canvas.translation.y - (templ.height - 1) * 0.5 / factor;
        double best = -1;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                best = std::max(best, windowNcc(top, canvas.image.view(), int(std::lround(left)) + dx, int(std::lround(upper)) + dy));
        total += best;
    }
    return total / PYRAMID_RULE_POSES;
}

PyramidRuleDecision decidePyramid(ImageView templ, int minReducedArea, double minScore)
{
    if (!std::isfinite(minScore) || minScore < -1 || minScore > 1)
        throw std::invalid_argument("pyramid rule threshold must be in [-1, 1]");
    const auto begin = std::chrono::steady_clock::now();
    const Pattern pattern(templ, minReducedArea);
    PyramidRuleDecision decision;
    decision.originalTop = int(pattern.levels().size()) - 1;
    // Flat backgrounds two template standard deviations below and above its mean.
    const auto& full = pattern.levels().front();
    const double deviation = full.norm / std::sqrt(double(templ.width) * templ.height);
    auto background = [&](double sign) { return std::uint8_t(std::clamp(std::floor(full.mean + sign * 2 * deviation + .5), 0., 255.)); };
    decision.lowBackground = background(-1);
    decision.highBackground = background(1);
    for (int level = 0; level <= decision.originalTop; ++level)
        decision.predictedScores.push_back(std::min(predictedTopScore(pattern, level, decision.lowBackground),
            predictedTopScore(pattern, level, decision.highBackground)));
    int selected = -1;
    for (int level = decision.originalTop; level >= 0 && selected < 0; --level)
        if (decision.predictedScores[level] >= minScore) selected = level;
    if (selected < 0) {
        const auto& scores = decision.predictedScores;
        const double best = *std::max_element(scores.begin(), scores.end());
        for (int level = decision.originalTop; level >= 0 && selected < 0; --level)
            if (scores[level] == best) selected = level;
    }
    decision.selectedTop = selected;
    decision.effectiveMinArea = selected == decision.originalTop ? minReducedArea : minAreaForTop(templ.width, templ.height, selected);
    decision.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    return decision;
}
} // namespace fpm::research
