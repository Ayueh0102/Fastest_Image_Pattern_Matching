// SPDX-License-Identifier: BSD-2-Clause
#include "match_verify.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fpm::research {
namespace {
constexpr double pi = 3.14159265358979323846;
bool inside(double x, double y, int w, int h) { return x >= 0 && y >= 0 && x <= w - 1 && y <= h - 1; }
double bilinear(ImageView s, double x, double y)
{
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    const int x1 = std::min(x0 + 1, s.width - 1), y1 = std::min(y0 + 1, s.height - 1);
    const double dx = x - x0, dy = y - y0;
    const auto at = [&](int px, int py) { return double(s.data[size_t(py) * s.stride + px]); };
    return (at(x0, y0) * (1 - dx) + at(x1, y0) * dx) * (1 - dy) + (at(x0, y1) * (1 - dx) + at(x1, y1) * dx) * dy;
}
} // namespace

MatchVerifyDecision verifyMatch(ImageView source, ImageView templ, Point center, double angleDegrees,
    double scale, double score, double minScore, bool invertSource, const MatchVerifyRules& rules)
{
    if (!std::isfinite(rules.borderMinInside) || rules.borderMinInside < 0 || rules.borderMinInside > 1 ||
        !std::isfinite(rules.smallTemplateArea) || rules.smallTemplateArea < 0)
        throw std::invalid_argument("Match verification needs border fraction 0..1 and small-template area >= 0");
    if (!source.data || !templ.data || source.width < 1 || source.height < 1 || templ.width < 1 || templ.height < 1 ||
        source.stride < source.width || templ.stride < templ.width)
        throw std::invalid_argument("Match verification needs nonempty source and template views");
    MatchVerifyDecision d;
    d.requiredScore = minScore;
    const int w = templ.width, h = templ.height;
    if (rules.smallTemplateArea > 0) {
        const double area = double(w) * h * scale * scale;
        if (area < rules.smallTemplateArea) d.requiredScore = 1 - (1 - minScore) * std::sqrt(area / rules.smallTemplateArea);
        if (score < d.requiredScore) {
            d.keep = false;
            d.rejectedSize = true;
        }
    }
    if (rules.borderMinInside > 0) {
        const double a = angleDegrees * pi / 180, c = scale * std::cos(a), s = scale * std::sin(a);
        const auto map = [&](double tx, double ty) {
            const double u = tx - w / 2., v = ty - h / 2.;
            return Point{center.x + c * u + s * v, center.y - s * u + c * v};
        };
        // The footprint is the affine image of the template pixel grid, so it stays
        // inside exactly when its four extreme pixels do.
        const std::pair<int, int> extremes[] = {{0, 0}, {w - 1, 0}, {0, h - 1}, {w - 1, h - 1}};
        bool all = true;
        for (auto [tx, ty] : extremes) {
            const Point p = map(tx, ty);
            all = all && inside(p.x, p.y, source.width, source.height);
        }
        if (!all) {
            d.border = true;
            double n = 0, st = 0, sv = 0, stt = 0, svv = 0, stv = 0;
            for (int ty = 0; ty < h; ++ty) for (int tx = 0; tx < w; ++tx) {
                const Point p = map(tx, ty);
                if (!inside(p.x, p.y, source.width, source.height)) continue;
                const double t = templ.data[size_t(ty) * templ.stride + tx];
                double v = bilinear(source, p.x, p.y);
                if (invertSource) v = 255 - v;
                n += 1;
                st += t;
                sv += v;
                stt += t * t;
                svv += v * v;
                stv += t * v;
            }
            d.insideFraction = n / (double(w) * h);
            const double vt = stt - st * st / std::max(n, 1.), vv = svv - sv * sv / std::max(n, 1.);
            d.insideScore = n >= 2 && vt > 0 && vv > 0 ? (stv - st * sv / n) / std::sqrt(vt * vv) : 0;
            if (d.insideFraction < rules.borderMinInside || d.insideScore < minScore) {
                d.keep = false;
                d.rejectedBorder = true;
            }
        }
    }
    return d;
}
} // namespace fpm::research
