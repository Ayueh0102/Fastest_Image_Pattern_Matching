// SPDX-License-Identifier: BSD-2-Clause
#include "fine_request_profile.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
void partition(const fpm_coarse_detail::FineRequestProfileStats& value)
{
    require(value.requests == value.within_call_duplicates + value.cross_call_duplicates +
        value.tracked_unique_keys + value.untracked_call_keys, "request counts do not partition");
    require(value.tracked_unique_keys <= value.history_limit_keys, "history exceeded its limit");
}
fpm::detail::FineRequest request(double x = 12.5)
{
    return {{x, 7.25}, 30.};
}
}

int main()
{
    try {
        using fpm_coarse_detail::FineRequestProfiler;
        std::uint8_t pixels[256] = {}, otherPixels[256] = {};
        const fpm::ImageView source{pixels, 8, 8, 16};
        fpm::TemplateLevel templ; templ.image = fpm::Image(3, 4);
        FineRequestProfiler profiler;
        auto a = request();
        auto nearX = a; nearX.anchor.x = std::nextafter(a.anchor.x, INFINITY);
        auto nearY = a; nearY.anchor.y = std::nextafter(a.anchor.y, INFINITY);
        auto nearAngle = a; nearAngle.angle = std::nextafter(a.angle, INFINITY);
        profiler.observe(source, templ, {a, a, nearX, nearY, nearAngle});
        auto value = profiler.stats(); partition(value);
        require(value.calls == 1 && value.requests == 5 && value.within_call_duplicates == 1 &&
            value.cross_call_duplicates == 0 && value.tracked_unique_keys == 4,
            "nearby floating-point values were merged");
        require(value.history_limit_keys == 65536 && value.history_key_payload_bytes > 0 &&
            value.peak_call_key_capacity_bytes >= value.history_key_payload_bytes,
            "history/vector payload accounting missing");
        const auto payloadPerKey = value.history_key_payload_bytes / value.tracked_unique_keys;
        profiler.observe(source, templ, {a, a, nearX, nearX});
        value = profiler.stats(); partition(value);
        require(value.calls == 2 && value.requests == 9 && value.within_call_duplicates == 3 &&
            value.cross_call_duplicates == 2 && value.tracked_unique_keys == 4,
            "cross-call duplicates double-counted within-call occurrences");

        // Positive and negative zero are different input bit patterns in all
        // three coordinates, even if a warp happens to produce equal pixels.
        profiler.reset();
        auto zero = fpm::detail::FineRequest{{0., 0.}, 0.};
        auto negX = zero; negX.anchor.x = -0.;
        auto negY = zero; negY.anchor.y = -0.;
        auto negAngle = zero; negAngle.angle = -0.;
        profiler.observe(source, templ, {zero, negX, negY, negAngle, negX});
        value = profiler.stats(); partition(value);
        require(value.tracked_unique_keys == 4 && value.within_call_duplicates == 1,
            "signed zero bit patterns were merged");

        // Identity includes each source descriptor field and both template
        // dimensions, but intentionally excludes template pixels/identity.
        profiler.reset();
        profiler.observe(source, templ, {a});
        auto changedSource = source; changedSource.data = otherPixels;
        profiler.observe(changedSource, templ, {a});
        changedSource = source; changedSource.width -= 1;
        profiler.observe(changedSource, templ, {a});
        changedSource = source; changedSource.height -= 1;
        profiler.observe(changedSource, templ, {a});
        changedSource = source; changedSource.stride += 1;
        profiler.observe(changedSource, templ, {a});
        fpm::TemplateLevel wider; wider.image = fpm::Image(4, 4);
        profiler.observe(source, wider, {a});
        fpm::TemplateLevel taller; taller.image = fpm::Image(3, 5);
        profiler.observe(source, taller, {a});
        fpm::TemplateLevel sameSize; sameSize.image = fpm::Image(3, 4);
        sameSize.image.row(0)[0] = 255;
        profiler.observe(source, sameSize, {a});
        value = profiler.stats(); partition(value);
        require(value.requests == 8 && value.tracked_unique_keys == 7 &&
            value.cross_call_duplicates == 1, "ROI descriptor identity is incomplete");
        require(value.history_key_payload_bytes == 7 * payloadPerKey,
            "history payload did not track key count");
        profiler.reset();
        value = profiler.stats(); partition(value);
        require(value.calls == 0 && value.requests == 0 && value.tracked_unique_keys == 0 &&
            value.within_call_duplicates == 0 && value.cross_call_duplicates == 0 &&
            value.untracked_call_keys == 0 && value.history_key_payload_bytes == 0 &&
            value.peak_call_key_capacity_bytes == 0, "reset did not clear profiler state");
        profiler.observe(source, templ, {a});
        require(profiler.stats().cross_call_duplicates == 0, "reset retained old keys");

        FineRequestProfiler bounded(2);
        const auto b = request(20.), c = request(30.), d = request(40.);
        bounded.observe(source, templ, {a, b});
        bounded.observe(source, templ, {a, a, c, c, c});
        value = bounded.stats(); partition(value);
        require(value.requests == 7 && value.tracked_unique_keys == 2 &&
            value.within_call_duplicates == 3 && value.cross_call_duplicates == 1 &&
            value.untracked_call_keys == 1, "saturated history lost exact local duplicates");
        bounded.observe(source, templ, {c, c, d, d, b, b});
        value = bounded.stats(); partition(value);
        require(value.requests == 13 && value.tracked_unique_keys == 2 &&
            value.within_call_duplicates == 6 && value.cross_call_duplicates == 2 &&
            value.untracked_call_keys == 3,
            "untracked repeat was incorrectly reported as globally unique or tracked");
        const auto peakBytes = value.peak_call_key_capacity_bytes;
        bounded.observe(source, templ, {});
        value = bounded.stats(); partition(value);
        require(value.calls == 4 && value.requests == 13 &&
            value.peak_call_key_capacity_bytes == peakBytes, "empty call changed request counts");
        bounded.reset();
        bounded.observe(source, templ, {c, c});
        value = bounded.stats(); partition(value);
        require(value.history_limit_keys == 2 && value.tracked_unique_keys == 1 &&
            value.within_call_duplicates == 1 && value.untracked_call_keys == 0,
            "reset did not preserve capacity or admit previously untracked key");

        FineRequestProfiler noHistory(0);
        noHistory.observe(source, templ, {a, a, b});
        noHistory.observe(source, templ, {a, a});
        value = noHistory.stats(); partition(value);
        require(value.requests == 5 && value.within_call_duplicates == 2 &&
            value.tracked_unique_keys == 0 && value.cross_call_duplicates == 0 &&
            value.untracked_call_keys == 3, "zero-capacity local deduplication failed");
        bool rejected = false;
        try { FineRequestProfiler invalid(65537); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "history larger than production bound was accepted");
        std::cout << "Fine request profiler tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
