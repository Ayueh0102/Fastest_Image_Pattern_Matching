// SPDX-License-Identifier: BSD-2-Clause
// Contract probe for the fixed-scale, no-downsampling research configuration.
// This exact-paste fixture checks the full-resolution, fixed-scale path while
// rotation search and continuous refinement remain enabled. It is not a claim
// of accuracy on independently captured or nonzero-rotation photographs.
#include "research_pipeline.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void sameMatches(const std::vector<fpm::Match>& first,
    const std::vector<fpm::Match>& second) {
    require(first.size() == second.size(), "Repeated result count changed");
    for (size_t i = 0; i < first.size(); ++i) {
        const auto& a = first[i];
        const auto& b = second[i];
        require(a.center.x == b.center.x && a.center.y == b.center.y &&
            a.angle == b.angle && a.score == b.score,
            "Repeated ordered pose or score changed");
        for (size_t corner = 0; corner < a.corners.size(); ++corner)
            require(a.corners[corner].x == b.corners[corner].x &&
                a.corners[corner].y == b.corners[corner].y,
                "Repeated ordered corners changed");
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--metal"))
            throw std::invalid_argument("Usage: fixed_scale_contract_test [--metal]");
        const bool metal = argc == 2;
        constexpr int width = 128, height = 96;
        constexpr int templateWidth = 32, templateHeight = 24;
        constexpr int pasteX = 43, pasteY = 31;
        fpm::Image templ(templateWidth, templateHeight), source(width, height);
        // Fixed integer PRNG gives an asymmetric, nonperiodic template without
        // sharing interpolation or coordinate code with the matching library.
        uint32_t state = 0x6f24a931u;
        auto next = [&] {
            state = state * 1664525u + 1013904223u;
            return uint8_t(state >> 24);
        };
        for (int y = 0; y < templateHeight; ++y)
            for (int x = 0; x < templateWidth; ++x) templ.row(y)[x] = next();
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) source.row(y)[x] = uint8_t(16 + (next() & 15));
        for (int y = 0; y < templateHeight; ++y)
            for (int x = 0; x < templateWidth; ++x)
                source.row(pasteY + y)[pasteX + x] = templ.row(y)[x];

        // A threshold equal to the nonsquare area does NOT mean no pyramid:
        // this public constructor preserves the legacy squared-sqrt rule.
        const fpm::Pattern areaThreshold(templ.view(), 768);
        const fpm::Pattern fullResolution(templ.view(), 784);
        require(areaThreshold.levels().size() > 1,
            "The historical nonsquare-area threshold contract changed");
        require(fullResolution.levels().size() == 1,
            "Safe full-resolution threshold unexpectedly reduced template");

        fpm::research::Settings settings;
        settings.backend = metal ? 1 : 0;
        settings.minReducedArea = 784;
        settings.scales = {1.};
        fpm::research::Options options;
        options.match.maxTargets = 4;
        options.match.minScore = .97;
        options.match.maxOverlap = .2;
        options.match.toleranceAngle = 20.;
        options.match.stopLayer = 0;
        options.match.subpixel = true;
        options.refine = true;
        fpm::research::Session session(source.view(), templ.view(), settings);
        const auto first = session.run(options);
        const auto second = session.run(options);
        const uint64_t levelZeroBytes = uint64_t(width) * height;
        for (const auto* result : {&first, &second}) {
            require(result->frame.host_pyramid_bytes == levelZeroBytes,
                "Fixed-scale full-resolution frame contains extra pyramid pixels");
            require(result->frame.models_matched == 1, "Unexpected scale-bank work");
            require(result->matches.size() == 1 && result->scales.size() == 1,
                "Exact planted target was missed or a false positive was returned");
            require(result->rescue.selected_backend == (metal ? 1 : 0),
                "Continuous refinement used the wrong backend");
            require(!result->proposalScales.empty(), "No proposal reached refinement");
            for (double scale : result->proposalScales)
                require(std::abs(scale - 1.) <= 1e-12, "Proposal scale is not locked to one");
            for (double scale : result->scales)
                require(std::abs(scale - 1.) <= 1e-12, "Continuous refinement changed fixed scale");
            const auto& match = result->matches.front();
            require(std::hypot(match.center.x - (pasteX + templateWidth / 2.),
                match.center.y - (pasteY + templateHeight / 2.)) <= .35,
                "Fixed-scale result has the wrong location");
            require(std::abs(match.angle) <= .5 && std::abs(match.angle) <= options.match.toleranceAngle,
                "Fixed-scale result has the wrong angle or escaped the search range");
            require(match.score >= options.match.minScore, "Returned result is below the acceptance threshold");
        }
        require(first.frame.generation == 1 && second.frame.generation == 2,
            "Repeated Session calls did not prepare consecutive frames");
        require(first.scales == second.scales && first.proposalScales == second.proposalScales,
            "Repeated ordered scales changed");
        sameMatches(first.matches, second.matches);
        sameMatches(first.proposals, second.proposals);
        std::cout << (metal ? "Metal" : "CPU")
            << " fixed-scale contract: one level, locked scale, rotation search, "
               "continuous refinement, planted target and exact repeats passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
