// SPDX-License-Identifier: BSD-2-Clause
// Deterministic NCC microbenchmark. Run alone: timings are not full-search
// latency. The dense score maps are deliberately retained for exact parity.
#include "internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace fpm;
using namespace fpm::detail;
namespace {
using Clock = std::chrono::steady_clock;
Image makeImage(int width, int height, unsigned seed)
{
	Image out(width, height);
	std::mt19937 random(seed);
	for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
		out.row(y)[x] = std::uint8_t(random() % 256);
	return out;
}
double median(std::vector<double> values)
{
	std::sort(values.begin(), values.end());
	return values[values.size() / 2];
}
}
int main(int argc, char** argv)
{
	try {
		const int width = argc > 1 ? std::stoi(argv[1]) : 640;
		const int height = argc > 2 ? std::stoi(argv[2]) : 480;
		const int repeats = argc > 3 ? std::stoi(argv[3]) : 3;
		if (width < 160 || height < 160 || repeats < 1) throw std::invalid_argument("usage: benchmark_ncc_pruning [width>=160 height>=160 repeats>=1]");
		std::cout << std::setprecision(12) << "{\"backend\":\"" << compiledBackend() << "\",\"source_width\":" << width
			<< ",\"source_height\":" << height << ",\"repeats\":" << repeats << ",\"runs\":[\n";
		bool first = true;
		for (int size : {16, 32, 64, 128}) for (bool structured : {false, true}) {
			auto source = makeImage(width, height, 240 + size);
			auto patch = makeImage(size, size, 420 + size);
			if (structured)
				for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x)
					patch.row(y)[x] = std::uint8_t(20 + (x >= size / 2 ? 100 : 0) + (y >= size / 2 ? 70 : 0) + patch.row(y)[x] % 20);
			for (int y = 0; y < size; ++y) std::copy_n(patch.row(y), size, source.row(11 + y) + 13);
			const auto templ = learnLevel(patch);
			for (double threshold : {.5, .8, .95}) {
				NccStats fullStats, prunedStats;
				// Untimed warm-up and correctness. No ground truth feeds the bound.
				const auto full = ncc(source.view(), templ, Backend::Auto, -INFINITY, &fullStats);
				const auto bounded = ncc(source.view(), templ, Backend::Auto, threshold, &prunedStats);
				std::size_t accepted = 0;
				for (std::size_t i = 0; i < full.values.size(); ++i) {
					if (full.values[i] >= threshold) ++accepted;
					if (std::isfinite(bounded.values[i]) ? bounded.values[i] != full.values[i] : full.values[i] >= threshold)
						throw std::runtime_error("threshold-score parity failed");
				}
				std::vector<double> baseline, pruned;
				for (int iteration = 0; iteration < repeats; ++iteration) {
					// Alternate order to reduce one-direction thermal/cache bias.
					for (int pass = 0; pass < 2; ++pass) {
						const bool prune = (iteration + pass) % 2 != 0;
						const auto start = Clock::now();
						const auto scores = ncc(source.view(), templ, Backend::Auto, prune ? threshold : -INFINITY);
						const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
						(prune ? pruned : baseline).push_back(elapsed);
						if (scores.values.empty()) throw std::runtime_error("empty score map");
					}
				}
				if (!first) std::cout << ",\n";
				first = false;
				const double fullMs = median(baseline), prunedMs = median(pruned);
				const std::uint64_t mapBytes = std::uint64_t(width - size + 1) * (height - size + 1) * sizeof(double);
				const std::uint64_t integralBytes = std::uint64_t(width + 1) * (height + 1) * sizeof(double) * 2;
				std::cout << "{\"template_size\":" << size << ",\"texture\":\"" << (structured ? "four_blocks_noise" : "noise")
					<< "\",\"threshold\":" << threshold << ",\"exhaustive_ms\":" << fullMs << ",\"pruned_ms\":" << prunedMs
					<< ",\"speedup\":" << fullMs / prunedMs << ",\"windows\":" << prunedStats.windows << ",\"accepted\":" << accepted
					<< ",\"projection_checks\":" << prunedStats.projectionChecks << ",\"partial_checks\":" << prunedStats.partialChecks
					<< ",\"rejected\":" << prunedStats.rejected << ",\"full_dot_products\":" << prunedStats.fullDotProducts
					<< ",\"dot_pixels\":" << prunedStats.dotPixels << ",\"exhaustive_dot_pixels\":" << fullStats.dotPixels
					<< ",\"saved_dot_fraction\":" << 1 - double(prunedStats.dotPixels) / fullStats.dotPixels
					<< ",\"estimated_map_integral_bytes\":" << mapBytes + integralBytes << ",\"threshold_parity\":true}";
			}
		}
		std::cout << "\n]}\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << e.what() << '\n'; return 1;
	}
}
