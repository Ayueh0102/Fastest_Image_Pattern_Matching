// SPDX-License-Identifier: BSD-2-Clause
// Plain data shared by the CUDA pilot (nvcc) and its fpm_core references (cl).
#pragma once
#include <cstdint>
#include <vector>

struct Plane { int width = 0, height = 0; std::vector<std::uint8_t> pixels; };
struct TemplatePlane { Plane image; double mean = 0, norm = 0; };
// One inverse-affine warp: output (x,y) samples anchor + R(-angle)(x,y).
struct WarpJob { double ax, ay, c, s; int width, height; std::uint64_t offset; };

struct Workload {
    std::vector<Plane> source;          // source pyramid, level 0..top
    std::vector<TemplatePlane> templ;   // template pyramid, level 0..top
    std::uint8_t border = 0;
    std::vector<double> angles;         // coarse angles at the top level
};

Workload loadWorkload(const char* source, const char* templ, int minArea, double tolerance);
// rotateCanvas geometry recomputed on the host; reference gets fpm_core's canvas.
WarpJob canvasJob(const Plane& top, int tw, int th, double angle, std::uint8_t border, Plane* reference, double* cpuMs);
// rotatedRoi geometry; reference gets fpm_core's ROI.
WarpJob roiJob(const Plane& level, int tw, int th, double ax, double ay, double angle, Plane* reference, double* cpuMs);
// fpm_core's ncc() (Backend::Auto, exhaustive) of one image.
std::vector<double> referenceNcc(const Plane& image, const TemplatePlane& templ, double* cpuMs);
// fpm_core's pyrDown.
Plane referencePyrDown(const Plane& image, double* cpuMs);
const char* coreBackend();
