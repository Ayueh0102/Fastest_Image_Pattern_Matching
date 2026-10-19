// SPDX-License-Identifier: BSD-2-Clause
#include "internal.hpp"
#include <algorithm>
#include <cmath>

namespace fpm::detail {
namespace {
// Modified Gram-Schmidt QR on a small, fixed, well-scaled design matrix.
// Columns use local pixel/angle-step coordinates, avoiding absolute-angle
// normal equations and explicit matrix inversion from the legacy code.
bool leastSquares(double a[27][10], const double* samples, int rows, int columns, double* coefficients)
{
	double r[10][10]{}, projected[10]{};
	for (int j = 0; j < columns; ++j) {
		for (int i = 0; i < j; ++i) {
			for (int k = 0; k < rows; ++k) r[i][j] += a[k][i] * a[k][j];
			for (int k = 0; k < rows; ++k) a[k][j] -= r[i][j] * a[k][i];
		}
		for (int k = 0; k < rows; ++k) r[j][j] += a[k][j] * a[k][j];
		r[j][j] = std::sqrt(r[j][j]);
		if (r[j][j] < 1e-12) return false;
		for (int k = 0; k < rows; ++k) {
			if (!std::isfinite(samples[k])) return false;
			a[k][j] /= r[j][j];
			projected[j] += a[k][j] * samples[k];
		}
	}
	for (int i = columns - 1; i >= 0; --i) {
		double value = projected[i];
		for (int j = i + 1; j < columns; ++j) value -= r[i][j] * coefficients[j];
		coefficients[i] = value / r[i][i];
	}
	return true;
}

// Solve (-H) * offset = gradient using Cholesky. Success also certifies that
// the fitted Hessian is negative definite (a local maximum, not a saddle).
bool maximumOffset(double h[3][3], const double* gradient, int size, double* offset)
{
	double l[3][3]{}, intermediate[3]{};
	for (int i = 0; i < size; ++i)
		for (int j = 0; j <= i; ++j) {
			double value = -h[i][j];
			for (int k = 0; k < j; ++k) value -= l[i][k] * l[j][k];
			if (i == j) {
				if (!std::isfinite(value) || value <= 1e-10) return false;
				l[i][j] = std::sqrt(value);
			} else l[i][j] = value / l[j][j];
		}
	for (int i = 0; i < size; ++i) {
		double value = gradient[i];
		for (int j = 0; j < i; ++j) value -= l[i][j] * intermediate[j];
		intermediate[i] = value / l[i][i];
	}
	for (int i = size - 1; i >= 0; --i) {
		double value = intermediate[i];
		for (int j = i + 1; j < size; ++j) value -= l[j][i] * offset[j];
		offset[i] = value / l[i][i];
		if (!std::isfinite(offset[i]) || std::abs(offset[i]) > 1) return false;
	}
	return true;
}
}

bool refine3d(const std::array<double, 27>& samples, std::array<double, 3>& offset)
{
	offset = {};
	double a[27][10]{}, z[10]{};
	int row = 0;
	for (int t = -1; t <= 1; ++t)
		for (int y = -1; y <= 1; ++y)
			for (int x = -1; x <= 1; ++x) {
				const double values[] = {double(x*x), double(y*y), double(t*t), double(x*y), double(x*t), double(y*t), double(x), double(y), double(t), 1};
				std::copy_n(values, 10, a[row++]);
			}
	if (!leastSquares(a, samples.data(), 27, 10, z)) return false;
	double h[3][3] = {{2*z[0], z[3], z[4]}, {z[3], 2*z[1], z[5]}, {z[4], z[5], 2*z[2]}};
	const double gradient[] = {z[6], z[7], z[8]};
	std::array<double, 3> fitted{};
	if (!maximumOffset(h, gradient, 3, fitted.data())) return false;
	offset = fitted;
	return true;
}

bool refine2d(const std::array<double, 9>& samples, Point& offset)
{
	offset = {};
	double a[27][10]{}, z[10]{};
	int row = 0;
	for (int y = -1; y <= 1; ++y)
		for (int x = -1; x <= 1; ++x) {
			const double values[] = {double(x*x), double(y*y), double(x*y), double(x), double(y), 1};
			std::copy_n(values, 6, a[row++]);
		}
	if (!leastSquares(a, samples.data(), 9, 6, z)) return false;
	double h[3][3] = {{2*z[0], z[2], 0}, {z[2], 2*z[1], 0}, {0, 0, 0}};
	const double gradient[] = {z[3], z[4]};
	double fitted[3]{};
	if (!maximumOffset(h, gradient, 2, fitted)) return false;
	offset = {fitted[0], fitted[1]};
	return true;
}
} // namespace fpm::detail
