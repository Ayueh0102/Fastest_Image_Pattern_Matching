// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "fpm/matcher.hpp"
#include <QImage>
#include <QJsonArray>
#include <QString>

// In a template-sized PNG, pure red marks the product colour, green the neutral
// reference, and optional blue a darker neutral reference. Black is ignored.
struct InspectionMask {
	QImage colour, reference, dark;
	QString path;
	static InspectionMask fromImage(const QImage& pixels, const QImage& templ, const QString& name);
	static InspectionMask read(const QString& path, const QImage& templ);
};
QJsonArray inspectColours(const QImage& source, const QImage& templ, const InspectionMask& mask,
	const std::vector<fpm::Match>& matches, const std::vector<double>& scales, double threshold);
