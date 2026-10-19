// SPDX-License-Identifier: BSD-2-Clause
#include "../tools/qt_gray_loader.hpp"
#include <QColor>
#include <QCoreApplication>
#include <QFile>
#include <QImage>
#include <QTemporaryDir>
#include <QString>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message)
{
	if (!condition) throw std::runtime_error(message);
}

// Independent copy of the runner's pre-change loading semantics. In particular,
// let Qt handle palette, alpha and 16-bit conversion before the fixed RGB weights.
// Do not compare a lossy file decoder with the pixels before encoding.
fpm::Image legacyOracle(const QImage& input)
{
	const QImage rgb = input.convertToFormat(QImage::Format_RGB888);
	if (rgb.isNull()) throw std::invalid_argument("Cannot decode image");
	fpm::Image output(rgb.width(), rgb.height());
	for (int y = 0; y < rgb.height(); ++y) {
		for (int x = 0; x < rgb.width(); ++x) {
			const auto* pixel = rgb.constScanLine(y) + 3 * x;
			output.row(y)[x] = std::uint8_t((77u * pixel[0] + 150u * pixel[1] +
				29u * pixel[2] + 128u) >> 8);
		}
	}
	return output;
}

void equalImages(const fpm::Image& actual, const fpm::Image& expected,
	const std::string& name)
{
	require(actual.width() == expected.width() && actual.height() == expected.height(),
		name + ": dimensions differ");
	for (int y = 0; y < actual.height(); ++y) {
		for (int x = 0; x < actual.width(); ++x) {
			require(actual.row(y)[x] == expected.row(y)[x], name + ": pixel mismatch at " +
				std::to_string(x) + "," + std::to_string(y));
		}
	}
}

void equalInfo(const fpm::io::GrayLoadInfo& info, const QImage& image,
	bool expectDirect, const std::string& name)
{
	require(info.directGray8 == expectDirect, name + ": wrong direct-gray indicator");
	const std::size_t expectedRgbBytes = expectDirect ? 0 :
		std::size_t(image.convertToFormat(QImage::Format_RGB888).sizeInBytes());
	require(info.rgbTemporaryBytes == expectedRgbBytes, name + ": wrong RGB temporary byte count");
}

void compareImage(const QImage& image, const std::string& name)
{
	require(!image.isNull(), name + ": invalid fixture");
	const auto originalFormat = image.format();
	// Keep independent bytes, not another implicitly shared QImage: this also
	// detects writes through a borrowed scanline pointer in the loader.
	const std::vector<unsigned char> originalBytes(image.constBits(),
		image.constBits() + image.sizeInBytes());
	const auto expected = legacyOracle(image);
	const bool direct = image.format() == QImage::Format_Grayscale8;
	fpm::io::GrayLoadInfo info{!direct, 12345};
	equalImages(fpm::io::grayFromQImage(image, fpm::io::GrayLoadMode::DirectGray8, &info),
		expected, name + " direct/default policy");
	equalInfo(info, image, direct, name);
	equalImages(fpm::io::grayFromQImage(image), expected, name + " default arguments");
	info = {true, 12345};
	equalImages(fpm::io::grayFromQImage(image, fpm::io::GrayLoadMode::LegacyRgb, &info),
		expected, name + " forced legacy");
	equalInfo(info, image, false, name + " forced legacy");
	require(image.format() == originalFormat &&
		std::equal(originalBytes.begin(), originalBytes.end(), image.constBits()),
		name + ": input pixels, padding or format changed");
}

QImage grayFixture(int width, int height)
{
	QImage image(width, height, QImage::Format_Grayscale8);
	require(!image.isNull(), "Cannot allocate gray fixture");
	// Initialize padding too, so accidental copying of bytesPerLine instead of
	// width is detected by the next row (and by an instrumented build).
	image.fill(219);
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
			image.scanLine(y)[x] = std::uint8_t((x + 37 * y) & 255);
	return image;
}

QImage colorFixture()
{
	QImage image(13, 9, QImage::Format_RGBA8888);
	const int alpha[] = {0, 1, 85, 128, 254, 255};
	for (int y = 0; y < image.height(); ++y) {
		for (int x = 0; x < image.width(); ++x) {
			image.setPixelColor(x, y, QColor((23 * x + 67 * y) & 255,
				(83 * x + 11 * y + 127) & 255, (41 * x + 97 * y + 53) & 255,
				alpha[(x + y) % 6]));
		}
	}
	return image;
}

void testGrayAndStride()
{
	const auto allValues = grayFixture(256, 3);
	compareImage(allValues, "all 256 gray values");
	const auto converted = fpm::io::grayFromQImage(allValues);
	for (int value = 0; value < 256; ++value)
		require(converted.row(0)[value] == value, "Gray value changed");
	for (int width : {1, 3, 5, 17, 257})
		compareImage(grayFixture(width, 7), "padded gray width " + std::to_string(width));

	constexpr int width = 7, height = 5, stride = 20;
	// 32-bit alignment satisfies QImage's external-memory contract. The stride
	// deliberately exceeds both width and Qt's usual rounded row length.
	std::vector<std::uint32_t> storage(stride * height / sizeof(std::uint32_t), 0xe7e7e7e7u);
	auto* bytes = reinterpret_cast<unsigned char*>(storage.data());
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
			bytes[y * stride + x] = std::uint8_t(17 * x + 29 * y);
	const auto before = storage;
	QImage external(bytes, width, height, stride, QImage::Format_Grayscale8);
	compareImage(external, "external gray stride");
	require(storage == before, "Loader modified external input or padding");
	const auto expected = legacyOracle(external);
	auto owned = fpm::io::grayFromQImage(external);
	std::fill(storage.begin(), storage.end(), 0);
	equalImages(owned, expected, "output owns pixels after source mutation");

	const auto* constBytes = reinterpret_cast<const unsigned char*>(before.data());
	const QImage readOnly(constBytes, width, height, stride, QImage::Format_Grayscale8);
	compareImage(readOnly, "read-only external gray stride");
}

void testFallbackFormats()
{
	const auto rgba = colorFixture();
	for (const auto format : {QImage::Format_RGB888, QImage::Format_RGB32,
		QImage::Format_RGBA8888, QImage::Format_ARGB32,
		QImage::Format_RGBA8888_Premultiplied, QImage::Format_ARGB32_Premultiplied}) {
		const auto image = rgba.convertToFormat(format);
		require(image.format() == format, "Color fixture conversion did not retain format");
		compareImage(image, "color/alpha format " + std::to_string(int(format)));
	}

	QImage indexed(19, 7, QImage::Format_Indexed8);
	QList<QRgb> palette;
	for (int i = 0; i < 256; ++i)
		palette.append(qRgba((3 * i + 19) & 255, (255 - i), (37 * i + 1) & 255,
			(i % 5 == 0) ? 0 : 255));
	indexed.setColorTable(palette);
	for (int y = 0; y < indexed.height(); ++y)
		for (int x = 0; x < indexed.width(); ++x)
			indexed.scanLine(y)[x] = std::uint8_t(11 * x + 31 * y);
	compareImage(indexed, "non-identity palette including transparency");

	// Even an identity palette is intentionally a fallback: the optimization
	// contract is restricted to the decoded Grayscale8 format.
	for (int i = 0; i < 256; ++i) palette[i] = qRgb(i, i, i);
	indexed.setColorTable(palette);
	compareImage(indexed, "identity indexed palette still uses fallback");
	for (const auto format : {QImage::Format_Mono, QImage::Format_MonoLSB}) {
		QImage mono(19, 7, format);
		mono.setColorTable({qRgb(233, 22, 51), qRgb(17, 211, 89)});
		mono.fill(0);
		for (int y = 0; y < mono.height(); ++y)
			for (int x = 0; x < mono.width(); ++x)
				mono.setPixel(x, y, (x + 3 * y) % 2);
		compareImage(mono, "monochrome format " + std::to_string(int(format)));
	}

	QImage gray16(13, 9, QImage::Format_Grayscale16);
	for (int y = 0; y < gray16.height(); ++y) {
		auto* row = reinterpret_cast<quint16*>(gray16.scanLine(y));
		for (int x = 0; x < gray16.width(); ++x)
			row[x] = quint16((9001 * x + 3293 * y) & 65535);
	}
	compareImage(gray16, "16-bit gray fallback");
}

template<class Function>
void expectInvalid(Function function, const std::string& name)
{
	bool caught = false;
	try { function(); }
	catch (const std::invalid_argument&) { caught = true; }
	require(caught, name + ": expected invalid_argument");
}

void testFilesAndErrors()
{
	QTemporaryDir directory;
	require(directory.isValid(), "Cannot create image test directory");
	const QImage gray = grayFixture(257, 11);
	const QImage color = colorFixture().convertToFormat(QImage::Format_RGB888);
	for (const char* format : {"PNG", "BMP", "JPEG"}) {
		for (const auto* image : {&gray, &color}) {
			const QString path = directory.filePath(QString("%1-%2.%3")
				.arg(image == &gray ? "gray" : "color").arg(image->width()).arg(format));
			require(image->save(path, format, 95), std::string("Cannot encode required format ") + format);
			const QImage decoded(path);
			require(!decoded.isNull(), "Cannot independently decode fixture");
			const auto expected = legacyOracle(decoded);
			for (const auto mode : {fpm::io::GrayLoadMode::DirectGray8, fpm::io::GrayLoadMode::LegacyRgb}) {
				fpm::io::GrayLoadInfo info{true, 12345};
				equalImages(fpm::io::readGrayImage(path, mode, &info), expected,
					path.toStdString() + " file roundtrip");
				equalInfo(info, decoded, mode == fpm::io::GrayLoadMode::DirectGray8 &&
					decoded.format() == QImage::Format_Grayscale8, path.toStdString());
			}
			equalImages(fpm::io::readGrayImage(path), expected, path.toStdString() + " default arguments");
		}
	}
	const QString invalidPath = directory.filePath("invalid.png");
	QFile invalid(invalidPath);
	require(invalid.open(QIODevice::WriteOnly), "Cannot create invalid image fixture");
	require(invalid.write("not an image\n") == 13, "Cannot write invalid image fixture");
	invalid.close();
	for (const auto mode : {fpm::io::GrayLoadMode::DirectGray8, fpm::io::GrayLoadMode::LegacyRgb}) {
		expectInvalid([&] { fpm::io::grayFromQImage(QImage(), mode); }, "null image");
		expectInvalid([&] { fpm::io::readGrayImage(invalidPath, mode); }, "invalid image file");
		expectInvalid([&] { fpm::io::readGrayImage(directory.filePath("missing.png"), mode); }, "missing image file");
	}
	expectInvalid([&] { fpm::io::grayFromQImage(gray, static_cast<fpm::io::GrayLoadMode>(99)); },
		"invalid loading mode");
}

} // namespace

int main(int argc, char** argv)
{
	QCoreApplication application(argc, argv);
	try {
		testGrayAndStride();
		testFallbackFormats();
		testFilesAndErrors();
		std::cout << "Qt gray loader: all gray values, padded/external stride, format fallbacks, "
			"PNG/BMP/JPEG and errors match the independent legacy oracle\n";
		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
