// SPDX-License-Identifier: BSD-2-Clause
#include "color_inspection.h"
#include "color_illuminant.hpp"
#include <QJsonObject>
#include <cmath>
#include <memory>
#include <stdexcept>

InspectionMask InspectionMask::read(const QString& path, const QImage& templ)
{
	return fromImage(QImage(path),templ,path);
}

InspectionMask InspectionMask::fromImage(const QImage& pixels, const QImage& templ, const QString& path)
{
	const QSize size=templ.size();
	const auto rgb=pixels.convertToFormat(QImage::Format_RGB888);
	if(rgb.isNull() || rgb.size()!=size) throw std::invalid_argument("The colour mask must match the template size.");
	InspectionMask mask;
	mask.path=path;
	QImage* planes[]={&mask.colour,&mask.reference,&mask.dark};
	int count[3]{};
	for(auto* plane:planes) { *plane=QImage(size,QImage::Format_Grayscale8);plane->fill(0); }
	for(int y=0;y<size.height();++y) for(int x=0;x<size.width();++x) {
		const auto* p=rgb.constScanLine(y)+3*x;
		if(!p[0] && !p[1] && !p[2]) continue;
		const int channel=p[0]==255?0:(p[1]==255?1:2);
		if(p[channel]!=255 || p[(channel+1)%3] || p[(channel+2)%3])
			throw std::invalid_argument("Use only pure red, green, blue and black in the colour mask.");
		planes[channel]->scanLine(y)[x]=255;
		++count[channel];
	}
	if(!count[0] || !count[1]) throw std::invalid_argument("Mark both a red inspection region and a green reference.");
	if(!count[2]) mask.dark={};
	// Check reference ink when the mask is loaded, before the next geometric run.
	const auto templateRgb=templ.convertToFormat(QImage::Format_RGB888);
	using namespace fpm::research;
	const Rgb8View t{templateRgb.constBits(),size.width(),size.height(),size_t(templateRgb.bytesPerLine())};
	const Rgb8View pixel{t.data,1,1,3};
	for(const auto* ref:{&mask.reference,&mask.dark}) if(!ref->isNull())
		IlluminantChromaVerifier(pixel,t,mask.colour.constBits(),mask.colour.bytesPerLine(),ref->constBits(),ref->bytesPerLine());
	return mask;
}

QJsonArray inspectColours(const QImage& source, const QImage& templ, const InspectionMask& mask,
	const std::vector<fpm::Match>& matches, const std::vector<double>& scales, double threshold)
{
	using namespace fpm::research;
	const auto src=source.convertToFormat(QImage::Format_RGB888), tmp=templ.convertToFormat(QImage::Format_RGB888);
	const auto view=[](const QImage& img){return Rgb8View{img.constBits(),img.width(),img.height(),size_t(img.bytesPerLine())};};
	const auto makeVerifier=[&](const QImage& reference) {
		return std::make_unique<IlluminantChromaVerifier>(view(src),view(tmp),mask.colour.constBits(),
			mask.colour.bytesPerLine(),reference.constBits(),reference.bytesPerLine());
	};
	auto primary=makeVerifier(mask.reference);
	std::unique_ptr<IlluminantChromaVerifier> secondary;
	QJsonArray out;
	for(size_t i=0;i<matches.size();++i) {
		const auto& m=matches[i];
		std::array<float,4> pose{float(m.center.x-src.width()/2.),float(src.height()/2.-m.center.y),
			float(m.angle),float(std::log(scales.at(i)))};
		auto value=primary->score(pose);
		bool dark=false;
		if(value.status==3 && !mask.dark.isNull()) {
			if(!secondary) secondary=makeVerifier(mask.dark);
			value=secondary->score(pose);dark=true;
		}
		QJsonObject row{{"status",value.status},{"reference",dark?"dark":"primary"},
			{"valid_fraction",value.validFraction},{"reference_fraction",value.referenceFraction}};
		if(value.status==0) {
			row["region_distance"]=value.regionDistance;
			row["decision"]=threshold==0?"measured":(value.regionDistance<=threshold?"within":"outside");
		} else row["decision"]="review";
		out.append(row);
	}
	return out;
}
