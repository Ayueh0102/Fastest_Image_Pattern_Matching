// SPDX-License-Identifier: BSD-2-Clause
// Self-contained test runner so the reference core builds/tests offline.
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace fpm;
using namespace fpm::detail;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void near(double value, double expected, double tolerance = 1e-9)
{
	if (!std::isfinite(value) || std::abs(value - expected) > tolerance)
		throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(value));
}
template<class F> void invalid(F f)
{
	try { f(); } catch (const std::invalid_argument&) { return; }
	throw std::runtime_error("expected invalid_argument");
}
Image noise(int w, int h, unsigned seed)
{
	Image out(w, h);
	std::mt19937 rng(seed);
	for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) out.row(y)[x] = std::uint8_t(rng() % 256);
	return out;
}
void paste(Image& out, const Image& patch, int x, int y)
{
	for (int row = 0; row < patch.height(); ++row) std::copy_n(patch.row(row), patch.width(), out.row(y + row) + x);
}
void imageAndPyramid()
{
	near(grayFromRgb(255,0,0),77); near(grayFromRgb(0,255,0),149);
	near(grayFromRgb(0,0,255),29); near(grayFromRgb(255,255,255),255);
	std::vector<std::uint8_t> storage(5 * 9, 77);
	ImageView view{storage.data(), 5, 5, 9};
	Image owned(view);
	storage[0] = 0;
	require(owned.row(0)[0] == 77, "image must own a copy");
	auto small = pyrDown(owned.view());
	require(small.width() == 3 && small.height() == 3, "odd dimensions use ceil half");
	for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x) near(small.row(y)[x], 77);
	Image ramp(5, 1);
	for (int x = 0; x < 5; ++x) ramp.row(0)[x] = x * 16;
	const auto reduced = pyrDown(ramp.view());
	near(reduced.row(0)[0], 12); near(reduced.row(0)[1], 32); near(reduced.row(0)[2], 52);
	Pattern p(owned.view(), 4);
	require(p.levels().size() == 3, "legacy area-based pyramid depth");
	near(p.levels()[0].norm, 0);
	invalid([&] { Pattern bad(view, 0); });
	invalid([&] { Pattern bad({storage.data(), 5, 5, 4}); });
	invalid([] { Pattern bad({}); });
}
void nccReference()
{
	const auto image = noise(19, 17, 10), patch = noise(7, 5, 20);
	const auto templ = learnLevel(Image(patch.view()));
	std::vector<std::uint8_t> padded(17 * 25);
	for (int y = 0; y < 17; ++y) std::copy_n(image.row(y), 19, padded.data() + y * 25);
	const auto scores = ncc({padded.data(), 19, 17, 25}, templ, Backend::Scalar);
	const auto simd = ncc(image.view(), templ, Backend::Auto);
	for (int y = 0; y < scores.height; ++y) for (int x = 0; x < scores.width; ++x) {
		double mean = 0;
		for (int j = 0; j < 5; ++j) for (int i = 0; i < 7; ++i) mean += image.row(y+j)[x+i];
		mean /= 35;
		double cross = 0, variance = 0;
		for (int j = 0; j < 5; ++j) for (int i = 0; i < 7; ++i) {
			const double d = image.row(y+j)[x+i] - mean;
			cross += d * (patch.row(j)[i] - templ.mean); variance += d*d;
		}
		near(scores.at(x,y), cross / std::sqrt(variance) / templ.norm);
		near(simd.at(x,y), scores.at(x,y), 0);
	}
	Image flat(7,5);
	const auto constantTemplate = ncc(image.view(), learnLevel(flat), Backend::Auto);
	for (double v : constantTemplate.values) near(v, 1);
	const auto constantSource = ncc(Image(19,17).view(), templ, Backend::Auto);
	for (double v : constantSource.values) near(v, 0);
	const auto exact = ncc(patch.view(), templ, Backend::Auto);
	near(exact.at(0,0), 1);
	Image inverted(patch.view());
	for (int y=0;y<5;++y) for (int x=0;x<7;++x) inverted.row(y)[x] = 255-inverted.row(y)[x];
	near(ncc(inverted.view(), templ, Backend::Auto).at(0,0), -1);
}
void simdOverflow()
{
	std::vector<std::uint8_t> a(200003,255), b(a);
	for (int length : {1,15,16,17,4096,4097,33025,200003}) {
		const std::uint64_t expected = std::uint64_t(length)*255*255;
		require(dotProduct(a.data(),b.data(),length,Backend::Scalar)==expected,"scalar overflow");
		require(dotProduct(a.data(),b.data(),length,Backend::Auto)==expected,"SIMD overflow/tail");
	}
}
void peakExtraction()
{
	ScoreMap map{100,30,std::vector<double>(3000,0)};
	for(int y=0;y<20;++y) for(int x=0;x<20;++x) map.at(x,y)=.99;
	map.at(90,25)=.90;
	for (bool block : {false,true}) {
		const auto peaks=selectPeaks(map,25,25,2,0,.8,block);
		require(peaks.size()==2,"broad first peak must not hide second target");
		require(peaks[1].x==90 && peaks[1].y==25,"second target location");
	}
	std::mt19937 rng(100);
	for (auto& v : map.values) v=double(rng()%100)/100;
	const auto scan=selectPeaks(map,13,7,60,.63,.3,false);
	const auto block=selectPeaks(map,13,7,60,.63,.3,true);
	require(scan.size()==block.size(),"BlockMax count");
	for(size_t i=0;i<scan.size();++i) require(scan[i].x==block[i].x && scan[i].y==block[i].y && scan[i].score==block[i].score,"BlockMax exact tie order");
	ScoreMap tiny{2,1,{1,.9}};
	const auto peaks=selectPeaks(tiny,1,1,5,1,0,true);
	require(peaks.size()==2,"overlap=1 must always consume a peak");
}
void geometry()
{
	const auto a=makeMatch({0,0},10,10,0,.99), b=makeMatch({4,0},10,10,0,.95);
	near(overlap(a,b),.6);
	require(suppress({a,b},5,.5).size()==1,"intersection/template-area NMS");
	require(suppress({a,b},5,.7).size()==2,"allowed overlap");
	require(suppress({a,a},5,1).size()==1,"full duplicates");
	near(overlap(a,makeMatch({10,0},10,10,0,1)),0);
	near(overlap(a,makeMatch({10,0},10,10,90,1)),1);
	const double root=std::sqrt(2.0);
	near(overlap(a,makeMatch({5,5-5*root},10,10,45,1)),2*root-2);
	const auto p=rotate({12,35},{8,9},-64);
	const auto original=rotate(p,{8,9},64);
	near(original.x,12); near(original.y,35);
	near(normalizeAngle(-180),180);
}
void subpixel()
{
	std::array<double,27> cube{}; int i=0;
	for(int t=-1;t<=1;++t) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x)
		cube[i++]=1-.2*(x-.2)*(x-.2)-.3*(y+.3)*(y+.3)-.1*(t-.4)*(t-.4);
	std::array<double,3> offset{};
	require(refine3d(cube,offset),"3D fit");
	near(offset[0],.2); near(offset[1],-.3); near(offset[2],.4);
	std::array<double,9> patch{}; i=0;
	for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) patch[i++]=1-.2*(x-.25)*(x-.25)-.3*(y+.4)*(y+.4);
	Point p;
	require(refine2d(patch,p),"2D fit"); near(p.x,.25); near(p.y,-.4);
	cube.fill(1); require(!refine3d(cube,offset),"flat fit rejected");
	i=0;
	for(int t=-1;t<=1;++t) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) cube[i++]=x*x-y*y-t*t;
	require(!refine3d(cube,offset),"saddle rejected");
	cube[0]=std::numeric_limits<double>::quiet_NaN();
	require(!refine3d(cube,offset),"nonfinite fit rejected");
}
void matching()
{
	Image source=noise(161,123,1), templ=noise(31,23,2);
	paste(source,templ,20,30); paste(source,templ,100,80);
	Pattern pattern(templ.view(),64);
	MatchConfig config; config.minScore=.99; config.maxTargets=5;
	const auto matches=fpm::match(source.view(),pattern,config);
	require(matches.size()==2,"two planted targets");
	near(matches[0].center.x,35.5); near(matches[0].center.y,41.5);
	near(matches[1].center.x,115.5); near(matches[1].center.y,91.5);
	for (const auto& m:matches) { near(m.angle,0); near(m.score,1); }
	config.stopLayer=1; config.minScore=.5; config.maxTargets=2;
	const auto fast=fpm::match(source.view(),pattern,config);
	require(fast.size()==2,"FastMode two targets");
	for(const auto& m:fast) {
		require(std::min(std::hypot(m.center.x-35.5,m.center.y-41.5),std::hypot(m.center.x-115.5,m.center.y-91.5))<=2,"FastMode full-size output coordinates");
		near(std::hypot(m.corners[1].x-m.corners[0].x,m.corners[1].y-m.corners[0].y),31);
	}
	config.stopLayer=2;
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.stopLayer=0; config.minScore=.99; config.maxTargets=5;
	for(int y=0;y<source.height();++y) for(int x=0;x<source.width();++x) source.row(y)[x]=255-source.row(y)[x];
	config.invertSource=true;
	require(fpm::match(source.view(),pattern,config).size()==2,"source inversion");
	config.minScore=std::numeric_limits<double>::quiet_NaN();
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.minScore=.99; config.maxTargets=0;
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.maxTargets=1; config.angleRanges={{20,10}};
	invalid([&]{ fpm::match(source.view(),pattern,config); });
	config.angleRanges.clear();
	require(fpm::match(Image(2,2).view(),pattern,config).empty(),"oversized template");
}
void rotatedMatching()
{
	Image templ=noise(23,17,30), source=noise(100,90,31);
	for(int y=0;y<17;++y) for(int x=0;x<23;++x) source.row(30+22-x)[40+y]=templ.row(y)[x];
	Pattern pattern(templ.view(),256);
	MatchConfig c; c.minScore=.95; c.maxTargets=1; c.angleRanges={{90,90},{-30,-20}};
	const auto matches=fpm::match(source.view(),pattern,c);
	require(matches.size()==1,"two angle ranges find +90 object");
	near(matches[0].angle,90); near(matches[0].center.x,48.5); near(matches[0].center.y,40.5);
	c.angleRanges={{-90,-90}};
	require(fpm::match(source.view(),pattern,c).empty(),"excluded angle");
	// No-pyramid path and subpixel at fixed angle must also be safe.
	Pattern flat(templ.view(),10000); c.angleRanges={{90,90}}; c.subpixel=true;
	const auto fine=fpm::match(source.view(),flat,c);
	require(fine.size()==1,"layer-zero refinement");
	near(fine[0].center.x,48.5,.2); near(fine[0].center.y,40.5,.2);
}

void sixRotatedCopies()
{
	// Independently render six actual copies. This catches angle-sign, pyramid
	// propagation and candidate loss without assuming every rotor arm is equal.
	auto intensity=[](double x,double y) {
		return 20 + 150*std::exp(-((x-16)*(x-16)/50+(y-13)*(y-13)/24)) +
			110*std::exp(-((x-34)*(x-34)/22+(y-24)*(y-24)/40)) +
			75*std::exp(-((x-14)*(x-14)/16+(y-30)*(y-30)/12));
	};
	Image templ(51,39), source(380,260);
	for (int y=0;y<39;++y) for (int x=0;x<51;++x)
		templ.row(y)[x]=std::uint8_t(std::lround(intensity(x,y)));
	const double angles[]={0,60,120,180,-120,-60};
	const Point centers[]={{70.3,70.4},{180.3,70.4},{290.3,70.4},{70.3,180.4},{180.3,180.4},{290.3,180.4}};
	for (int y=0;y<source.height();++y) for (int x=0;x<source.width();++x) {
		double value=20;
		for (int i=0;i<6;++i) {
			const double a=angles[i]*3.14159265358979323846/180;
			const double dx=x-centers[i].x, dy=y-centers[i].y;
			// Inverse of the image-space CCW pose; no production rotate/warp call.
			const double tx=std::cos(a)*dx-std::sin(a)*dy+25.5;
			const double ty=std::sin(a)*dx+std::cos(a)*dy+19.5;
			value+=intensity(tx,ty)-20;
		}
		source.row(y)[x]=std::uint8_t(std::clamp(std::lround(value),0L,255L));
	}
	for (int area : {64,256,1024}) for (bool subpixel : {false,true}) {
		Pattern pattern(templ.view(),area);
		MatchConfig config; config.maxTargets=10; config.minScore=.95; config.toleranceAngle=180;
		config.subpixel=subpixel;
		const auto matches=fpm::match(source.view(),pattern,config);
		require(matches.size()==6,"six true rotated copies must survive coarse-to-fine search at score .95");
		bool seen[6]{};
		double maxPositionError=0, maxAngleError=0, minScore=1;
		for (const auto& m : matches) {
			int found=-1;
			// Integer angular sampling at this template width is atan(2/51)
			// = 2.246 degrees. This test verifies recall/distinct poses at that
			// resolution; a quadratic fit is not an accuracy guarantee for every
			// texture. The separate fractionalPose test checks known fit accuracy.
			const bool precise=subpixel && area>=256;
			const double angleTolerance=precise ? .1 : std::atan(2./51)*180/3.14159265358979323846;
			for (int i=0;i<6;++i) if (!seen[i] && std::hypot(m.center.x-centers[i].x,m.center.y-centers[i].y)<(precise?.02:1.5) &&
				std::abs(std::remainder(m.angle-angles[i],360.0))<angleTolerance) { found=i; break; }
			if (found<0) std::cerr<<"six-copy rejected area="<<area<<" subpixel="<<subpixel<<" x="<<m.center.x<<" y="<<m.center.y<<" angle="<<m.angle<<" score="<<m.score<<'\n';
			require(found>=0,"unexpected six-copy pose or duplicate"); seen[found]=true;
			maxPositionError=std::max(maxPositionError,std::hypot(m.center.x-centers[found].x,m.center.y-centers[found].y));
			maxAngleError=std::max(maxAngleError,std::abs(std::remainder(m.angle-angles[found],360.0)));
			minScore=std::min(minScore,m.score);
		}
		std::cout<<"six-copy area "<<area<<", subpixel "<<subpixel<<": max position error "<<maxPositionError
			<<", max angle error "<<maxAngleError<<", minimum score "<<minScore<<'\n';
	}
}

void fractionalPose()
{
	// Independently render a continuous asymmetric texture at a known pose.
	// This checks end-to-end coordinate/angle precision, not just the fitter.
	auto intensity=[](double x,double y) {
		return std::uint8_t(std::clamp(std::lround(20 +
			150*std::exp(-((x-16)*(x-16)/50+(y-13)*(y-13)/24)) +
			110*std::exp(-((x-34)*(x-34)/22+(y-24)*(y-24)/40)) +
			75*std::exp(-((x-14)*(x-14)/16+(y-30)*(y-30)/12))),0L,255L));
	};
	Image templ(51,39);
	for(int y=0;y<39;++y) for(int x=0;x<51;++x) templ.row(y)[x]=intensity(x,y);
	for(double angle : {0.0,21.3}) {
		const Point origin{50.3,70.4};
		Image source(150,130);
		for(int y=0;y<130;++y) for(int x=0;x<150;++x) {
			const auto p=rotate({x-origin.x,y-origin.y},{0,0},-angle);
			source.row(y)[x]=intensity(p.x,p.y);
		}
		Pattern pattern(templ.view(),256);
		MatchConfig c; c.maxTargets=1; c.minScore=.9; c.toleranceAngle=angle==0?0:40;
		const auto integer=fpm::match(source.view(),pattern,c);
		c.subpixel=true;
		const auto fractional=fpm::match(source.view(),pattern,c);
		require(integer.size()==1 && fractional.size()==1,"fractional pose detected");
		const auto truth=makeMatch(origin,51,39,-angle,1);
		auto error=[&](const Match& m) {return std::hypot(m.center.x-truth.center.x,m.center.y-truth.center.y);};
		std::cout << "fractional pose " << angle << ": integer error " << error(integer[0]) << ", refined error " << error(fractional[0]) << ", angle " << fractional[0].angle << '\n';
		require(error(fractional[0])<error(integer[0]),"subpixel improves known position");
		require(error(fractional[0])<.25,"subpixel position within a quarter pixel");
		near(fractional[0].angle,angle,.5);
	}
}
}
int main()
{
	const std::pair<const char*,std::function<void()>> tests[]={{"image/pyramid",imageAndPyramid},{"NCC/reference/stride",nccReference},{"SIMD/overflow/tails",simdOverflow},{"peaks/BlockMax",peakExtraction},{"geometry/NMS",geometry},{"match/six-rotated-copies",sixRotatedCopies},{"subpixel/fits",subpixel},{"match/validation/invert",matching},{"rotation/ranges/layer0",rotatedMatching},{"subpixel/known-fractional-pose",fractionalPose}};
	int failed=0;
	for(const auto& test:tests) {
		try { test.second(); std::cout<<"PASS "<<test.first<<'\n'; }
		catch(const std::exception& e) { ++failed; std::cerr<<"FAIL "<<test.first<<": "<<e.what()<<'\n'; }
	}
	return failed ? 1 : 0;
}
