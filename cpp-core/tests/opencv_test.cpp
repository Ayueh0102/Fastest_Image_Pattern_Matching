#include "internal.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void compareWarp(const fpm::Image& actual, const cv::Mat& expected, double& maxError, double& maxMean)
{
	double sum=0, largest=0;
	for (int y=0;y<expected.rows;++y) for (int x=0;x<expected.cols;++x) {
		const double error=std::abs(int(actual.row(y)[x])-int(expected.at<unsigned char>(y,x)));
		sum+=error; largest=std::max(largest,error);
	}
	const double mean=sum/expected.total();
	maxError=std::max(maxError,largest); maxMean=std::max(maxMean,mean);
	// OpenCV quantizes bilinear weights to 1/32; fpm uses full double precision.
	// For 8-bit input the two-axis weight error is bounded by 8 gray levels,
	// plus integer rounding. A wrong sign, anchor or padding is much larger.
	if (largest>9 || mean>1.5) throw std::runtime_error("affine sampling differs from independent OpenCV transform");
}

int main()
{
	try {
		cv::RNG rng(400);
		for (const auto size : {cv::Size(1,9),cv::Size(19,1),cv::Size(7,5),cv::Size(128,93)}) {
			cv::Mat source(size,CV_8U); rng.fill(source,cv::RNG::UNIFORM,0,256);
			cv::Mat expected; cv::pyrDown(source,expected);
			const auto actual=fpm::detail::pyrDown({source.data,source.cols,source.rows,source.step});
			for(int y=0;y<expected.rows;++y) for(int x=0;x<expected.cols;++x)
				if(actual.row(y)[x]!=expected.at<unsigned char>(y,x)) throw std::runtime_error("pyramid pixel mismatch");
		}
		double largest=0;
		for(int trial=0;trial<30;++trial) {
			cv::Mat parent(41,53,CV_8U); rng.fill(parent,cv::RNG::UNIFORM,0,256);
			const cv::Mat source=parent(cv::Rect(2,3,47,35)); // Noncontiguous source.
			const cv::Mat templ=source(cv::Rect(10,11,11+trial%7,9+trial%5));
			cv::Mat expected; cv::matchTemplate(source,templ,expected,cv::TM_CCOEFF_NORMED);
			const auto level=fpm::detail::learnLevel(fpm::Image({templ.data,templ.cols,templ.rows,templ.step}));
			const auto actual=fpm::detail::ncc({source.data,source.cols,source.rows,source.step},level,fpm::Backend::Auto);
			for(int y=0;y<expected.rows;++y) for(int x=0;x<expected.cols;++x)
				largest=std::max(largest,std::abs(actual.at(x,y)-expected.at<float>(y,x)));
		}
		if(largest>2e-5) throw std::runtime_error("NCC vs OpenCV exceeds 2e-5");
		std::cout<<"pyrDown exact; maximum NCC error = "<<largest<<'\n';
		double warpMax=0, warpMean=0;
		for (const auto size : {cv::Size(95,78),cv::Size(96,79)}) {
			cv::Mat parent(size.height+3,size.width+5,CV_8U); rng.fill(parent,cv::RNG::UNIFORM,0,256);
			const auto source=parent(cv::Rect(2,1,size.width,size.height));
			const fpm::ImageView view{source.data,source.cols,source.rows,source.step};
			for (double angle : {0.,90.,180.,-90.,21.3,60.,-119.95}) {
				const auto actual=fpm::detail::rotateCanvas(view,23,17,angle,211);
				const cv::Point2d center((size.width-1)*.5,(size.height-1)*.5);
				auto matrix=cv::getRotationMatrix2D(center,angle,1);
				matrix.at<double>(0,2)+=(actual.image.width()-1)*.5-center.x;
				matrix.at<double>(1,2)+=(actual.image.height()-1)*.5-center.y;
				cv::Mat expected;
				cv::warpAffine(source,expected,matrix,cv::Size(actual.image.width(),actual.image.height()),cv::INTER_LINEAR,cv::BORDER_CONSTANT,211);
				compareWarp(actual.image,expected,warpMax,warpMean);
				for (const auto anchor : {fpm::Point{24.37,31.69},fpm::Point{.3,-.2}}) {
					const auto roi=fpm::detail::rotatedRoi(view,23,17,anchor,angle);
					auto inverse=cv::getRotationMatrix2D(cv::Point2d(0,0),-angle,1);
					inverse.at<double>(0,2)=anchor.x-3*(inverse.at<double>(0,0)+inverse.at<double>(0,1));
					inverse.at<double>(1,2)=anchor.y-3*(inverse.at<double>(1,0)+inverse.at<double>(1,1));
					cv::warpAffine(source,expected,inverse,cv::Size(29,23),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT,0);
					compareWarp(roi,expected,warpMax,warpMean);
				}
			}
		}
		std::cout<<"rotation canvas/ROI: maximum gray error = "<<warpMax<<", largest mean = "<<warpMean<<'\n';
		return 0;
	} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
