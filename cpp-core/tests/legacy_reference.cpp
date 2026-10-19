// Audit-only driver: unchanged macOS algorithm, without Qt or display code.
#include "match_core.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc,char** argv)
{
	try {
		std::string srcPath,tmplPath; s_MatchConfig c;
		for(int i=1;i<argc;++i) {
			const std::string arg=argv[i];
			auto value=[&] { if(i+1>=argc) throw std::runtime_error("missing argument"); return std::string(argv[++i]); };
			if(arg=="--src") srcPath=value(); else if(arg=="--tmpl") tmplPath=value();
			else if(arg=="--target-num") c.iMaxPos=std::stoi(value());
			else if(arg=="--score") c.dScore=std::stod(value());
			else if(arg=="--overlap") c.dMaxOverlap=std::stod(value());
			else if(arg=="--angle") c.dToleranceAngle=std::stod(value());
			else if(arg=="--min-area") c.iMinReduceArea=std::stoi(value());
			else if(arg=="--subpixel") c.bSubPixel=true;
			else if(arg=="--no-simd") c.bUseSIMD=false;
			else throw std::runtime_error("unsupported reference argument");
		}
		const auto src=cv::imread(srcPath,cv::IMREAD_GRAYSCALE),tmpl=cv::imread(tmplPath,cv::IMREAD_GRAYSCALE);
		if(src.empty() || tmpl.empty()) throw std::runtime_error("missing input image");
		CMatchCore matcher; s_TemplData pattern; matcher.LearnPattern(tmpl,pattern,c.iMinReduceArea);
		const auto t0=std::chrono::steady_clock::now();
		const auto results=matcher.Match(src,tmpl,pattern,c);
		const auto t1=std::chrono::steady_clock::now();
		std::cout<<std::setprecision(12)<<"{\"match_ms\":"<<std::chrono::duration<double,std::milli>(t1-t0).count()<<",\"matches\":[";
		for(size_t i=0;i<results.size();++i) {
			const auto& r=results[i]; if(i) std::cout<<',';
			std::cout<<"{\"score\":"<<r.dMatchScore<<",\"angle\":"<<r.dMatchedAngle<<",\"pos_x\":"<<r.ptCenter.x<<",\"pos_y\":"<<r.ptCenter.y<<'}';
		}
		std::cout<<"]}\n"; return 0;
	} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
