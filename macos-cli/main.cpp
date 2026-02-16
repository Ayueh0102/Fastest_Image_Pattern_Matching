#include "match_core.h"
#include <iostream>
#include <string>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QApplication>
#include "mainwindow.h"

using namespace std;
using namespace cv;

static void printUsage(const char* progName)
{
	cout << "Usage: " << progName << " [options]" << endl;
	cout << "  (no arguments)        Launch GUI" << endl;
	cout << "Options:" << endl;
	cout << "  --src <path>          Source image path (required for CLI)" << endl;
	cout << "  --tmpl <path>         Template image path (required for CLI)" << endl;
	cout << "  --target-num <n>      Max number of targets [70]" << endl;
	cout << "  --score <0-1>         Min match score [0.5]" << endl;
	cout << "  --angle <deg>         Tolerance angle [0]" << endl;
	cout << "  --overlap <0-0.8>     Max overlap ratio [0]" << endl;
	cout << "  --min-area <n>        Min reduce area [256]" << endl;
	cout << "  --no-simd             Disable SIMD optimization" << endl;
	cout << "  --subpixel            Enable subpixel estimation" << endl;
	cout << "  --invert              Invert source image" << endl;
	cout << "  --no-display          Don't show result window" << endl;
}

static bool hasCliArgs(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++)
	{
		string arg = argv[i];
		if (arg == "--src" || arg == "--tmpl")
			return true;
	}
	return false;
}

static int runCli(int argc, char* argv[])
{
	string srcPath, tmplPath;
	s_MatchConfig config;
	bool bDisplay = true;

	for (int i = 1; i < argc; i++)
	{
		string arg = argv[i];
		if (arg == "--src" && i + 1 < argc) srcPath = argv[++i];
		else if (arg == "--tmpl" && i + 1 < argc) tmplPath = argv[++i];
		else if (arg == "--target-num" && i + 1 < argc) config.iMaxPos = atoi(argv[++i]);
		else if (arg == "--score" && i + 1 < argc) config.dScore = atof(argv[++i]);
		else if (arg == "--angle" && i + 1 < argc) config.dToleranceAngle = atof(argv[++i]);
		else if (arg == "--overlap" && i + 1 < argc) config.dMaxOverlap = atof(argv[++i]);
		else if (arg == "--min-area" && i + 1 < argc) config.iMinReduceArea = atoi(argv[++i]);
		else if (arg == "--no-simd") config.bUseSIMD = false;
		else if (arg == "--subpixel") config.bSubPixel = true;
		else if (arg == "--invert") config.bBitwiseNot = true;
		else if (arg == "--no-display") bDisplay = false;
		else if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
		else { cerr << "Unknown option: " << arg << endl; printUsage(argv[0]); return 1; }
	}

	if (srcPath.empty() || tmplPath.empty())
	{
		cerr << "Error: --src and --tmpl are required." << endl;
		printUsage(argv[0]);
		return 1;
	}

	// Load images
	Mat matSrc = imread(srcPath, IMREAD_GRAYSCALE);
	if (matSrc.empty())
	{
		cerr << "Error: Cannot load source image: " << srcPath << endl;
		return 1;
	}

	Mat matDst = imread(tmplPath, IMREAD_GRAYSCALE);
	if (matDst.empty())
	{
		cerr << "Error: Cannot load template image: " << tmplPath << endl;
		return 1;
	}

	cout << "Source: " << matSrc.cols << "x" << matSrc.rows << endl;
	cout << "Template: " << matDst.cols << "x" << matDst.rows << endl;
	cout << "Config: TargetNum=" << config.iMaxPos
		 << " Score=" << config.dScore
		 << " Angle=" << config.dToleranceAngle
		 << " Overlap=" << config.dMaxOverlap
		 << " SIMD=" << (config.bUseSIMD ? "ON" : "OFF")
		 << " SubPixel=" << (config.bSubPixel ? "ON" : "OFF") << endl;

	// Learn pattern
	CMatchCore matcher;
	s_TemplData templData;
	matcher.LearnPattern(matDst, templData, config.iMinReduceArea);

	// Match
	auto t1 = chrono::high_resolution_clock::now();
	vector<s_SingleTargetMatch> results = matcher.Match(matSrc, matDst, templData, config);
	auto t2 = chrono::high_resolution_clock::now();
	double dTimeMs = chrono::duration<double, milli>(t2 - t1).count();

	cout << endl;
	cout << "Execution Time: " << (int)dTimeMs << " ms" << endl;
	cout << "Found: " << results.size() << " match(es)" << endl;
	cout << endl;

	// Print results table
	printf("%-6s %-10s %-12s %-14s %-14s\n", "Index", "Score", "Angle", "PosX", "PosY");
	printf("%-6s %-10s %-12s %-14s %-14s\n", "-----", "-----", "-----", "----", "----");
	for (int i = 0; i < (int)results.size(); i++)
	{
		printf("%-6d %-10.3f %-12.3f %-14.3f %-14.3f\n",
			i, results[i].dMatchScore, results[i].dMatchedAngle,
			results[i].ptCenter.x, results[i].ptCenter.y);
	}

	// Visualization
	if (bDisplay && !results.empty())
	{
		Mat matShow;
		cvtColor(matSrc, matShow, COLOR_GRAY2BGR);

		for (int i = 0; i < (int)results.size(); i++)
		{
			const s_SingleTargetMatch& r = results[i];
			Point ptLT((int)r.ptLT.x, (int)r.ptLT.y);
			Point ptRT((int)r.ptRT.x, (int)r.ptRT.y);
			Point ptRB((int)r.ptRB.x, (int)r.ptRB.y);
			Point ptLB((int)r.ptLB.x, (int)r.ptLB.y);
			Point ptC((int)r.ptCenter.x, (int)r.ptCenter.y);

			line(matShow, ptLT, ptRT, Scalar(0, 255, 0), 2, LINE_AA);
			line(matShow, ptRT, ptRB, Scalar(0, 255, 0), 2, LINE_AA);
			line(matShow, ptRB, ptLB, Scalar(0, 255, 0), 2, LINE_AA);
			line(matShow, ptLB, ptLT, Scalar(0, 255, 0), 2, LINE_AA);

			// Cross mark at center
			line(matShow, ptC - Point(10, 0), ptC + Point(10, 0), Scalar(0, 0, 255), 2);
			line(matShow, ptC - Point(0, 10), ptC + Point(0, 10), Scalar(0, 0, 255), 2);

			string str = to_string(i);
			putText(matShow, str, (ptLT + ptRT) / 2, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
		}

		// Resize for display if too large
		double dScale = 1.0;
		int maxDim = max(matShow.cols, matShow.rows);
		if (maxDim > 1600)
			dScale = 1600.0 / maxDim;

		Mat matDisplay;
		if (dScale < 1.0)
			resize(matShow, matDisplay, Size(), dScale, dScale);
		else
			matDisplay = matShow;

		imshow("Match Result", matDisplay);
		cout << "Press any key to close..." << endl;
		waitKey(0);
	}

	return 0;
}

int main(int argc, char* argv[])
{
	// If --help requested, always show usage
	for (int i = 1; i < argc; i++)
	{
		string arg = argv[i];
		if (arg == "--help" || arg == "-h")
		{
			printUsage(argv[0]);
			return 0;
		}
	}

	// CLI mode: when --src or --tmpl arguments are present
	if (hasCliArgs(argc, argv))
		return runCli(argc, argv);

	// GUI mode: no CLI arguments
	QApplication app(argc, argv);
	MainWindow w;
	w.show();
	return app.exec();
}
