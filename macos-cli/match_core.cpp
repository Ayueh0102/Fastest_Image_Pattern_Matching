#include "match_core.h"
#include <algorithm>
#include <iostream>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

// Comparators
static bool compareScoreBig2Small(const s_MatchParameter& lhs, const s_MatchParameter& rhs)
{
	return lhs.dMatchScore > rhs.dMatchScore;
}
static bool comparePtWithAngle(const pair<Point2f, double>& lhs, const pair<Point2f, double>& rhs)
{
	return lhs.second < rhs.second;
}

// ARM Neon SIMD convolution
#if HAS_NEON
static inline int32_t neon_hsum_s32(int32x4_t v)
{
	int32x2_t tmp = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
	return vget_lane_s32(vpadd_s32(tmp, tmp), 0);
}

// Apple Silicon (M1+) supports ARMv8.4 dot product: vdotq_u32
// This processes 16 bytes per iteration with a single instruction (4x faster than widening multiply)
#if defined(__ARM_FEATURE_DOTPROD)
static inline int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength)
{
	const int iBlockSize = 16, Block = iLength / iBlockSize;
	uint32x4_t SumV = vdupq_n_u32(0);

	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		uint8x16_t SrcK = vld1q_u8(pCharKernel + Y);
		uint8x16_t SrcC = vld1q_u8(pCharConv + Y);
		SumV = vdotq_u32(SumV, SrcK, SrcC);
	}

	int32_t Sum = (int32_t)vaddvq_u32(SumV);

	for (int Y = Block * iBlockSize; Y < iLength; Y++)
		Sum += pCharKernel[Y] * pCharConv[Y];

	return Sum;
}
#else
// Fallback: widening multiply for older ARM without dot product
static inline int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength)
{
	const int iBlockSize = 16, Block = iLength / iBlockSize;
	int32x4_t SumV = vdupq_n_s32(0);

	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		uint8x16_t SrcK = vld1q_u8(pCharKernel + Y);
		uint8x16_t SrcC = vld1q_u8(pCharConv + Y);
		int16x8_t SrcK_L = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(SrcK)));
		int16x8_t SrcK_H = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(SrcK)));
		int16x8_t SrcC_L = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(SrcC)));
		int16x8_t SrcC_H = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(SrcC)));
		SumV = vaddq_s32(SumV, vmull_s16(vget_low_s16(SrcK_L), vget_low_s16(SrcC_L)));
		SumV = vaddq_s32(SumV, vmull_s16(vget_high_s16(SrcK_L), vget_high_s16(SrcC_L)));
		SumV = vaddq_s32(SumV, vmull_s16(vget_low_s16(SrcK_H), vget_low_s16(SrcC_H)));
		SumV = vaddq_s32(SumV, vmull_s16(vget_high_s16(SrcK_H), vget_high_s16(SrcC_H)));
	}

	int32_t Sum = neon_hsum_s32(SumV);

	for (int Y = Block * iBlockSize; Y < iLength; Y++)
		Sum += pCharKernel[Y] * pCharConv[Y];

	return Sum;
}
#endif
#endif

// ============ CMatchCore Implementation ============

void CMatchCore::LearnPattern(const Mat& matDst, s_TemplData& templData, int iMinReduceArea)
{
	templData.clear();

	int iTopLayer = GetTopLayer(&matDst, (int)sqrt((double)iMinReduceArea));
	buildPyramid(matDst, templData.vecPyramid, iTopLayer);
	templData.iBorderColor = mean(matDst).val[0] < 128 ? 255 : 0;
	int iSize = (int)templData.vecPyramid.size();
	templData.resize(iSize);

	for (int i = 0; i < iSize; i++)
	{
		double invArea = 1.0 / ((double)templData.vecPyramid[i].rows * templData.vecPyramid[i].cols);
		Scalar templMean, templSdv;
		double templNorm = 0, templSum2 = 0;

		meanStdDev(templData.vecPyramid[i], templMean, templSdv);
		templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] +
					templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

		if (templNorm < DBL_EPSILON)
			templData.vecResultEqual1[i] = true;

		templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] +
					templMean[2] * templMean[2] + templMean[3] * templMean[3];

		templSum2 /= invArea;
		templNorm = std::sqrt(templNorm);
		templNorm /= std::sqrt(invArea);

		templData.vecInvArea[i] = invArea;
		templData.vecTemplMean[i] = templMean;
		templData.vecTemplNorm[i] = templNorm;
	}
	templData.bIsPatternLearned = true;
}

int CMatchCore::GetTopLayer(const Mat* matTempl, int iMinDstLength)
{
	int iTopLayer = 0;
	int iMinReduceArea = iMinDstLength * iMinDstLength;
	int iArea = matTempl->cols * matTempl->rows;
	while (iArea > iMinReduceArea)
	{
		iArea /= 4;
		iTopLayer++;
	}
	return iTopLayer;
}

vector<s_SingleTargetMatch> CMatchCore::Match(const Mat& matSrc, const Mat& matDst,
	s_TemplData& templData, const s_MatchConfig& config)
{
	vector<s_SingleTargetMatch> results;

	if (matSrc.empty() || matDst.empty())
		return results;
	if ((matDst.cols < matSrc.cols && matDst.rows > matSrc.rows) ||
		(matDst.cols > matSrc.cols && matDst.rows < matSrc.rows))
		return results;
	if (matDst.size().area() > matSrc.size().area())
		return results;
	if (!templData.bIsPatternLearned)
		return results;

	int iTopLayer = GetTopLayer(&matDst, (int)sqrt((double)config.iMinReduceArea));

	vector<Mat> vecMatSrcPyr;
	if (config.bBitwiseNot)
	{
		Mat matNewSrc = 255 - matSrc;
		buildPyramid(matNewSrc, vecMatSrcPyr, iTopLayer);
	}
	else
		buildPyramid(matSrc, vecMatSrcPyr, iTopLayer);

	s_TemplData* pTemplData = &templData;

	double dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iTopLayer].cols,
		pTemplData->vecPyramid[iTopLayer].rows)) * R2D;

	vector<double> vecAngles;
	if (config.dToleranceAngle < VISION_TOLERANCE)
		vecAngles.push_back(0.0);
	else
	{
		for (double dAngle = 0; dAngle < config.dToleranceAngle + dAngleStep; dAngle += dAngleStep)
			vecAngles.push_back(dAngle);
		for (double dAngle = -dAngleStep; dAngle > -config.dToleranceAngle - dAngleStep; dAngle -= dAngleStep)
			vecAngles.push_back(dAngle);
	}

	int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
	Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

	int iSize = (int)vecAngles.size();
	vector<s_MatchParameter> vecMatchParameter;

	vector<double> vecLayerScore(iTopLayer + 1, config.dScore);
	for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
		vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

	Size sizePat = pTemplData->vecPyramid[iTopLayer].size();
	bool bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size().area() / sizePat.area() > 500) && config.iMaxPos > 10;

	for (int i = 0; i < iSize; i++)
	{
		Mat matRotatedSrc, matR = getRotationMatrix2D(ptCenter, vecAngles[i], 1);
		Mat matResult;
		Point ptMaxLoc;
		double dValue, dMaxVal;

		Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(),
			pTemplData->vecPyramid[iTopLayer].size(), vecAngles[i]);

		float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
		float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
		matR.at<double>(0, 2) += fTranslationX;
		matR.at<double>(1, 2) += fTranslationY;
		warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest,
			INTER_LINEAR, BORDER_CONSTANT, Scalar(pTemplData->iBorderColor));

		MatchTemplate(matRotatedSrc, pTemplData, matResult, iTopLayer, false);

		if (bCalMaxByBlock)
		{
			s_BlockMax blockMax(matResult, pTemplData->vecPyramid[iTopLayer].size());
			blockMax.GetMaxValueLoc(dMaxVal, ptMaxLoc);
			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;
			vecMatchParameter.push_back(s_MatchParameter(
				Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
			for (int j = 0; j < config.iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc,
					pTemplData->vecPyramid[iTopLayer].size(), dValue, config.dMaxOverlap, blockMax);
				if (dValue < vecLayerScore[iTopLayer])
					break;
				vecMatchParameter.push_back(s_MatchParameter(
					Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
			}
		}
		else
		{
			minMaxLoc(matResult, 0, &dMaxVal, 0, &ptMaxLoc);
			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;
			vecMatchParameter.push_back(s_MatchParameter(
				Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
			for (int j = 0; j < config.iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc,
					pTemplData->vecPyramid[iTopLayer].size(), dValue, config.dMaxOverlap);
				if (dValue < vecLayerScore[iTopLayer])
					break;
				vecMatchParameter.push_back(s_MatchParameter(
					Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
			}
		}
	}
	sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);

	int iDstW = pTemplData->vecPyramid[iTopLayer].cols, iDstH = pTemplData->vecPyramid[iTopLayer].rows;

	int iStopLayer = 0;
	vector<s_MatchParameter> vecAllResult;

	for (int i = 0; i < (int)vecMatchParameter.size(); i++)
	{
		double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
		Point2f ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);

		double dAngleStepLocal = atan(2.0 / max(iDstW, iDstH)) * R2D;
		vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStepLocal;
		vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStepLocal;

		if (iTopLayer <= iStopLayer)
		{
			vecMatchParameter[i].pt = Point2d(ptLT * ((iTopLayer == 0) ? 1 : 2));
			vecAllResult.push_back(vecMatchParameter[i]);
		}
		else
		{
			for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
			{
				dAngleStepLocal = atan(2.0 / max(pTemplData->vecPyramid[iLayer].cols,
					pTemplData->vecPyramid[iLayer].rows)) * R2D;
				vector<double> vecAnglesLocal;
				double dMatchedAngle = vecMatchParameter[i].dMatchAngle;

				if (config.dToleranceAngle < VISION_TOLERANCE)
					vecAnglesLocal.push_back(0.0);
				else
					for (int k = -1; k <= 1; k++)
						vecAnglesLocal.push_back(dMatchedAngle + dAngleStepLocal * k);

				Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f,
					(vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
				int iLocalSize = (int)vecAnglesLocal.size();
				vector<s_MatchParameter> vecNewMatchParameter(iLocalSize);
				int iMaxScoreIndex = 0;
				double dBigValue = -1;

				for (int j = 0; j < iLocalSize; j++)
				{
					Mat matResult, matRotatedSrc;
					double dMaxValue = 0;
					Point ptMaxLoc;
					GetRotatedROI(vecMatSrcPyr[iLayer], pTemplData->vecPyramid[iLayer].size(),
						ptLT * 2, vecAnglesLocal[j], matRotatedSrc);

					MatchTemplate(matRotatedSrc, pTemplData, matResult, iLayer, config.bUseSIMD);
					minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
					vecNewMatchParameter[j] = s_MatchParameter(ptMaxLoc, dMaxValue, vecAnglesLocal[j]);

					if (vecNewMatchParameter[j].dMatchScore > dBigValue)
					{
						iMaxScoreIndex = j;
						dBigValue = vecNewMatchParameter[j].dMatchScore;
					}
					// Subpixel data collection
					if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 ||
						ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
						vecNewMatchParameter[j].bPosOnBorder = true;
					if (!vecNewMatchParameter[j].bPosOnBorder)
					{
						for (int y = -1; y <= 1; y++)
							for (int x = -1; x <= 1; x++)
								vecNewMatchParameter[j].vecResult[x + 1][y + 1] =
									matResult.at<float>(ptMaxLoc + Point(x, y));
					}
				}
			if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
				break;

				// Subpixel estimation at layer 0
				if (config.bSubPixel && iLayer == 0 &&
					!vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder &&
					iMaxScoreIndex != 0 && iMaxScoreIndex != 2)
				{
					double dNewX = 0, dNewY = 0, dNewAngle = 0;
					SubPixEstimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle,
						dAngleStepLocal, iMaxScoreIndex);
					vecNewMatchParameter[iMaxScoreIndex].pt = Point2d(dNewX, dNewY);
					vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
				}

				double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;
				Point2f ptPaddingLT = ptRotatePt2f(ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f(3, 3);
				Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x,
					vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
				pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);

				if (iLayer == iStopLayer)
				{
					vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
					vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
				}
				else
				{
					vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
					vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStepLocal / 2;
					vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStepLocal / 2;
					ptLT = pt;
				}
			}
		}
	}
	FilterWithScore(&vecAllResult, config.dScore);

	// NMS with rotated rects
	iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
	iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

	for (int i = 0; i < (int)vecAllResult.size(); i++)
	{
		Point2f ptLT, ptRT, ptRB, ptLB;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
		ptLT = vecAllResult[i].pt;
		ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
		ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
		ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
		vecAllResult[i].rectR = RotatedRect(ptLT, ptRT, ptRB);
	}
	FilterWithRotatedRect(&vecAllResult, cv::TM_CCOEFF_NORMED, config.dMaxOverlap);

	sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);

	// Build output
	int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;
	int iMatchSize = (int)vecAllResult.size();

	for (int i = 0; i < iMatchSize; i++)
	{
		s_SingleTargetMatch sstm;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

		sstm.ptLT = vecAllResult[i].pt;
		sstm.ptRT = Point2d(sstm.ptLT.x + iW * cos(dRAngle), sstm.ptLT.y - iW * sin(dRAngle));
		sstm.ptLB = Point2d(sstm.ptLT.x + iH * sin(dRAngle), sstm.ptLT.y + iH * cos(dRAngle));
		sstm.ptRB = Point2d(sstm.ptRT.x + iH * sin(dRAngle), sstm.ptRT.y + iH * cos(dRAngle));
		sstm.ptCenter = Point2d(
			(sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4,
			(sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
		sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
		sstm.dMatchScore = vecAllResult[i].dMatchScore;

		if (sstm.dMatchedAngle < -180) sstm.dMatchedAngle += 360;
		if (sstm.dMatchedAngle > 180) sstm.dMatchedAngle -= 360;

		results.push_back(sstm);
		if ((int)results.size() == config.iMaxPos)
			break;
	}

	return results;
}

void CMatchCore::MatchTemplate(Mat& matSrc, s_TemplData* pTemplData, Mat& matResult, int iLayer, bool bUseSIMD)
{
#if HAS_NEON
	if (bUseSIMD)
	{
		matResult.create(matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
			matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
		matResult.setTo(0);
		Mat& matTemplate = pTemplData->vecPyramid[iLayer];

		int t_r_end = matTemplate.rows;
		for (int r = 0; r < matResult.rows; r++)
		{
			float* r_matResult = matResult.ptr<float>(r);
			uchar* r_source = matSrc.ptr<uchar>(r);
			uchar *r_template, *r_sub_source;
			for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
			{
				r_template = matTemplate.ptr<uchar>();
				r_sub_source = r_source;
				for (int t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols)
				{
					*r_matResult = *r_matResult + IM_Conv_SIMD(r_template, r_sub_source, matTemplate.cols);
				}
			}
		}
	}
	else
#endif
	{
		(void)bUseSIMD;
		matchTemplate(matSrc, pTemplData->vecPyramid[iLayer], matResult, cv::TM_CCORR);
	}

	CCOEFF_Denominator(matSrc, pTemplData, matResult, iLayer);
}

void CMatchCore::CCOEFF_Denominator(Mat& matSrc, s_TemplData* pTemplData, Mat& matResult, int iLayer)
{
	if (pTemplData->vecResultEqual1[iLayer])
	{
		matResult = Scalar::all(1);
		return;
	}
	double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;

	Mat sum, sqsum;
	integral(matSrc, sum, sqsum, CV_64F);

	q0 = (double*)sqsum.data;
	q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
	q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
	q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

	double* p0 = (double*)sum.data;
	double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
	double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows * sum.step);
	double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

	int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
	int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

	double dTemplMean0 = pTemplData->vecTemplMean[iLayer][0];
	double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
	double dInvArea = pTemplData->vecInvArea[iLayer];

	for (int i = 0; i < matResult.rows; i++)
	{
		float* rrow = matResult.ptr<float>(i);
		int idx = i * sumstep;
		int idx2 = i * sqstep;

		for (int j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1)
		{
			double num = rrow[j], t;
			double wndMean2 = 0, wndSum2 = 0;

			t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
			wndMean2 += t * t;
			num -= t * dTemplMean0;
			wndMean2 *= dInvArea;

			t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
			wndSum2 += t;

			double diff2 = MAX(wndSum2 - wndMean2, 0);
			if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * wndSum2))
				t = 0;
			else
				t = std::sqrt(diff2) * dTemplNorm;

			if (fabs(num) < t)
				num /= t;
			else if (fabs(num) < t * 1.125)
				num = num > 0 ? 1 : -1;
			else
				num = 0;

			rrow[j] = (float)num;
		}
	}
}

void CMatchCore::GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI)
{
	double dAngle_radian = dAngle * D2R;
	Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
	Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
	Size sizePadding(size.width + 6, size.height + 6);

	Mat rMat = getRotationMatrix2D(ptC, dAngle, 1);
	rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
	rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;
	warpAffine(matSrc, matROI, rMat, sizePadding);
}

Size CMatchCore::GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle)
{
	double dRAngle_radian = dRAngle * D2R;
	Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1), ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
	Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
	Point2f ptLT_R = ptRotatePt2f(Point2f(ptLT), ptCenter, dRAngle_radian);
	Point2f ptLB_R = ptRotatePt2f(Point2f(ptLB), ptCenter, dRAngle_radian);
	Point2f ptRB_R = ptRotatePt2f(Point2f(ptRB), ptCenter, dRAngle_radian);
	Point2f ptRT_R = ptRotatePt2f(Point2f(ptRT), ptCenter, dRAngle_radian);

	float fTopY = max(max(ptLT_R.y, ptLB_R.y), max(ptRB_R.y, ptRT_R.y));
	float fBottomY = min(min(ptLT_R.y, ptLB_R.y), min(ptRB_R.y, ptRT_R.y));
	float fRightX = max(max(ptLT_R.x, ptLB_R.x), max(ptRB_R.x, ptRT_R.x));
	float fLeftX = min(min(ptLT_R.x, ptLB_R.x), min(ptRB_R.x, ptRT_R.x));

	if (dRAngle > 360) dRAngle -= 360;
	else if (dRAngle < 0) dRAngle += 360;

	if (fabs(fabs(dRAngle) - 90) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 270) < VISION_TOLERANCE)
		return Size(sizeSrc.height, sizeSrc.width);
	else if (fabs(dRAngle) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 180) < VISION_TOLERANCE)
		return sizeSrc;

	double dAngle = dRAngle;
	if (dAngle > 0 && dAngle < 90) { ; }
	else if (dAngle > 90 && dAngle < 180) dAngle -= 90;
	else if (dAngle > 180 && dAngle < 270) dAngle -= 180;
	else if (dAngle > 270 && dAngle < 360) dAngle -= 270;

	float fH1 = (float)(sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R));
	float fH2 = (float)(sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R));

	int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
	int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

	Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

	bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
		|| (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height)
		|| sizeDst.area() > sizeRet.area();
	if (bWrongSize)
		sizeRet = Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));

	return sizeRet;
}

Point2f CMatchCore::ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle)
{
	double dHeight = ptOrg.y * 2;
	double dY1 = dHeight - ptInput.y;
	double dY2 = dHeight - ptOrg.y;

	double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
	double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

	dY = -dY + dHeight;
	return Point2f((float)dX, (float)dY);
}

void CMatchCore::FilterWithScore(vector<s_MatchParameter>* vec, double dScore)
{
	sort(vec->begin(), vec->end(), compareScoreBig2Small);
	int iSize = (int)vec->size(), iIndexDelete = iSize + 1;
	for (int i = 0; i < iSize; i++)
	{
		if ((*vec)[i].dMatchScore < dScore)
		{
			iIndexDelete = i;
			break;
		}
	}
	if (iIndexDelete == iSize + 1)
		return;
	vec->erase(vec->begin() + iIndexDelete, vec->end());
}

void CMatchCore::FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap)
{
	int iMatchSize = (int)vec->size();
	for (int i = 0; i < iMatchSize - 1; i++)
	{
		if (vec->at(i).bDelete)
			continue;
		for (int j = i + 1; j < iMatchSize; j++)
		{
			if (vec->at(j).bDelete)
				continue;
			RotatedRect rect1 = vec->at(i).rectR;
			RotatedRect rect2 = vec->at(j).rectR;
			vector<Point2f> vecInterSec;
			int iInterSecType = rotatedRectangleIntersection(rect1, rect2, vecInterSec);
			if (iInterSecType == INTERSECT_NONE)
				continue;
			else if (iInterSecType == INTERSECT_FULL)
			{
				int iDeleteIndex;
				if (iMethod == cv::TM_SQDIFF)
					iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
				else
					iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
				vec->at(iDeleteIndex).bDelete = true;
			}
			else
			{
				if (vecInterSec.size() < 3)
					continue;
				SortPtWithCenter(vecInterSec);
				double dArea = contourArea(vecInterSec);
				double dRatio = dArea / rect1.size.area();
				if (dRatio > dMaxOverLap)
				{
					int iDeleteIndex;
					if (iMethod == cv::TM_SQDIFF)
						iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
					else
						iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
					vec->at(iDeleteIndex).bDelete = true;
				}
			}
		}
	}
	vector<s_MatchParameter>::iterator it;
	for (it = vec->begin(); it != vec->end();)
	{
		if ((*it).bDelete)
			it = vec->erase(it);
		else
			++it;
	}
}

Point CMatchCore::GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate,
	double& dMaxValue, double dMaxOverlap)
{
	int iStartX = (int)(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
	int iStartY = (int)(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
	rectangle(matResult, Rect(iStartX, iStartY,
		(int)(2 * sizeTemplate.width * (1 - dMaxOverlap)),
		(int)(2 * sizeTemplate.height * (1 - dMaxOverlap))),
		Scalar(-1), cv::FILLED);
	Point ptNewMaxLoc;
	minMaxLoc(matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	return ptNewMaxLoc;
}

Point CMatchCore::GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate,
	double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax)
{
	int iStartX = int(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
	int iStartY = int(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
	Rect rectIgnore(iStartX, iStartY,
		int(2 * sizeTemplate.width * (1 - dMaxOverlap)),
		int(2 * sizeTemplate.height * (1 - dMaxOverlap)));
	rectangle(matResult, rectIgnore, Scalar(-1), cv::FILLED);
	blockMax.UpdateMax(rectIgnore);
	Point ptReturn;
	blockMax.GetMaxValueLoc(dMaxValue, ptReturn);
	return ptReturn;
}

void CMatchCore::SortPtWithCenter(vector<Point2f>& vecSort)
{
	int iSize = (int)vecSort.size();
	Point2f ptCenter;
	for (int i = 0; i < iSize; i++)
		ptCenter += vecSort[i];
	ptCenter /= iSize;

	vector<pair<Point2f, double>> vecPtAngle(iSize);
	for (int i = 0; i < iSize; i++)
	{
		vecPtAngle[i].first = vecSort[i];
		Point2f vec1(vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
		float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
		float fDot = vec1.x;

		if (vec1.y < 0)
			vecPtAngle[i].second = acos(fDot / fNormVec1) * R2D;
		else if (vec1.y > 0)
			vecPtAngle[i].second = 360 - acos(fDot / fNormVec1) * R2D;
		else
		{
			if (vec1.x - ptCenter.x > 0)
				vecPtAngle[i].second = 0;
			else
				vecPtAngle[i].second = 180;
		}
	}
	sort(vecPtAngle.begin(), vecPtAngle.end(), comparePtWithAngle);
	for (int i = 0; i < iSize; i++)
		vecSort[i] = vecPtAngle[i].first;
}

bool CMatchCore::SubPixEstimation(vector<s_MatchParameter>* vec, double* dNewX, double* dNewY,
	double* dNewAngle, double dAngleStep, int iMaxScoreIndex)
{
	Mat matA(27, 10, CV_64F);
	Mat matZ(10, 1, CV_64F);
	Mat matS(27, 1, CV_64F);

	double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
	double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
	double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
	int iRow = 0;

	for (int theta = 0; theta <= 2; theta++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				double dX = dX_maxScore + x;
				double dY = dY_maxScore + y;
				double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
				matA.at<double>(iRow, 0) = dX * dX;
				matA.at<double>(iRow, 1) = dY * dY;
				matA.at<double>(iRow, 2) = dT * dT;
				matA.at<double>(iRow, 3) = dX * dY;
				matA.at<double>(iRow, 4) = dX * dT;
				matA.at<double>(iRow, 5) = dY * dT;
				matA.at<double>(iRow, 6) = dX;
				matA.at<double>(iRow, 7) = dY;
				matA.at<double>(iRow, 8) = dT;
				matA.at<double>(iRow, 9) = 1.0;
				matS.at<double>(iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
				iRow++;
			}
		}
	}

	matZ = (matA.t() * matA).inv() * matA.t() * matS;
	Mat matZ_t;
	transpose(matZ, matZ_t);
	double* dZ = matZ_t.ptr<double>(0);
	Mat matK1 = (Mat_<double>(3, 3) <<
		(2 * dZ[0]), dZ[3], dZ[4],
		dZ[3], (2 * dZ[1]), dZ[5],
		dZ[4], dZ[5], (2 * dZ[2]));
	Mat matK2 = (Mat_<double>(3, 1) << -dZ[6], -dZ[7], -dZ[8]);
	Mat matDelta = matK1.inv() * matK2;

	*dNewX = matDelta.at<double>(0, 0);
	*dNewY = matDelta.at<double>(1, 0);
	*dNewAngle = matDelta.at<double>(2, 0) * R2D;
	return true;
}
