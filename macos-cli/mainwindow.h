#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QScrollArea>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QTableWidget>
#include <QStatusBar>
#include <QMenuBar>
#include <QGroupBox>
#include <QElapsedTimer>
#include <QPixmap>
#include <QSize>

#include <opencv2/opencv.hpp>
#include "match_core.h"

class ImageView : public QLabel
{
	Q_OBJECT
public:
	ImageView(QWidget* parent = nullptr);
	void setImage(const cv::Mat& mat, const QSize& viewSize = QSize());
	void setResults(const std::vector<s_SingleTargetMatch>& results, const cv::Size& tmplSize);
	void clearResults();
	double scale() const { return m_dScale; }

signals:
	void mouseMoved(int imgX, int imgY);

protected:
	void wheelEvent(QWheelEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;

private:
	void refresh();
	cv::Mat m_matOriginal;
	std::vector<s_SingleTargetMatch> m_vecResults;
	cv::Size m_tmplSize;
	double m_dScale = 1.0;
	double m_dBaseScale = 1.0;
	int m_iScaleTimes = 0;
	static constexpr int MAX_SCALE_TIMES = 10;
	static constexpr int MIN_SCALE_TIMES = 0;
	static constexpr double SCALE_RATIO = 1.25;
};

class MainWindow : public QMainWindow
{
	Q_OBJECT
public:
	MainWindow(QWidget* parent = nullptr);

protected:
	void dragEnterEvent(QDragEnterEvent* event) override;
	void dropEvent(QDropEvent* event) override;

private slots:
	void onLoadSrc();
	void onLoadDst();
	void onExecute();
	void onLanguageChanged(int index);
	void onMouseMoved(int imgX, int imgY);

private:
	void loadSrcImage(const QString& path);
	void loadDstImage(const QString& path);
	void updateLanguage();
	void setupUI();

	// Image data
	cv::Mat m_matSrc;
	cv::Mat m_matDst;
	s_TemplData m_templData;
	CMatchCore m_matcher;
	std::vector<s_SingleTargetMatch> m_vecResults;

	// Widgets
	QScrollArea* m_srcScrollArea;
	ImageView* m_srcView;
	QLabel* m_dstView;
	QPushButton* m_btnExecute;

	// Parameters
	QSpinBox* m_spinTargetNum;
	QDoubleSpinBox* m_spinOverlap;
	QDoubleSpinBox* m_spinScore;
	QDoubleSpinBox* m_spinAngle;
	QSpinBox* m_spinMinArea;

	// Options
	QCheckBox* m_chkSIMD;
	QCheckBox* m_chkSubPixel;
	QComboBox* m_cmbLanguage;

	// Results
	QTableWidget* m_tableResults;
	QLabel* m_lblNum;

	// Status bar labels
	QLabel* m_lblExecTime;
	QLabel* m_lblSrcSize;
	QLabel* m_lblDstSize;
	QLabel* m_lblPixelPos;

	// Language labels (for dynamic update)
	QLabel* m_lblTargetNum;
	QLabel* m_lblOverlap;
	QLabel* m_lblScore;
	QLabel* m_lblAngle;
	QLabel* m_lblMinArea;
	QLabel* m_lblLanguage;
	QLabel* m_lblNumLabel;
	QGroupBox* m_grpParams;

	// Language data
	struct LangStrings {
		QString targetNum, overlap, score, angle, minArea;
		QString execute, language, paramSetting;
		QString index, scoreCol, angleCol, posX, posY;
		QString execTime, srcSize, dstSize, pixelPos;
		QString num, subPixel, loadSrc, loadDst, file;
	};
	LangStrings m_lang[3]; // 0=繁中, 1=簡中, 2=English
	int m_iLangIndex = 2;  // default English
};
