#include "mainwindow.h"

#include <QApplication>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QHeaderView>
#include <QPainter>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QScrollBar>
#include <cmath>

// ============ Helper: cv::Mat → QImage ============

static QImage matToQImage(const cv::Mat& mat)
{
	if (mat.empty()) return QImage();
	if (mat.type() == CV_8UC1)
	{
		QImage img(mat.data, mat.cols, mat.rows, (int)mat.step, QImage::Format_Grayscale8);
		return img.copy();
	}
	else if (mat.type() == CV_8UC3)
	{
		cv::Mat rgb;
		cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
		QImage img(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
		return img.copy();
	}
	return QImage();
}

// ============ ImageView ============

ImageView::ImageView(QWidget* parent) : QLabel(parent)
{
	setMouseTracking(true);
	setAlignment(Qt::AlignTop | Qt::AlignLeft);
}

void ImageView::setImage(const cv::Mat& mat, const QSize& viewSize)
{
	m_matOriginal = mat.clone();
	m_iScaleTimes = 0;
	m_vecResults.clear();

	// Auto-fit: same logic as MFC original
	// scale = min(viewWidth / imgWidth, viewHeight / imgHeight)
	if (!viewSize.isEmpty() && !mat.empty())
	{
		double dScaleX = viewSize.width() / (double)mat.cols;
		double dScaleY = viewSize.height() / (double)mat.rows;
		m_dBaseScale = std::min(dScaleX, dScaleY);
	}
	else
	{
		m_dBaseScale = 1.0;
	}
	m_dScale = m_dBaseScale;
	refresh();
}

void ImageView::setResults(const std::vector<s_SingleTargetMatch>& results, const cv::Size& tmplSize)
{
	m_vecResults = results;
	m_tmplSize = tmplSize;
	refresh();
}

void ImageView::clearResults()
{
	m_vecResults.clear();
	refresh();
}

void ImageView::wheelEvent(QWheelEvent* event)
{
	if (m_matOriginal.empty()) return;

	if (event->angleDelta().y() > 0 && m_iScaleTimes < MAX_SCALE_TIMES)
		m_iScaleTimes++;
	else if (event->angleDelta().y() < 0 && m_iScaleTimes > MIN_SCALE_TIMES)
		m_iScaleTimes--;

	m_dScale = m_dBaseScale * std::pow(SCALE_RATIO, m_iScaleTimes);
	refresh();
	event->accept();
}

void ImageView::mouseMoveEvent(QMouseEvent* event)
{
	if (m_matOriginal.empty()) return;
	QPoint pos = event->pos();
	int imgX = (int)(pos.x() / m_dScale);
	int imgY = (int)(pos.y() / m_dScale);
	if (imgX >= 0 && imgX < m_matOriginal.cols && imgY >= 0 && imgY < m_matOriginal.rows)
		emit mouseMoved(imgX, imgY);
}

void ImageView::refresh()
{
	if (m_matOriginal.empty()) { clear(); return; }

	int w = (int)(m_matOriginal.cols * m_dScale);
	int h = (int)(m_matOriginal.rows * m_dScale);

	// MFC approach: resize first, then draw overlays with scaled coordinates
	cv::Mat matColor;
	if (m_matOriginal.channels() == 1)
		cv::cvtColor(m_matOriginal, matColor, cv::COLOR_GRAY2BGR);
	else
		matColor = m_matOriginal.clone();

	cv::Mat matResize;
	cv::resize(matColor, matResize, cv::Size(w, h));

	// Draw results overlay on resized image (same as MFC RefreshSrcView)
	for (int i = 0; i < (int)m_vecResults.size(); i++)
	{
		const s_SingleTargetMatch& r = m_vecResults[i];
		// Scale coordinates to display size (same as MFC)
		cv::Point ptLT((int)(m_dScale * r.ptLT.x), (int)(m_dScale * r.ptLT.y));
		cv::Point ptRT((int)(m_dScale * r.ptRT.x), (int)(m_dScale * r.ptRT.y));
		cv::Point ptRB((int)(m_dScale * r.ptRB.x), (int)(m_dScale * r.ptRB.y));
		cv::Point ptLB((int)(m_dScale * r.ptLB.x), (int)(m_dScale * r.ptLB.y));
		cv::Point ptC((int)(m_dScale * r.ptCenter.x), (int)(m_dScale * r.ptCenter.y));

		// Dashed red-white lines
		auto drawDash = [&](cv::Point p1, cv::Point p2) {
			cv::LineIterator it(matResize, p1, p2, 8, false);
			for (int j = 0; j < it.count; j++, it++)
			{
				if (j % 3 == 0)
					{ (*it)[0] = 255; (*it)[1] = 255; (*it)[2] = 255; }
				else
					{ (*it)[0] = 0; (*it)[1] = 0; (*it)[2] = 255; }
			}
		};
		drawDash(ptLT, ptLB);
		drawDash(ptLB, ptRB);
		drawDash(ptRB, ptRT);
		drawDash(ptRT, ptLT);

		// Green corner L-marks
		cv::Point ptDis1, ptDis2;
		if (m_tmplSize.width > m_tmplSize.height)
		{
			ptDis1 = (ptLB - ptLT) / 3;
			ptDis2 = (ptRT - ptLT) / 3 * m_tmplSize.height / m_tmplSize.width;
		}
		else
		{
			ptDis1 = (ptLB - ptLT) / 3 * m_tmplSize.width / m_tmplSize.height;
			ptDis2 = (ptRT - ptLT) / 3;
		}
		cv::Scalar green(0, 255, 0);
		cv::line(matResize, ptLT, ptLT + ptDis1 / 2, green, 1, cv::LINE_AA);
		cv::line(matResize, ptLT, ptLT + ptDis2 / 2, green, 1, cv::LINE_AA);
		cv::line(matResize, ptRT, ptRT + ptDis1 / 2, green, 1, cv::LINE_AA);
		cv::line(matResize, ptRT, ptRT - ptDis2 / 2, green, 1, cv::LINE_AA);
		cv::line(matResize, ptRB, ptRB - ptDis1 / 2, green, 1, cv::LINE_AA);
		cv::line(matResize, ptRB, ptRB - ptDis2 / 2, green, 1, cv::LINE_AA);
		cv::line(matResize, ptLB, ptLB - ptDis1 / 2, green, 1, cv::LINE_AA);
		cv::line(matResize, ptLB, ptLB + ptDis2 / 2, green, 1, cv::LINE_AA);

		// Diagonal L mark at top-left
		drawDash(ptLT + ptDis1, ptLT + ptDis2);

		// Center cross (with anti-aliasing, same as MFC)
		cv::line(matResize, ptC - cv::Point(5, 0), ptC + cv::Point(5, 0), green, 1, cv::LINE_AA);
		cv::line(matResize, ptC - cv::Point(0, 5), ptC + cv::Point(0, 5), green, 1, cv::LINE_AA);

		// Index text
		std::string str = std::to_string(i);
		cv::putText(matResize, str, (ptLT + ptRT) / 2, cv::FONT_HERSHEY_PLAIN, 1, green);
	}

	QImage img = matToQImage(matResize);
	setPixmap(QPixmap::fromImage(img));
	setFixedSize(w, h);
}

// ============ MainWindow ============

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
	// Init language strings
	m_lang[0] = { // 繁體中文
		"目標數量:", "最大重疊比例:", "分數(相似度):", "容許角度:", "最小縮減面積:",
		"執行", "語言:", "參數設定",
		"索引", "分數", "角度(度)", "位置X", "位置Y",
		"執行時間", "來源影像大小", "模板影像大小", "像素位置",
		"數量:", "次像素", "讀取來源影像", "讀取模板影像", "檔案"
	};
	m_lang[1] = { // 簡體中文
		"目标数量:", "最大重叠比例:", "分数(相似度):", "容许角度:", "最小缩减面积:",
		"执行", "语言:", "参数设定",
		"索引", "分数", "角度(度)", "位置X", "位置Y",
		"执行时间", "源图像大小", "模板图像大小", "像素位置",
		"数量:", "亚像素", "读取源图像", "读取模板图像", "文件"
	};
	m_lang[2] = { // English
		"Target Number:", "Max OverLap Ratio:", "Score (Similarity):", "Tolerance Angle:", "Min Reduced Area:",
		"Execute", "Language:", "Parameter Setting",
		"Index", "Score", "Angle (deg)", "PosX", "PosY",
		"Execution Time", "Source Image Size", "Dst Image Size", "Pixel Pos",
		"Num:", "SubPixel", "Read Source Image", "Read Dst Image", "File"
	};

	setupUI();
	setAcceptDrops(true);
	setWindowTitle("Image Match Tool");
	resize(1100, 700);
}

void MainWindow::setupUI()
{
	// Central widget
	QWidget* central = new QWidget(this);
	setCentralWidget(central);

	QHBoxLayout* mainLayout = new QHBoxLayout(central);

	// === Left: Source image view ===
	m_srcView = new ImageView();
	m_srcScrollArea = new QScrollArea();
	m_srcScrollArea->setWidget(m_srcView);
	m_srcScrollArea->setWidgetResizable(false);
	m_srcScrollArea->setMinimumSize(600, 500);
	m_srcScrollArea->setStyleSheet("background: #f0f0f0;");
	mainLayout->addWidget(m_srcScrollArea, 3);

	// === Right panel ===
	QVBoxLayout* rightLayout = new QVBoxLayout();
	mainLayout->addLayout(rightLayout, 1);

	// Template view
	m_dstView = new QLabel();
	m_dstView->setFixedSize(160, 120);
	m_dstView->setFrameStyle(QFrame::Box);
	m_dstView->setAlignment(Qt::AlignCenter);
	m_dstView->setStyleSheet("background: #e0e0e0;");
	m_dstView->setText("Template");
	rightLayout->addWidget(m_dstView);

	// Execute button
	m_btnExecute = new QPushButton("Execute");
	m_btnExecute->setFixedHeight(30);
	connect(m_btnExecute, &QPushButton::clicked, this, &MainWindow::onExecute);
	rightLayout->addWidget(m_btnExecute);

	// Parameter group
	m_grpParams = new QGroupBox("Parameter Setting");
	QFormLayout* paramForm = new QFormLayout(m_grpParams);
	paramForm->setSpacing(4);

	m_spinTargetNum = new QSpinBox(); m_spinTargetNum->setRange(1, 200); m_spinTargetNum->setValue(70);
	m_spinOverlap = new QDoubleSpinBox(); m_spinOverlap->setRange(0, 0.8); m_spinOverlap->setSingleStep(0.1); m_spinOverlap->setValue(0);
	m_spinScore = new QDoubleSpinBox(); m_spinScore->setRange(0, 1); m_spinScore->setSingleStep(0.1); m_spinScore->setValue(0.5);
	m_spinAngle = new QDoubleSpinBox(); m_spinAngle->setRange(0, 180); m_spinAngle->setSingleStep(1); m_spinAngle->setValue(0);
	m_spinMinArea = new QSpinBox(); m_spinMinArea->setRange(64, 2048); m_spinMinArea->setValue(256);

	m_lblTargetNum = new QLabel("Target Number:");
	m_lblOverlap = new QLabel("Max OverLap Ratio:");
	m_lblScore = new QLabel("Score (Similarity):");
	m_lblAngle = new QLabel("Tolerance Angle:");
	m_lblMinArea = new QLabel("Min Reduced Area:");

	paramForm->addRow(m_lblTargetNum, m_spinTargetNum);
	paramForm->addRow(m_lblOverlap, m_spinOverlap);
	paramForm->addRow(m_lblScore, m_spinScore);
	paramForm->addRow(m_lblAngle, m_spinAngle);
	paramForm->addRow(m_lblMinArea, m_spinMinArea);

	rightLayout->addWidget(m_grpParams);

	// Options row
	QHBoxLayout* optLayout = new QHBoxLayout();
	m_lblLanguage = new QLabel("Language:");
	m_cmbLanguage = new QComboBox();
	m_cmbLanguage->addItems({"Chinese (Traditional)", "Chinese (Simplified)", "English"});
	m_cmbLanguage->setCurrentIndex(2);
	connect(m_cmbLanguage, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onLanguageChanged);
	optLayout->addWidget(m_lblLanguage);
	optLayout->addWidget(m_cmbLanguage);
	rightLayout->addLayout(optLayout);

	QHBoxLayout* chkLayout = new QHBoxLayout();
	m_chkSIMD = new QCheckBox("SIMD"); m_chkSIMD->setChecked(true);
	m_chkSubPixel = new QCheckBox("SubPixel");
	chkLayout->addWidget(m_chkSIMD);
	chkLayout->addWidget(m_chkSubPixel);
	rightLayout->addLayout(chkLayout);

	// Num label
	QHBoxLayout* numLayout = new QHBoxLayout();
	m_lblNumLabel = new QLabel("Num:");
	m_lblNum = new QLabel("0");
	numLayout->addWidget(m_lblNumLabel);
	numLayout->addWidget(m_lblNum);
	numLayout->addStretch();
	rightLayout->addLayout(numLayout);

	// Results table
	m_tableResults = new QTableWidget(0, 5);
	m_tableResults->setHorizontalHeaderLabels({"Index", "Score", "Angle (deg)", "PosX", "PosY"});
	m_tableResults->horizontalHeader()->setStretchLastSection(true);
	m_tableResults->setSelectionBehavior(QAbstractItemView::SelectRows);
	m_tableResults->setEditTriggers(QAbstractItemView::NoEditTriggers);
	m_tableResults->verticalHeader()->setVisible(false);
	rightLayout->addWidget(m_tableResults, 1);

	// Menu bar
	QMenu* fileMenu = menuBar()->addMenu("File");
	QAction* actLoadSrc = fileMenu->addAction("Read Source Image");
	QAction* actLoadDst = fileMenu->addAction("Read Dst Image");
	connect(actLoadSrc, &QAction::triggered, this, &MainWindow::onLoadSrc);
	connect(actLoadDst, &QAction::triggered, this, &MainWindow::onLoadDst);

	// Status bar
	m_lblExecTime = new QLabel("Execution Time");
	m_lblSrcSize = new QLabel("Source Image Size");
	m_lblDstSize = new QLabel("Dst Image Size");
	m_lblPixelPos = new QLabel("Pixel Pos");
	statusBar()->addWidget(m_lblExecTime, 1);
	statusBar()->addWidget(m_lblSrcSize, 1);
	statusBar()->addWidget(m_lblDstSize, 1);
	statusBar()->addWidget(m_lblPixelPos, 1);

	// Mouse tracking from image view
	connect(m_srcView, &ImageView::mouseMoved, this, &MainWindow::onMouseMoved);
}

void MainWindow::onLoadSrc()
{
	QString path = QFileDialog::getOpenFileName(this, "Load Source Image", "",
		"Images (*.bmp *.jpg *.jpeg *.png);;All Files (*)");
	if (!path.isEmpty())
		loadSrcImage(path);
}

void MainWindow::onLoadDst()
{
	QString path = QFileDialog::getOpenFileName(this, "Load Template Image", "",
		"Images (*.bmp *.jpg *.jpeg *.png);;All Files (*)");
	if (!path.isEmpty())
		loadDstImage(path);
}

void MainWindow::loadSrcImage(const QString& path)
{
	m_matSrc = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE);
	if (m_matSrc.empty()) return;

	// Pass viewport size for auto-fit (same as MFC LoadSrc logic)
	QSize viewSize = m_srcScrollArea->viewport()->size();
	m_srcView->setImage(m_matSrc, viewSize);
	m_vecResults.clear();
	m_lblSrcSize->setText(QString("Src: %1x%2").arg(m_matSrc.cols).arg(m_matSrc.rows));
}

void MainWindow::loadDstImage(const QString& path)
{
	m_matDst = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE);
	if (m_matDst.empty()) return;

	// Show template in dstView (auto-fit to view area)
	cv::Mat matShow;
	double scale = std::min((double)m_dstView->width() / m_matDst.cols,
	                        (double)m_dstView->height() / m_matDst.rows);
	cv::resize(m_matDst, matShow, cv::Size(), scale, scale);
	QImage img = matToQImage(matShow);
	m_dstView->setPixmap(QPixmap::fromImage(img));

	m_lblDstSize->setText(QString("Tmpl: %1x%2").arg(m_matDst.cols).arg(m_matDst.rows));

	// Auto-learn pattern
	s_MatchConfig cfg;
	cfg.iMinReduceArea = m_spinMinArea->value();
	m_matcher.LearnPattern(m_matDst, m_templData, cfg.iMinReduceArea);
}

void MainWindow::onExecute()
{
	if (m_matSrc.empty() || m_matDst.empty() || !m_templData.bIsPatternLearned)
		return;

	s_MatchConfig config;
	config.iMaxPos = m_spinTargetNum->value();
	config.dMaxOverlap = m_spinOverlap->value();
	config.dScore = m_spinScore->value();
	config.dToleranceAngle = m_spinAngle->value();
	config.iMinReduceArea = m_spinMinArea->value();
	config.bUseSIMD = m_chkSIMD->isChecked();
	config.bSubPixel = m_chkSubPixel->isChecked();

	// Re-learn if MinArea changed
	m_matcher.LearnPattern(m_matDst, m_templData, config.iMinReduceArea);

	QElapsedTimer timer;
	timer.start();
	m_vecResults = m_matcher.Match(m_matSrc, m_matDst, m_templData, config);
	qint64 elapsed = timer.elapsed();

	m_lblExecTime->setText(QString("Time: %1 ms").arg(elapsed));
	m_lblNum->setText(QString::number(m_vecResults.size()));

	// Fill table
	m_tableResults->setRowCount((int)m_vecResults.size());
	for (int i = 0; i < (int)m_vecResults.size(); i++)
	{
		const s_SingleTargetMatch& r = m_vecResults[i];
		m_tableResults->setItem(i, 0, new QTableWidgetItem(QString::number(i)));
		m_tableResults->setItem(i, 1, new QTableWidgetItem(QString::number(r.dMatchScore, 'f', 3)));
		m_tableResults->setItem(i, 2, new QTableWidgetItem(QString::number(r.dMatchedAngle, 'f', 3)));
		m_tableResults->setItem(i, 3, new QTableWidgetItem(QString::number(r.ptCenter.x, 'f', 3)));
		m_tableResults->setItem(i, 4, new QTableWidgetItem(QString::number(r.ptCenter.y, 'f', 3)));
	}

	// Draw results on image
	m_srcView->setResults(m_vecResults, cv::Size(m_matDst.cols, m_matDst.rows));
}

void MainWindow::onLanguageChanged(int index)
{
	m_iLangIndex = index;
	updateLanguage();
}

void MainWindow::updateLanguage()
{
	const LangStrings& L = m_lang[m_iLangIndex];
	m_lblTargetNum->setText(L.targetNum);
	m_lblOverlap->setText(L.overlap);
	m_lblScore->setText(L.score);
	m_lblAngle->setText(L.angle);
	m_lblMinArea->setText(L.minArea);
	m_btnExecute->setText(L.execute);
	m_lblLanguage->setText(L.language);
	m_grpParams->setTitle(L.paramSetting);
	m_lblNumLabel->setText(L.num);
	m_chkSubPixel->setText(L.subPixel);

	m_tableResults->setHorizontalHeaderLabels({L.index, L.scoreCol, L.angleCol, L.posX, L.posY});

	// Update menu
	QList<QAction*> actions = menuBar()->actions();
	if (!actions.isEmpty())
	{
		actions[0]->setText(L.file);
		QMenu* menu = actions[0]->menu();
		if (menu)
		{
			QList<QAction*> items = menu->actions();
			if (items.size() >= 2)
			{
				items[0]->setText(L.loadSrc);
				items[1]->setText(L.loadDst);
			}
		}
	}
}

void MainWindow::onMouseMoved(int imgX, int imgY)
{
	m_lblPixelPos->setText(QString("Pos: %1, %2").arg(imgX).arg(imgY));
}

void MainWindow::dragEnterEvent(QDragEnterEvent* event)
{
	if (event->mimeData()->hasUrls())
		event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* event)
{
	const QList<QUrl> urls = event->mimeData()->urls();
	if (urls.isEmpty()) return;

	QString path = urls.first().toLocalFile();
	if (path.isEmpty()) return;

	// Drop on left side = source, right side = template
	QPoint pos = event->position().toPoint();
	int midX = width() / 2;

	if (pos.x() < midX)
		loadSrcImage(path);
	else
		loadDstImage(path);
}
