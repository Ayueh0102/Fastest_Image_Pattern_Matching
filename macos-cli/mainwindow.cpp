#include "mainwindow.h"
#include "template_editor.h"
#include <QCheckBox>
#include <QComboBox>
#include <QColorSpace>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QFileInfo>
#include <QDateTime>
#include <QFormLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QSaveFile>
#include <QSignalBlocker>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QtConcurrent/QtConcurrentRun>
#include <algorithm>
#include <cmath>
#include <utility>

namespace {
enum Engine { Original, ResearchCpu, Metal, Cuda, FullMetal };
fpm::ImageView imageView(const QImage& image)
{
	return {image.constBits(), image.width(), image.height(), std::size_t(image.bytesPerLine())};
}
fpm::ColourImageView rgbView(const QImage& rgb)
{
	return {rgb.constBits(),rgb.width(),rgb.height(),std::size_t(rgb.bytesPerLine())};
}
QImage grayFromImage(QImage rgb)
{
	// Already gray8: equal RGB channels would round back to the same byte.
	// Retain the decoder's stride and stored orientation without an RGB copy.
	if (rgb.format() == QImage::Format_Grayscale8) {
		// Match the display metadata of the old newly allocated gray image.
		rgb.setDevicePixelRatio(1);
		rgb.setColorSpace(QColorSpace{});
		return rgb;
	}
	rgb = std::move(rgb).convertToFormat(QImage::Format_RGB888);
	if (rgb.isNull()) return {};
	QImage gray(rgb.size(), QImage::Format_Grayscale8);
	if(gray.isNull()) return {};
	for (int y = 0; y < rgb.height(); ++y)
		for (int x = 0; x < rgb.width(); ++x) {
			const auto* p = rgb.constScanLine(y) + 3 * x;
			gray.scanLine(y)[x] = fpm::grayFromRgb(p[0], p[1], p[2]);
		}
	return gray;
}
QImage readGray(const QString& path,QImage* original=nullptr)
{
	QImageReader reader(path);
	reader.setAutoTransform(false);
	auto decoded=reader.read();
	if(original) *original=decoded;
	return grayFromImage(std::move(decoded));
}
}
QImage MainWindow::recipeGray(const QImage& image) { return grayFromImage(image); }

ImageView::ImageView(QWidget* parent) : QWidget(parent) { setMouseTracking(true); }
void ImageView::setImage(const QImage& image, const QSize& viewport,bool keepZoom)
{
	const bool preserve=keepZoom && image_.size()==image.size();
	image_ = image; results_.clear(); selected_ = -1; if(image.isNull()) {setFixedSize(1,1);update();} else if(preserve) update(); else fit(viewport);
}
void ImageView::fit(const QSize& viewport)
{
	if (image_.isNull()) return;
	fitScale_ = std::min(double(std::max(1, viewport.width())) / image_.width(), double(std::max(1, viewport.height())) / image_.height());
	scale_ = fitScale_; resizeImage();
}
void ImageView::resizeImage()
{
	setFixedSize(std::max(1, int(std::ceil(image_.width() * scale_))), std::max(1, int(std::ceil(image_.height() * scale_))));
	update();
}
void ImageView::setResults(const std::vector<fpm::Match>& results) { results_ = results; selected_ = -1; update(); }
void ImageView::selectResult(int index) { selected_ = index; update(); }
void ImageView::wheelEvent(QWheelEvent* event)
{
	if (image_.isNull()) return;
	const double delta = event->pixelDelta().isNull() ? event->angleDelta().y() / 120.0 : event->pixelDelta().y() / 40.0;
	scale_ = std::clamp(scale_ * std::pow(1.2, delta), fitScale_ * .25, std::max(8.0, fitScale_));
	resizeImage(); event->accept();
}
void ImageView::mouseMoveEvent(QMouseEvent* event)
{
	const int x = int(event->position().x() / scale_), y = int(event->position().y() / scale_);
	if (x >= 0 && y >= 0 && x < image_.width() && y < image_.height()) emit mouseMoved(x, y);
}
void ImageView::paintEvent(QPaintEvent*)
{
	QPainter painter(this);
	painter.fillRect(rect(), QColor("#20252b"));
	if (image_.isNull()) return;
	painter.drawImage(QRectF(0, 0, image_.width() * scale_, image_.height() * scale_), image_);
	painter.setRenderHint(QPainter::Antialiasing);
	for (std::size_t i = 0; i < results_.size(); ++i) {
		const auto& m = results_[i];
		QPolygonF polygon;
		for (const auto& p : m.corners) polygon << QPointF(p.x * scale_, p.y * scale_);
		painter.setPen(QPen(int(i) == selected_ ? QColor("#ffd166") : QColor("#2ee695"), int(i) == selected_ ? 3 : 1.5));
		painter.setBrush(Qt::NoBrush); painter.drawPolygon(polygon);
		const QPointF center(m.center.x * scale_, m.center.y * scale_);
		painter.drawLine(center - QPointF(5, 0), center + QPointF(5, 0));
		painter.drawLine(center - QPointF(0, 5), center + QPointF(0, 5));
		painter.drawText(polygon[0] + QPointF(4, -4), QString::number(i));
	}
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
	setWindowTitle("Image Pattern Matching — C++"); resize(1180, 820); setAcceptDrops(true);
	auto* central = new QWidget; setCentralWidget(central);
	auto* main = new QHBoxLayout(central);
	auto* split = new QSplitter(Qt::Vertical);
	view_ = new ImageView; view_->setObjectName("sourceView");
	imageScroll_ = new QScrollArea; imageScroll_->setWidget(view_); imageScroll_->setMinimumSize(400, 260);
	imageScroll_->setStyleSheet("QScrollArea { background: #20252b; }");
	split->addWidget(imageScroll_);
	table_ = new QTableWidget(0, 6); table_->setObjectName("results");
	table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	table_->setSelectionBehavior(QAbstractItemView::SelectRows); table_->setSelectionMode(QAbstractItemView::SingleSelection);
	table_->setEditTriggers(QAbstractItemView::NoEditTriggers); table_->verticalHeader()->hide();
	split->addWidget(table_); split->setSizes({560, 180}); main->addWidget(split, 1);
	auto* panel = new QWidget; auto* side = new QVBoxLayout(panel); panel->setMinimumWidth(335);
	auto* panelScroll = new QScrollArea; panelScroll->setWidget(panel); panelScroll->setWidgetResizable(true); panelScroll->setMinimumWidth(360);
	auto* controls = new QVBoxLayout; controls->addWidget(panelScroll, 1);
	panelScroll->setMaximumWidth(420); main->addLayout(controls);
	language_ = new QComboBox; language_->addItems({"繁體中文", "简体中文", "English"}); side->addWidget(language_);
	loadSource_ = new QPushButton; loadTemplate_ = new QPushButton;
	loadSource_->setObjectName("loadSource"); loadTemplate_->setObjectName("loadTemplate");
	buildOperatorPanel(side,controls);
	side->addWidget(loadSource_);
	crop_=new QPushButton;crop_->setObjectName("makeTemplate");side->addWidget(crop_);
	connect(crop_,&QPushButton::clicked,this,&MainWindow::makeTemplate);
#ifdef FPM_GUI_MULTIMEDIA
	liveButton_=new QPushButton;liveButton_->setObjectName("liveInput");side->addWidget(liveButton_);
	liveInfo_=new QLabel;liveInfo_->setObjectName("liveInfo");liveInfo_->setTextFormat(Qt::PlainText);liveInfo_->setWordWrap(true);side->addWidget(liveInfo_);liveInfo_->hide();
	connect(liveButton_,&QPushButton::clicked,this,&MainWindow::chooseLiveInput);
	saveFrame_=new QPushButton;saveFrame_->setObjectName("saveFrame");side->addWidget(saveFrame_);
	connect(saveFrame_,&QPushButton::clicked,this,[this]{
		auto path=QFileDialog::getSaveFileName(this,saveFrame_->text(),{},"PNG (*.png)");
		if(path.isEmpty()) return;
		if(!path.endsWith(".png",Qt::CaseInsensitive)) path+=".png";
		if(!saveSampledFrame(path)) status_->setText(text("無法儲存影格","无法保存帧","Could not save frame"));
	});
	connect(this,&MainWindow::matchingFinished,this,[this](bool ok){if(!ok && live_) stopLive(status_->text());});
#endif
	side->addWidget(batchButton_); sourceInfo_ = new QLabel;sourceInfo_->setWordWrap(true);sourceInfo_->setTextFormat(Qt::PlainText);side->addWidget(sourceInfo_);
	templateInfo_ = new QLabel;templateInfo_->setWordWrap(true);templateInfo_->setTextFormat(Qt::PlainText);side->addWidget(templateInfo_);
	templatePreview_ = new QLabel; templatePreview_->setObjectName("templatePreview"); templatePreview_->setMinimumHeight(120); templatePreview_->setMaximumHeight(150);
	templatePreview_->setAlignment(Qt::AlignCenter); side->addWidget(templatePreview_);
	side->addWidget(editSettings_);
	parameters_ = new QGroupBox; auto* form = new QFormLayout(parameters_);
	form->addRow(loadTemplate_);
	expectedLabel_=new QLabel;expected_=new QSpinBox;expected_->setObjectName("expectedCount");expected_->setRange(0,9999);
	form->addRow(expectedLabel_,expected_);
	engine_ = new QComboBox; engine_->setObjectName("engine"); engine_->addItem("",0);
#ifdef FPM_GUI_RESEARCH
	scene_=new QComboBox;scene_->setObjectName("scenePreset");
	for(int i=0;i<4;++i) scene_->addItem("");
	form->addRow(scene_);
	loadTuning_=new QPushButton;loadTuning_->setObjectName("loadTuning");form->addRow(loadTuning_);
	connect(loadTuning_,&QPushButton::clicked,this,[this]{
		const auto path=QFileDialog::getOpenFileName(this,loadTuning_->text(),{},"JSON (*.json)");
		if(!path.isEmpty()) loadTuning(path);
	});
	engine_->addItem("",1);
#ifdef Q_OS_MACOS
	engine_->addItem("",2);
	engine_->addItem("",FullMetal);
#endif
#ifdef FPM_GUI_CUDA
	engine_->addItem("",Cuda);
#endif
#endif
	engineLabel_ = new QLabel; form->addRow(engineLabel_,engine_);
	searchColour_=new QCheckBox; searchColour_->setObjectName("searchColour"); form->addRow(searchColour_);
	auto* scaleRow = new QHBoxLayout;
	scaleMin_=new QDoubleSpinBox; scaleMax_=new QDoubleSpinBox; scaleStep_=new QDoubleSpinBox;
	QDoubleSpinBox* scales[]={scaleMin_,scaleMax_,scaleStep_};
	const char* names[]={"scaleMin","scaleMax","scaleStep"};
	for(int i=0;i<3;++i) { scales[i]->setObjectName(names[i]);scales[i]->setRange(.01,8);scales[i]->setDecimals(3);scales[i]->setSingleStep(.05);scales[i]->setValue(i==2?.25:1.);scaleRow->addWidget(scales[i]); }
	scaleLabel_=new QLabel;scaleLabel_->setWordWrap(true);form->addRow(scaleLabel_,scaleRow);
	nccMode_=new QComboBox;nccMode_->setObjectName("nccMode");nccMode_->addItem("",0);nccMode_->addItem("",2);nccMode_->addItem("",3);
	nccLabel_=new QLabel;form->addRow(nccLabel_,nccMode_);
	coarseOverlap_=new QCheckBox;coarseOverlap_->setObjectName("coarseOverlap");form->addRow(coarseOverlap_);
	continuous_=new QCheckBox;continuous_->setObjectName("continuousRefinement");continuous_->setChecked(true);form->addRow(continuous_);
#ifdef FPM_GUI_RESEARCH
	shortSide_=new QSpinBox;shortSide_->setObjectName("minTopShortSide");shortSide_->setRange(0,32768);
	shortSideLabel_=new QLabel;form->addRow(shortSideLabel_,shortSide_);
	proposalScore_=new QDoubleSpinBox;proposalScore_->setRange(0,1);proposalScore_->setDecimals(3);proposalScore_->setSingleStep(.05);
	proposalScore_->setValue(.5);proposalScore_->setObjectName("proposalScore");proposalLabel_=new QLabel;form->addRow(proposalLabel_,proposalScore_);
	verifySmall_=new QCheckBox;verifySmall_->setObjectName("verifySmall");form->addRow(verifySmall_);
	dedup_=new QCheckBox;dedup_->setObjectName("dedup");form->addRow(dedup_);
	checkColour_=new QCheckBox;checkColour_->setObjectName("checkColour");checkColour_->setEnabled(false);form->addRow(checkColour_);
	loadMask_=new QPushButton;loadMask_->setEnabled(false);form->addRow(loadMask_);
	colourThresholdLabel_=new QLabel;colourThreshold_=new QDoubleSpinBox;colourThreshold_->setObjectName("colourThreshold");
	colourThreshold_->setDecimals(5);colourThreshold_->setRange(0,1);colourThreshold_->setSingleStep(.001);
	colourThreshold_->setEnabled(false);form->addRow(colourThresholdLabel_,colourThreshold_);
	connect(checkColour_,&QCheckBox::toggled,colourThreshold_,&QWidget::setEnabled);
	connect(loadMask_,&QPushButton::clicked,this,[this]{
		const auto path=QFileDialog::getOpenFileName(this,loadMask_->text(),{},"PNG (*.png)");
		if(!path.isEmpty()) loadInspectionMask(path);
	});
	connect(scene_,&QComboBox::activated,this,&MainWindow::applyScene);
#endif

	targets_ = new QSpinBox; targets_->setRange(1, 10000); targets_->setValue(70); targets_->setObjectName("targetCount");
	score_ = new QDoubleSpinBox; score_->setRange(0, 1); score_->setDecimals(3); score_->setSingleStep(.05); score_->setValue(.5); score_->setObjectName("score");
	overlap_ = new QDoubleSpinBox; overlap_->setRange(0, 1); overlap_->setSingleStep(.1); overlap_->setObjectName("overlap");
	angle_ = new QDoubleSpinBox; angle_->setRange(0, 180); angle_->setSuffix("°"); angle_->setObjectName("angle");
	minArea_ = new QSpinBox; minArea_->setRange(1, 1000000); minArea_->setValue(256); minArea_->setObjectName("minArea");
	QWidget* fields[] = {targets_, score_, overlap_, angle_, minArea_};
	for (int i = 0; i < 5; ++i) { labels_[i] = new QLabel; form->addRow(labels_[i], fields[i]); }
	useRanges_ = new QCheckBox; useRanges_->setObjectName("useRanges"); form->addRow(useRanges_);
	for (int i = 0; i < 2; ++i) {
		auto* row = new QHBoxLayout;
		for (int j = 0; j < 2; ++j) {
			const int k = i * 2 + j;
			ranges_[k] = new QDoubleSpinBox; ranges_[k]->setRange(-180, 180); ranges_[k]->setSuffix("°");
			ranges_[k]->setObjectName(QString("range%1").arg(k)); ranges_[k]->setEnabled(false);
			ranges_[k]->setValue(k < 2 ? 40 + j * 20 : -110 + j * 10);
			row->addWidget(ranges_[k]); if (j == 0) row->addWidget(new QLabel("→"));
		}
		form->addRow(row);
	}
	subpixel_ = new QCheckBox; subpixel_->setObjectName("subpixel");
	invert_ = new QCheckBox; invert_->setObjectName("invert");
	simd_ = new QCheckBox; simd_->setChecked(true); simd_->setObjectName("simd");
	fast_ = new QCheckBox; fast_->setObjectName("fast");
	form->addRow(subpixel_); form->addRow(invert_); form->addRow(simd_); form->addRow(fast_); side->addWidget(parameters_);side->addWidget(saveRecipe_);
	parameters_->hide();saveRecipe_->hide();
	connect(editSettings_,&QPushButton::toggled,parameters_,&QWidget::setVisible);
	connect(editSettings_,&QPushButton::toggled,saveRecipe_,&QWidget::setVisible);
	execute_ = new QPushButton; execute_->setObjectName("execute"); execute_->setMinimumHeight(36); controls->addWidget(execute_);
	auto* resultActions = new QHBoxLayout; controls->addLayout(resultActions);
	fit_ = new QPushButton; resultActions->addWidget(fit_);
	export_ = new QPushButton; export_->setObjectName("export"); export_->setEnabled(false); resultActions->addWidget(export_); side->addStretch();
	status_ = new QLabel; status_->setObjectName("status"); status_->setWordWrap(true); statusBar()->addWidget(status_, 1);
	connect(language_, &QComboBox::currentIndexChanged, this, &MainWindow::updateLanguage);
	connect(loadSource_, &QPushButton::clicked, this, [this] { const auto p = QFileDialog::getOpenFileName(this, loadSource_->text(), {}, "Images (*.bmp *.png *.jpg *.jpeg *.pgm);;All files (*)"); if (!p.isEmpty() && loadSource(p) && !templ_.isNull()) execute(); });
	connect(loadTemplate_, &QPushButton::clicked, this, [this] { const auto p = QFileDialog::getOpenFileName(this, loadTemplate_->text(), {}, "Images (*.bmp *.png *.jpg *.jpeg *.pgm);;All files (*)"); if (!p.isEmpty()) loadTemplate(p); });
	connect(execute_, &QPushButton::clicked, this, &MainWindow::execute);
	connect(&watcher_, &QFutureWatcher<MatchJobResult>::finished, this, &MainWindow::complete);
	connect(fit_, &QPushButton::clicked, this, [this] { view_->fit(imageScroll_->viewport()->size()); });
	connect(useRanges_, &QCheckBox::toggled, this, [this](bool on) { angle_->setEnabled(!on); for (auto* r : ranges_) r->setEnabled(on); });
	connect(fast_, &QCheckBox::toggled, this, [this](bool on) { subpixel_->setEnabled(!on); });
	connect(table_, &QTableWidget::currentCellChanged, this, [this](int row, int, int, int) { view_->selectResult(row); });
	connect(view_, &ImageView::mouseMoved, this, [this](int x, int y) { sourceInfo_->setText(text("待檢 · ","待检 · ","Source · ")+QString("%1  ·  %2 × %3 px  ·  X %4, Y %5").arg(QFileInfo(sourcePath_).fileName()).arg(source_.width()).arg(source_.height()).arg(x).arg(y)); });
	connect(export_, &QPushButton::clicked, this, [this] {
		const auto path = QFileDialog::getSaveFileName(this, export_->text(), "matches.json", "JSON (*.json)");
		if (!path.isEmpty()) status_->setText(exportResults(path) ? text("已匯出結果", "已导出结果", "Results exported") : text("無法儲存檔案", "无法保存文件", "Cannot save file"));
	});
	connect(engine_, &QComboBox::currentIndexChanged, this, [this] { updateEngineControls(); });
	connect(continuous_, &QCheckBox::toggled, this, [this] { updateEngineControls(); });
	connect(searchColour_,&QCheckBox::toggled,this,[this] {
		pattern_.reset();
#ifdef FPM_GUI_RESEARCH
		session_.reset(); sessionKey_.clear();
#endif
		clearResults();
		view_->setImage(searchColour_->isChecked()?sourceColour_:source_,imageScroll_->viewport()->size());
		updateTemplatePreview(); setBusy(false);
	});
	defaultSettings_=recipeSettings();
	for(auto* button:{openRecipe_,loadSource_,batchButton_,resumeBatch_,stopBatch_,editSettings_}) button->setMinimumHeight(36);
	for(auto* field:recipeFields()) {
		if(auto* v=qobject_cast<QSpinBox*>(field)) connect(v,&QSpinBox::valueChanged,this,&MainWindow::refreshRecipe);
		else if(auto* v=qobject_cast<QDoubleSpinBox*>(field)) connect(v,&QDoubleSpinBox::valueChanged,this,&MainWindow::refreshRecipe);
		else if(auto* v=qobject_cast<QCheckBox*>(field)) connect(v,&QCheckBox::toggled,this,&MainWindow::refreshRecipe);
		else if(auto* v=qobject_cast<QComboBox*>(field)) connect(v,&QComboBox::currentIndexChanged,this,&MainWindow::refreshRecipe);
	}
	setBusy(false); updateEngineControls();
}

QString MainWindow::text(const char* zh, const char* cn, const char* en) const
{
	return QString::fromUtf8(language_->currentIndex() == 0 ? zh : language_->currentIndex() == 1 ? cn : en);
}
void MainWindow::updateLanguage()
{
	crop_->setText(text("從影像製作模板…","从图像制作模板…","Create template from image…"));
#ifdef FPM_GUI_MULTIMEDIA
	saveFrame_->setText(text("儲存取樣影格 PNG…","保存采样帧 PNG…","Save sampled frame as PNG…"));
	liveButton_->setText(live_?text("停止即時取像","停止实时取像","Stop live input"):text("開啟即時來源…","打开实时来源…","Open live source…"));
#endif
	openRecipe_->setText(text("開啟料號配方…","打开料号配方…","Open part recipe…"));
	editSettings_->setText(text("編輯料號設定","编辑料号设置","Edit part settings"));
	saveRecipe_->setText(text("儲存料號配方…","保存料号配方…","Save part recipe…"));
	expectedLabel_->setText(text("每張預期件數","每张预期件数","Expected per image"));
	expected_->setSpecialValueText(text("未設定","未设置","Unspecified"));
	batchButton_->setText(text("批次檢查…","批次检查…","Inspect a batch…"));
	stopBatch_->setText(text("結束本批","结束本批","End batch"));
	loadSource_->setText(templ_.isNull()?text("選擇來源影像…","选择源图像…","Open source image…"):text("選圖並檢查…","选图并检查…","Inspect an image…"));
	refreshRecipe();refreshInspection();refreshBatch();
	sourceInfo_->setVisible(!source_.isNull());templateInfo_->setVisible(!templ_.isNull());
	if(!source_.isNull()) sourceInfo_->setText(text("待檢 · ","待检 · ","Source · ")+QString("%1  ·  %2 × %3 px").arg(QFileInfo(sourcePath_).fileName()).arg(source_.width()).arg(source_.height()));
	if(!templ_.isNull()) templateInfo_->setText(text("模板 · ","模板 · ","Template · ")+QString("%1  ·  %2 × %3 px").arg(QFileInfo(templatePath_).fileName()).arg(templ_.width()).arg(templ_.height()));
	loadTemplate_->setText(text("讀取模板影像", "读取模板图像", "Open template"));
	parameters_->setTitle(text("比對設定", "匹配设置", "Match settings"));
	searchColour_->setText(text("依模板顏色定位","依模板颜色定位","Locate by template colour"));
	searchColour_->setToolTip(text("模板應包含工件色與周圍色。以兩者的色彩對比在整張影像找位置。","模板应包含工件色与周围色。以两者的色彩对比在整张图像找位置。","Include the part colour and its surroundings in the template. Search the whole image by colour contrast."));
	engineLabel_->setText(text("運算方式", "运算方式", "Engine"));
	engine_->setItemText(0,text("CPU 原有流程", "CPU 原有流程", "CPU original"));
	if(engine_->count()>1) engine_->setItemText(1,text("CPU 尺度搜尋", "CPU 尺度搜索", "CPU scale search"));
	if(engine_->findData(Metal)>=0) engine_->setItemText(engine_->findData(Metal),text("Metal 尺度搜尋", "Metal 尺度搜索", "Metal scale search"));
#ifdef FPM_GUI_RESEARCH
	if(engine_->findData(Cuda)>=0) engine_->setItemText(engine_->findData(Cuda),text("CUDA 搜尋＋CPU 姿態精修 (SM80+)","CUDA 搜索＋CPU 姿态精修 (SM80+)","CUDA search + CPU refinement (SM80+)"));
	if(engine_->findData(FullMetal)>=0) engine_->setItemText(engine_->findData(FullMetal),(searchColour_->isChecked()?text("完整 Metal 色彩特徵搜尋","完整 Metal 色彩特征搜索","Full Metal colour-feature search"):text("完整 Metal 灰階搜尋","完整 Metal 灰阶搜索","Full Metal grayscale")));
	scene_->setItemText(0,text("套用場景設定…","应用场景设置…","Apply a scene preset…"));
	scene_->setItemText(1,text("固定方向、原尺寸","固定方向、原尺寸","Upright, original size"));
	scene_->setItemText(2,text("小模板、多角度","小模板、多角度","Small rotated templates"));
	scene_->setItemText(3,text("大量相同物件、多角度","大量相同物件、多角度","Many identical rotated objects"));
	loadTuning_->setText(text("套用已標註樣本的調參結果…","应用已标注样本的调参结果…","Apply calibration result…"));
	loadMask_->setText(text("載入模板驗色遮罩…","载入模板验色掩膜…","Load colour mask…"));
	loadMask_->setToolTip(text("與模板同尺寸的 PNG：純紅標驗色區，純綠標中性參考，可用純藍標第二中性參考。其餘填黑。","与模板同尺寸的 PNG：纯红标验色区，纯绿标中性参考，可用纯蓝标第二中性参考。其余填黑。","Template-sized PNG. Red = colour samples, green = neutral reference, optional blue = darker neutral reference. Black elsewhere."));
	checkColour_->setToolTip(inspectionMask_.path);
	checkColour_->setText(text("定位後驗色","定位后验色","Inspect colour of located objects"));
	colourThresholdLabel_->setText(text("區域色度距離上限","区域色度距离上限","Region chroma limit"));
	colourThreshold_->setSpecialValueText(text("只量測","只测量","Measure only"));
	shortSideLabel_->setText(text("頂層短邊下限 (px)","顶层短边下限 (px)","Top-level short side (px)"));
	shortSide_->setSpecialValueText(text("關閉","关闭","Off"));
	proposalLabel_->setText(text("候選最低分","候选最低分","Proposal threshold"));
	verifySmall_->setText(text("檢查邊界與小模板分數","检查边界与小模板分数","Verify borders and small-template scores"));
	dedup_->setText(text("合併重複候選 (k=3)","合并重复候选 (k=3)","Deduplicate coarse candidates (k=3)"));
#endif
	scaleLabel_->setText(text("尺度 起／迄／間隔", "尺度 起／止／间隔", "Scale from/to/step"));
	scaleMin_->setToolTip(text("尺度候選最小值；1 是原大小。精修可在候選附近微調尺度。", "尺度候选最小值；1 是原大小。精修可在候选附近微调尺度。", "Minimum proposal scale; 1 is original size. Refinement may move beyond the proposal range."));
	scaleMax_->setToolTip(text("尺度候選最大值，最多 64 個候選尺度。", "尺度候选最大值，最多 64 个候选尺度。", "Maximum proposal scale; at most 64 scales."));
	scaleStep_->setToolTip(text("尺度候選間隔；最後會包含最大值。", "尺度候选间隔；最后包含最大值。", "Proposal step; the maximum is included."));
	nccLabel_->setText(text("GPU 資料重用", "GPU 数据复用", "GPU data reuse"));
	nccMode_->setItemText(0,text("原有模式", "原有模式", "Original"));
	nccMode_->setItemText(1,text("相鄰位置重用", "相邻位置复用", "Neighbor reuse"));
	nccMode_->setItemText(2,text("區塊共享（實驗）", "区块共享（实验）", "Shared tile (experimental)"));
	nccMode_->setToolTip(text("相鄰重用在部分大圖較快，小圖不一定；保留原有模式作比較。", "相邻复用在部分大图较快，小图不一定；保留原有模式作比较。", "Neighbor reuse benefits some large workloads; smaller cases can be slower."));
	coarseOverlap_->setText(text("CPU／GPU 重疊執行（較多記憶體）", "CPU／GPU 重叠执行（更多内存）", "Overlap CPU/GPU (more memory)"));
	continuous_->setText(text("連續姿態精修與錯峰救援", "连续姿态精修与错峰救援", "Continuous refinement and rescue"));
	labels_[0]->setText(text("搜尋數量上限", "搜索数量上限", "Max targets"));
	labels_[1]->setText(text("最低分數", "最低分数", "Minimum score"));
	labels_[2]->setText(text("最大重疊比例", "最大重叠比例", "Max overlap"));
	labels_[3]->setText(text("容許旋轉 ±", "允许旋转 ±", "Rotation ±"));
	labels_[4]->setText(text("金字塔縮減面積", "金字塔缩减面积", "Reduced area"));
	useRanges_->setText(text("兩組角度範圍（輸出角度）", "两组角度范围（输出角度）", "Two output angle ranges"));
	useRanges_->setToolTip(text("正值為逆時針；左值須小於或等於右值。", "正值为逆时针；左值须小于等于右值。", "Positive = counterclockwise. Each lower bound must not exceed its upper bound."));
	subpixel_->setText(text("次像素位置與角度精修", "亚像素位置与角度精修", "Subpixel refinement"));
	invert_->setText(text("反轉來源影像明暗", "反转源图像明暗", "Invert source intensities"));
	simd_->setText(text("使用 SIMD 加速", "使用 SIMD 加速", "Use SIMD acceleration"));
	fast_->setText(text("快速模式（半解析度，降低精度）", "快速模式（半分辨率，降低精度）", "Fast mode (half resolution)"));
	execute_->setText(busy_ ? text("比對中…", "匹配中…", "Matching…") : (hasCompleted_?text("重新檢查本張","重新检查本张","Reinspect this image"):text("執行比對", "执行匹配", "Run match")));
	fit_->setText(text("影像符合視窗", "图像适应窗口", "Fit image to view"));
	export_->setText(text("匯出 JSON 結果", "导出 JSON 结果", "Export results as JSON"));
	QStringList headers{text("索引", "索引", "Index"), text("分數", "分数", "Score"), text("角度 °", "角度 °", "Angle °"), "X", "Y", text("尺度", "尺度", "Scale")};
	if(table_->columnCount()==7) headers.append(text("色度距離","色度距离","Chroma distance"));
	table_->setHorizontalHeaderLabels(headers);
}
void MainWindow::clearResults()
{
	inspectionFailed_=false;results_.clear(); resultScales_.clear(); provenance_={}; hasCompleted_=false; table_->setRowCount(0); view_->setResults({}); export_->setEnabled(false); status_->clear();
}
bool MainWindow::loadSource(const QString& path)
{
	return !live_ && !batchActive() && readSource(path);
}
bool MainWindow::readSource(const QString& path)
{
	if (busy_) return false;
	QImage decoded; const auto image = readGray(path,&decoded);
	if (image.isNull()) { clearSource();inspectionFailed_=true;refreshInspection();status_->setText(text("無法讀取影像：", "无法读取图像：", "Cannot read image: ") + path); return false; }
	installSource(image,std::move(decoded),path,false);return true;
}
void MainWindow::installSource(QImage gray,QImage original,const QString& name,bool keepZoom)
{
	source_=std::move(gray);sourcePath_=name;sourceColour_=std::move(original);frameInfo_={};
#ifdef FPM_GUI_MULTIMEDIA
	if(!live_) liveInfo_->hide();
#endif
#ifdef FPM_GUI_RESEARCH
	sessionSourceDirty_=bool(session_);
#endif
	clearResults(); view_->setImage(searchColour_->isChecked()?sourceColour_:source_, imageScroll_->viewport()->size(),keepZoom);
	sourceInfo_->setToolTip(name); setBusy(false);
}
void MainWindow::makeTemplate()
{
	if(batchActive()) return;
#ifdef FPM_GUI_MULTIMEDIA
	if(live_) {
		stopLive();
		if(busy_) {
			connect(this,&MainWindow::matchingFinished,this,[this]{makeTemplate();},Qt::SingleShotConnection);
			return;
		}
	}
#endif
	if(busy_) return;
	if(sourceColour_.isNull()) {
		const auto path=QFileDialog::getOpenFileName(this,crop_->text(),{},"Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff *.pgm);;All files (*)");
		if(path.isEmpty() || !loadSource(path)) return;
	}
	TemplateEditor editor(sourceColour_,language_->currentIndex(),this);
	if(editor.exec()!=QDialog::Accepted) return;
	auto tile=editor.selectedTile();auto gray=recipeGray(tile);
	if(gray.isNull()) {status_->setText(text("無法建立模板","无法创建模板","Could not create template"));return;}
	installTemplate(std::move(gray),std::move(tile),QFileInfo(sourcePath_).completeBaseName()+"-crop.png");
	editSettings_->setChecked(true);
}
bool MainWindow::loadTemplate(const QString& path)
{
	if (live_ || busy_ || batchActive()) return false;
	QImage original;
	const auto image=readGray(path,&original);
	if(image.isNull()) {status_->setText(text("無法讀取模板：","无法读取模板：","Cannot read template: ")+path);return false;}
	installTemplate(image,std::move(original),path);
	return true;
}
void MainWindow::installTemplate(QImage gray,QImage original,const QString& name)
{
	templ_=std::move(gray); templatePath_=name; pattern_.reset();assetsChanged_=true;
	templateColour_=std::move(original); chroma_.reset();
#ifdef FPM_GUI_RESEARCH
	session_.reset();sessionKey_.clear();
	inspectionMask_={};checkColour_->setChecked(false);checkColour_->setEnabled(false);loadMask_->setEnabled(true);
#endif
	clearResults(); updateTemplatePreview();
	templateInfo_->setToolTip(name);setBusy(false);
}
void MainWindow::updateTemplatePreview()
{
	auto preview=searchColour_->isChecked()?templateColour_:templ_;
	if(preview.isNull()) {templatePreview_->clear();return;}
	preview.setDevicePixelRatio(1);
	templatePreview_->setPixmap(QPixmap::fromImage(preview).scaled(290,135,Qt::KeepAspectRatio,Qt::SmoothTransformation));
}
void MainWindow::setBusy(bool busy)
{
	busy_=busy;
	const bool editable=!busy && !batchActive() && !live_;
	parameters_->setEnabled(editable);loadSource_->setEnabled(editable);loadTemplate_->setEnabled(editable);
	openRecipe_->setEnabled(editable);editSettings_->setEnabled(editable);
	saveRecipe_->setEnabled(editable && !templ_.isNull());batchButton_->setEnabled(editable && !templ_.isNull());
	crop_->setEnabled(editable || (live_ && !source_.isNull()));
#ifdef FPM_GUI_MULTIMEDIA
	liveButton_->setEnabled(editable || live_);
	saveFrame_->setVisible(!live_ && !frameInfo_.isEmpty());saveFrame_->setEnabled(editable);
#endif
	templatePreview_->setVisible(!templ_.isNull());
	execute_->setVisible(!live_ && !batchActive() && !source_.isNull() && !templ_.isNull());
	execute_->setEnabled(editable && !source_.isNull() && !templ_.isNull()); export_->setEnabled(!busy && hasCompleted_);
	updateLanguage();
}
void MainWindow::updateEngineControls()
{
	const bool research=engine_->currentData().toInt()!=0,metal=engine_->currentData().toInt()==2;
	for(auto* spin:{scaleMin_,scaleMax_,scaleStep_}) spin->setEnabled(research);
	continuous_->setEnabled(research);nccMode_->setEnabled(metal);coarseOverlap_->setEnabled(metal);
#ifdef FPM_GUI_RESEARCH
	shortSide_->setEnabled(research);verifySmall_->setEnabled(research);dedup_->setEnabled(research);
	proposalScore_->setEnabled(research && continuous_->isChecked());
#endif
}
#ifdef FPM_GUI_RESEARCH
bool MainWindow::loadInspectionMask(const QString& path)
{
	if(live_ || busy_ || batchActive() || templ_.isNull()) return false;
	try {
		inspectionMask_=InspectionMask::read(path,templateColour_);
		assetsChanged_=true;checkColour_->setEnabled(true);checkColour_->setChecked(true);checkColour_->setToolTip(path);refreshRecipe();return true;
	} catch(const std::exception& e) {status_->setText(QString::fromUtf8(e.what()));return false;}
}

bool MainWindow::loadTuning(const QString& path)
{
	if(live_ || busy_ || batchActive()) return false;
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly)) { status_->setText(file.errorString());return false; }
	const auto data=QJsonDocument::fromJson(file.readAll()).object();
	const int backend=data["backend"].toInt(-1);
	const int route=backend==0?ResearchCpu:(backend==1?Metal:Cuda);
	const QString choice=data["selected_on_dev"].toString();
	const auto arms=data["arms"].toObject();
	if(data["schema"]!="fpm.scene-tuning.v1" || backend<0 || backend>2 || engine_->findData(route)<0 ||
	   choice.isEmpty() || !arms.value(choice).isObject()) {
		status_->setText(text("調參檔未選出可用設定，或此裝置缺少量測時的後端。","调参文件未选出可用设置，或此设备缺少测量时的后端。","No usable calibration or its measured backend is unavailable."));return false;
	}
	auto job=data["baseline"].toObject();
	const auto changes=arms.value(choice).toObject();
	for(auto it=changes.begin();it!=changes.end();++it) job[it.key()]=it.value();
	// Reconstruct a bank representable by the desktop's min/max/step controls.
	const auto scales=job["scales"].toArray();
	const double lo=scales.isEmpty()?0.:scales.first().toDouble(), hi=scales.isEmpty()?0.:scales.last().toDouble();
	const double step=scales.size()>1?scales.at(1).toDouble()-lo:.25;
	bool validScales=!scales.isEmpty() && scales.size()<=64 && lo>=.01 && hi<=8 && step>=.01-1.e-10;
	for(qsizetype i=0;i<scales.size();++i) {
		const double x=scales[i].toDouble();
		validScales=validScales && scales[i].isDouble() && std::abs(x*1000-std::round(x*1000))<1.e-8 &&
			(i==0 || x>scales[i-1].toDouble()) && std::abs(x-std::min(hi,lo+i*step))<1.e-8;
	}
	const auto validNumber=[&](const char* key,double low,double high,bool integral=false) {
		const auto v=job[key];const double x=v.toDouble();
		return v.isDouble() && x>=low && x<=high && (!integral || x==std::floor(x));
	};
	const bool verify=job.contains("border_min_inside") || job.contains("small_template_area");
	if(!validNumber("angle",0,180) || !validNumber("score",0,1) || !validNumber("proposal_score",0,1) ||
	   !validNumber("limit",1,10000,true) || !validNumber("overlap",0,1) || !validNumber("stop_layer",0,1,true) ||
	   !validNumber("min_area",1,1000000,true) || !validScales ||
	   (job.contains("min_top_short_side") && !validNumber("min_top_short_side",0,32768,true)) ||
	   (verify && (job["border_min_inside"].toDouble()!=.5 || job["small_template_area"].toDouble()!=576))) {
		status_->setText(text("調參檔中的參數無效。","调参文件中的参数无效。","Invalid calibration settings."));return false;
	}
	engine_->setCurrentIndex(engine_->findData(route));applyScene(2);
	scaleMin_->setValue(lo);scaleMax_->setValue(hi);scaleStep_->setValue(step);
	angle_->setValue(job["angle"].toDouble());targets_->setValue(job["limit"].toInt());minArea_->setValue(job["min_area"].toInt());
	shortSide_->setValue(job["min_top_short_side"].toInt());score_->setValue(job["score"].toDouble());
	proposalScore_->setValue(job["proposal_score"].toDouble());overlap_->setValue(job["overlap"].toDouble());
	fast_->setChecked(job["stop_layer"].toInt()==1);verifySmall_->setChecked(verify);
	simd_->setChecked(true);invert_->setChecked(false);nccMode_->setCurrentIndex(0);
	status_->setText(text("已套用調參結果 · ","已应用调参结果 · ","Applied calibration · ")+choice);return true;
}

void MainWindow::applyScene(int index)
{
	if(index<1 || index>3) return;
	int route=engine_->currentData().toInt();
	if(route==Original || (index==3 && route==Metal)) {
		route=ResearchCpu;
		if(engine_->findData(Cuda)>=0) route=Cuda;
		else if(engine_->findData(Metal)>=0) route=index==3?FullMetal:Metal;
		engine_->setCurrentIndex(engine_->findData(route));
	}
	useRanges_->setChecked(false);angle_->setValue(index==1?0:180);
	scaleMin_->setValue(1);scaleMax_->setValue(1);scaleStep_->setValue(.25);
	minArea_->setValue(256);shortSide_->setValue(index==2?12:0);verifySmall_->setChecked(index==2);
	dedup_->setChecked(index==3);targets_->setValue(index==3?600:100);score_->setValue(.8);
	proposalScore_->setValue(.5);overlap_->setValue(.2);continuous_->setChecked(index!=1);
	fast_->setChecked(index!=1);subpixel_->setChecked(false);coarseOverlap_->setChecked(false);
	// This control applies editable settings. It never labels later manual edits as a preset.
	scene_->setCurrentIndex(0);updateEngineControls();
}
#endif
void MainWindow::execute()
{
	if (busy_ || source_.isNull() || templ_.isNull()) return;
	const auto error=configError();
	if(!error.isEmpty()) {clearResults();inspectionFailed_=true;status_->setText(error);refreshInspection();emit matchingFinished(false);return;}
	fpm::MatchConfig config;
	config.maxTargets = targets_->value(); config.minScore = score_->value(); config.maxOverlap = overlap_->value();
	config.toleranceAngle = angle_->value(); config.subpixel = subpixel_->isChecked(); config.invertSource = invert_->isChecked();
	config.backend = simd_->isChecked() ? fpm::Backend::Auto : fpm::Backend::Scalar;
	config.stopLayer = fast_->isChecked() ? 1 : 0;
	if (config.stopLayer) config.subpixel = false;
	if (useRanges_->isChecked()) for (int i = 0; i < 2; ++i) config.angleRanges.push_back({ranges_[i*2]->value(), ranges_[i*2+1]->value()});
	const int area = minArea_->value(), engine=engine_->currentData().toInt();
	const auto cached = learnedArea_ == area ? pattern_ : nullptr;
	const QImage source = source_, templ = templ_;
	const auto colourSource=sourceColour_,colourTemplate=templateColour_;
	const bool searchColour=searchColour_->isChecked();
	const auto cachedChroma=chroma_;
	const auto flatMessage=text("色彩定位需要兩種有對比的顏色。請把工件邊緣與周圍背景一起框進模板，或改用灰階定位。","色彩定位需要两种有对比的颜色。请把工件边缘与周围背景一起框进模板，或改用灰阶定位。","Chroma search needs contrasting colours. Include the part edge and background in the template, or use grayscale.");
	QJsonArray angleRanges;for(auto r:config.angleRanges) angleRanges.append(QJsonArray{r.min,r.max});
	const char* engineNames[]={"cpu_original","cpu_research","metal_research","cuda_search_cpu_pose","full_metal"};
	QJsonObject provenance{{"engine",engineNames[engine]},{"search_feature",searchColour?"chroma_cocg":"gray"},
		{"recipe",QFileInfo(recipePath_).completeBaseName()},{"recipe_sha256",recipeDigest_},
		{"recipe_modified",!recipePath_.isEmpty() && (assetsChanged_ || savedSettings_!=recipeSettings())},
		{"inspection_started_utc",QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},{"expected_count",expected_->value()},
		{"source",sourcePath_},{"template",templatePath_},{"source_width",source.width()},{"source_height",source.height()},
		{"max_targets",config.maxTargets},{"minimum_score",config.minScore},{"max_overlap",config.maxOverlap},
		{"rotation_tolerance",config.toleranceAngle},{"angle_ranges",angleRanges},{"subpixel",config.subpixel},
		{"invert_source",config.invertSource},{"stop_layer",config.stopLayer},{"min_reduced_area",area}};
	if(!frameInfo_.isEmpty()) provenance["input_frame"]=frameInfo_;
#ifdef FPM_GUI_RESEARCH
	fpm::research::Settings settings;settings.backend=engine==Cuda?2:(engine==Metal || engine==FullMetal?1:0);
	settings.fullGpu=engine==FullMetal;settings.minReducedArea=area;settings.scales.clear();
	settings.minTopShortSide=engine?shortSide_->value():0;
	fpm::research::Options options;options.match=config;options.nccMode=engine==2?nccMode_->currentData().toInt():0;
	options.overlap=engine==2 && coarseOverlap_->isChecked();options.refine=continuous_->isChecked();
	options.proposalScore=proposalScore_->value();
	if(engine && verifySmall_->isChecked()) options.verify={.5,576};
	options.candidateDedupTopK=engine && dedup_->isChecked()?3:0;
	if(settings.fullGpu) { options.fullGpuPeakMode=2; options.fullGpuFineMode=4; }
	if(engine) {
		const double lo=scaleMin_->value(),hi=scaleMax_->value(),step=scaleStep_->value();
		if(lo>hi || (hi-lo)/step>63+1e-9) {
			clearResults();inspectionFailed_=true;refreshInspection();status_->setText(text("尺度起點須小於等於終點，且最多 64 個尺度。", "尺度起点须小于等于终点，且最多 64 个尺度。", "Scale from must not exceed to; at most 64 scales."));emit matchingFinished(false);return;
		}
		for(int i=0;i<64 && lo+i*step<hi-1e-9;++i) settings.scales.push_back(lo+i*step);
		settings.scales.push_back(hi);
	}
	QJsonArray scales;for(double scale:settings.scales) scales.append(scale);
	provenance["proposal_scales"]=scales;provenance["ncc_mode"]=options.nccMode;provenance["coarse_overlap"]=options.overlap;
	provenance["continuous_refinement"]=engine && options.refine;provenance["cpu_final_verification"]=engine && !settings.fullGpu && options.refine;
	provenance["min_top_short_side"]=settings.minTopShortSide;provenance["proposal_score"]=options.proposalScore;
	provenance["candidate_dedup_top_k"]=options.candidateDedupTopK;provenance["border_min_inside"]=options.verify.borderMinInside;
	provenance["small_template_area"]=options.verify.smallTemplateArea;
	if(engine==Cuda) { provenance["ncc_mode"]=6; provenance["fine_ncc_mode"]=5; provenance["transfer_mode"]=3; }
	if(settings.fullGpu) { provenance["full_gpu_peak_mode"]=2; provenance["full_gpu_fine_mode"]=4; }
	const QString key=QString::number(engine)+":"+QString::number(area)+":"+QString::number(settings.minTopShortSide)+":"+QString::fromUtf8(QJsonDocument(scales).toJson(QJsonDocument::Compact));
	const auto cachedSession=key==sessionKey_?session_:nullptr;
	const bool updateSource=cachedSession && sessionSourceDirty_;
	const auto colourMask=checkColour_->isChecked()?inspectionMask_:InspectionMask{};
	const double colourLimit=colourThreshold_->value();
#endif
	clearResults(); setBusy(true); status_->setText(text("正在比對…可繼續縮放檢視影像。", "正在匹配…可继续缩放查看图像。", "Matching… You can continue zooming the image."));
	watcher_.setFuture(QtConcurrent::run([source, templ, config, area, cached, engine, provenance, searchColour, cachedChroma, colourSource, colourTemplate, flatMessage
#ifdef FPM_GUI_RESEARCH
		,settings,options,key,cachedSession,updateSource,colourMask,colourLimit
#endif
	] {
		MatchJobResult result;result.provenance=provenance;
		try {
			QElapsedTimer timer; timer.start();
			auto srcView=imageView(source), tmplView=imageView(templ);
			fpm::Image projected;
			if(searchColour) {
				result.chroma=cachedChroma;
				if(!result.chroma) {
					const auto rgb=colourTemplate.convertToFormat(QImage::Format_RGB888);
					try { result.chroma=std::make_shared<fpm::ChromaTemplate>(rgbView(rgb)); }
					catch(const std::domain_error&) { result.error=flatMessage;return result; }
				}
				tmplView=result.chroma->feature().view();
#ifdef FPM_GUI_RESEARCH
				if(!engine || !cachedSession || updateSource)
#endif
				{
					const auto rgb=colourSource.convertToFormat(QImage::Format_RGB888);
					projected=result.chroma->project(rgbView(rgb)); srcView=projected.view();
				}
				result.provenance["feature_ms"]=timer.nsecsElapsed()/1e6; timer.restart();
			}
#ifdef FPM_GUI_RESEARCH
			if(engine) {
				result.session=cachedSession?cachedSession:std::make_shared<fpm::research::Session>(srcView,tmplView,settings);
				if(updateSource) result.session->updateSource(srcView);
				result.provenance["session_reused"]=bool(cachedSession);
				result.provenance["source_updated"]=updateSource;
				result.sessionKey=key;result.learnMs=timer.nsecsElapsed()/1e6;
				const auto output=result.session->run(options);result.matches=output.matches;result.scales=output.scales;result.matchMs=output.totalMs;
				if(!settings.fullGpu) result.provenance["frame_generation"]=double(output.frame.generation);
				result.provenance["source_generation"]=double(output.sourceGeneration);
				result.provenance["prepare_ms"]=output.prepareMs;result.provenance["proposal_ms"]=output.proposalMs;
				result.provenance["refine_ms"]=output.refineMs;result.provenance["nms_ms"]=output.nmsMs;
				result.provenance["frame_gpu_retained_bytes"]=double(output.frame.coarse_gpu_bytes+output.frame.fine_gpu_bytes);
				result.provenance["pose_gpu_retained_bytes"]=double(output.pose.gpu_buffer_bytes);
				result.provenance["rescued_seeds"]=double(output.rescue.selected_seeds);
				result.provenance["refinement_backend"]=output.rescue.selected_backend;
				result.provenance["configured_refinement_backend"]=output.configuredRefinementBackend;
				result.provenance["refinement_candidates"]=double(output.refinementCandidates);
				result.provenance["proposal_count"]=double(output.proposals.size());
				result.provenance["verification_rejected_border"]=double(output.verifyRejectedBorder);
				result.provenance["verification_rejected_size"]=double(output.verifyRejectedSize);
				if(settings.fullGpu) {
					result.provenance["full_gpu_retained_bytes"]=double(output.fullGpu.gpuBytes);
					result.provenance["full_gpu_device_ms"]=output.fullGpu.deviceMs;
					result.provenance["full_gpu_intermediate_readbacks"]=double(output.fullGpu.intermediateReadbacks);
				}
			} else
#endif
			{
				result.pattern = cached ? cached : std::make_shared<const fpm::Pattern>(tmplView, area);
				result.learnMs = timer.nsecsElapsed() / 1e6; timer.restart();
				result.matches = fpm::match(srcView, *result.pattern, config);
				result.matchMs = timer.nsecsElapsed() / 1e6;result.scales.assign(result.matches.size(),1.);
			}
			result.provenance["match_ms"]=result.matchMs;result.provenance["setup_ms"]=result.learnMs;
#ifdef FPM_GUI_RESEARCH
			if(!colourMask.path.isEmpty()) {
				timer.restart();
				result.provenance["colour_results"]=inspectColours(colourSource,colourTemplate,colourMask,result.matches,result.scales,colourLimit);
				result.provenance["colour_ms"]=timer.nsecsElapsed()/1e6;result.provenance["colour_threshold"]=colourLimit;
				result.provenance["colour_mask"]=colourMask.path;
			}
#endif
		} catch (const std::exception& e) { result.error = QString::fromUtf8(e.what()); }
		catch (...) { result.error = "Unknown matching error"; }
		return result;
	}));
}

void MainWindow::complete()
{
	const auto result = watcher_.result();
	if (result.error.isEmpty()) {
		chroma_=result.chroma; pattern_ = result.pattern; learnedArea_ = minArea_->value(); results_ = result.matches;
		resultScales_=result.scales;provenance_=result.provenance;hasCompleted_=true;
#ifdef FPM_GUI_RESEARCH
		session_=result.session;sessionKey_=result.sessionKey;sessionSourceDirty_=false;
#endif
		table_->setRowCount(int(results_.size()));
		const bool colour=provenance_.contains("colour_results");
		table_->setColumnCount(colour?7:6);

		for (int i = 0; i < int(results_.size()); ++i) {
			const auto& r = results_[i];
			const QStringList values{QString::number(i), QString::number(r.score, 'f', 6), QString::number(r.angle, 'f', 3), QString::number(r.center.x, 'f', 3), QString::number(r.center.y, 'f', 3), QString::number(resultScales_[i], 'f', 5)};
			for (int col = 0; col < 6; ++col) table_->setItem(i, col, new QTableWidgetItem(values[col]));
			if(colour) {
				const auto row=provenance_["colour_results"].toArray().at(i).toObject();
				QString value=text("需複核","需复核","Review");
				if(row["status"].toInt()==0) {
					value=QString::number(row["region_distance"].toDouble(),'f',5);
					if(row["decision"]=="outside") value+=text(" 超限"," 超限"," outside");
				}
				table_->setItem(i,6,new QTableWidgetItem(value));
			}
		}
		int colourReview=0,colourOutside=0;
		for(const auto item:provenance_["colour_results"].toArray()) {
			const auto decision=item.toObject()["decision"].toString();
			colourReview+=decision=="review";colourOutside+=decision=="outside";
		}
		const int expected=provenance_["expected_count"].toInt();
		const bool needsReview=results_.empty() || (expected>0 && int(results_.size())!=expected) || colourReview || colourOutside;
		provenance_["inspection"]=QJsonObject{{"needs_review",needsReview},{"found",int(results_.size())},{"expected",expected},
			{"colour_review",colourReview},{"colour_outside",colourOutside}};
		view_->setResults(results_);
		if(colourReview || colourOutside) {
			const auto rows=provenance_["colour_results"].toArray();
			for(qsizetype row=0;row<rows.size();++row) {
				const auto decision=rows[row].toObject()["decision"].toString();
				if(decision=="outside" || decision=="review") {table_->setCurrentCell(int(row),6);break;}
			}
		}
		status_->setText(text("找到 %1 個目標 · 比對 %2 ms · 學習 %3 ms", "找到 %1 个目标 · 匹配 %2 ms · 学习 %3 ms", "%1 matches · Match %2 ms · Learn %3 ms").arg(results_.size()).arg(result.matchMs, 0, 'f', 1).arg(result.learnMs, 0, 'f', 1));
		status_->setText(engine_->currentText()+" · "+status_->text());
		if(provenance_.contains("feature_ms")) status_->setText(status_->text()+text(" · 色彩特徵 %1 ms"," · 色彩特征 %1 ms"," · Colour features %1 ms").arg(provenance_["feature_ms"].toDouble(),0,'f',1));
		if(colour) status_->setText(status_->text()+text(" · 驗色 %1 ms"," · 验色 %1 ms"," · Colour %1 ms").arg(provenance_["colour_ms"].toDouble(),0,'f',1));
	} else {inspectionFailed_=true;status_->setText(result.error);}
	setBusy(false); emit matchingFinished(result.error.isEmpty());
}
bool MainWindow::exportResults(const QString& path) const
{
	if(busy_ || !hasCompleted_) return false;
	QJsonArray matches;
	for (std::size_t i = 0; i < results_.size(); ++i) {
		const auto& m = results_[i]; QJsonArray corners;
		for (const auto& p : m.corners) corners.append(QJsonArray{p.x, p.y});
		matches.append(QJsonObject{{"index", int(i)}, {"score", m.score}, {"angle", m.angle}, {"pos_x", m.center.x}, {"pos_y", m.center.y}, {"corners", corners}, {"scale", resultScales_[i]}});
	}
	QSaveFile file(path);
	if (!file.open(QIODevice::WriteOnly)) return false;
	const auto bytes = QJsonDocument(QJsonObject{{"matches", matches},{"run",provenance_},{"schema_version",2}}).toJson();
	return file.write(bytes) == bytes.size() && file.commit();
}
void MainWindow::dragEnterEvent(QDragEnterEvent* event)
{
	if (!live_ && !busy_ && !batchActive() && event->mimeData()->hasUrls()) event->acceptProposedAction();
}
void MainWindow::dropEvent(QDropEvent* event)
{
	if (live_ || busy_ || batchActive() || event->mimeData()->urls().isEmpty()) return;
	QStringList paths;
	for(const auto& url:event->mimeData()->urls()) {
		if(!url.isLocalFile()) return;
		paths.append(url.toLocalFile());
	}
	if(paths.size()>1) {
		if(templ_.isNull()) {status_->setText(text("請先選擇料號配方或模板。","请先选择料号配方或模板。","Select a recipe or template first."));return;}
		chooseBatchOutput(paths);return;
	}
	const auto& path=paths.first();
	if(path.endsWith(".fpmrecipe",Qt::CaseInsensitive)) {loadRecipe(path);return;}
	if (editSettings_->isChecked() && event->position().x() >= imageScroll_->mapTo(this,QPoint(imageScroll_->width(),0)).x()) loadTemplate(path);
	else if(loadSource(path) && !templ_.isNull()) execute();
}
