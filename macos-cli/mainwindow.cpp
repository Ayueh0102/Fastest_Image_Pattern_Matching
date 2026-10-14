#include "mainwindow.h"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QFileDialog>
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

namespace {
fpm::ImageView imageView(const QImage& image)
{
	return {image.constBits(), image.width(), image.height(), std::size_t(image.bytesPerLine())};
}
QImage readGray(const QString& path)
{
	QImageReader reader(path);
	// Keep stored pixel orientation, matching the reference image coordinates.
	reader.setAutoTransform(false);
	const auto rgb = reader.read().convertToFormat(QImage::Format_RGB888);
	if (rgb.isNull()) return {};
	QImage gray(rgb.size(), QImage::Format_Grayscale8);
	for (int y = 0; y < rgb.height(); ++y)
		for (int x = 0; x < rgb.width(); ++x) {
			const auto* p = rgb.constScanLine(y) + 3 * x;
			gray.scanLine(y)[x] = fpm::grayFromRgb(p[0], p[1], p[2]);
		}
	return gray;
}
}

ImageView::ImageView(QWidget* parent) : QWidget(parent) { setMouseTracking(true); }
void ImageView::setImage(const QImage& image, const QSize& viewport)
{
	image_ = image; results_.clear(); selected_ = -1; fit(viewport);
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
	panelScroll->setMaximumWidth(420); main->addWidget(panelScroll);
	language_ = new QComboBox; language_->addItems({"繁體中文", "简体中文", "English"}); side->addWidget(language_);
	loadSource_ = new QPushButton; loadTemplate_ = new QPushButton;
	loadSource_->setObjectName("loadSource"); loadTemplate_->setObjectName("loadTemplate");
	side->addWidget(loadSource_); sourceInfo_ = new QLabel; side->addWidget(sourceInfo_);
	side->addWidget(loadTemplate_); templateInfo_ = new QLabel; side->addWidget(templateInfo_);
	templatePreview_ = new QLabel; templatePreview_->setMinimumHeight(120); templatePreview_->setMaximumHeight(150);
	templatePreview_->setAlignment(Qt::AlignCenter); side->addWidget(templatePreview_);
	parameters_ = new QGroupBox; auto* form = new QFormLayout(parameters_);
	engine_ = new QComboBox; engine_->setObjectName("engine"); engine_->addItem("",0);
#ifdef FPM_GUI_RESEARCH
	engine_->addItem("",1);
#ifdef Q_OS_MACOS
	engine_->addItem("",2);
#endif
#endif
	engineLabel_ = new QLabel; form->addRow(engineLabel_,engine_);
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
	form->addRow(subpixel_); form->addRow(invert_); form->addRow(simd_); form->addRow(fast_); side->addWidget(parameters_);
	execute_ = new QPushButton; execute_->setObjectName("execute"); execute_->setMinimumHeight(36); side->addWidget(execute_);
	fit_ = new QPushButton; side->addWidget(fit_);
	export_ = new QPushButton; export_->setObjectName("export"); export_->setEnabled(false); side->addWidget(export_); side->addStretch();
	status_ = new QLabel; status_->setObjectName("status"); status_->setWordWrap(true); statusBar()->addWidget(status_, 1);
	connect(language_, &QComboBox::currentIndexChanged, this, &MainWindow::updateLanguage);
	connect(loadSource_, &QPushButton::clicked, this, [this] { const auto p = QFileDialog::getOpenFileName(this, loadSource_->text(), {}, "Images (*.bmp *.png *.jpg *.jpeg *.pgm);;All files (*)"); if (!p.isEmpty()) loadSource(p); });
	connect(loadTemplate_, &QPushButton::clicked, this, [this] { const auto p = QFileDialog::getOpenFileName(this, loadTemplate_->text(), {}, "Images (*.bmp *.png *.jpg *.jpeg *.pgm);;All files (*)"); if (!p.isEmpty()) loadTemplate(p); });
	connect(execute_, &QPushButton::clicked, this, &MainWindow::execute);
	connect(&watcher_, &QFutureWatcher<MatchJobResult>::finished, this, &MainWindow::complete);
	connect(fit_, &QPushButton::clicked, this, [this] { view_->fit(imageScroll_->viewport()->size()); });
	connect(useRanges_, &QCheckBox::toggled, this, [this](bool on) { angle_->setEnabled(!on); for (auto* r : ranges_) r->setEnabled(on); });
	connect(fast_, &QCheckBox::toggled, this, [this](bool on) { subpixel_->setEnabled(!on); });
	connect(table_, &QTableWidget::currentCellChanged, this, [this](int row, int, int, int) { view_->selectResult(row); });
	connect(view_, &ImageView::mouseMoved, this, [this](int x, int y) { sourceInfo_->setText(QString("%1 × %2 px   ·   X %3, Y %4").arg(source_.width()).arg(source_.height()).arg(x).arg(y)); });
	connect(export_, &QPushButton::clicked, this, [this] {
		const auto path = QFileDialog::getSaveFileName(this, export_->text(), "matches.json", "JSON (*.json)");
		if (!path.isEmpty()) status_->setText(exportResults(path) ? text("已匯出結果", "已导出结果", "Results exported") : text("無法儲存檔案", "无法保存文件", "Cannot save file"));
	});
	connect(engine_, &QComboBox::currentIndexChanged, this, [this] { updateEngineControls(); });
	updateLanguage(); setBusy(false); updateEngineControls();
}

QString MainWindow::text(const char* zh, const char* cn, const char* en) const
{
	return QString::fromUtf8(language_->currentIndex() == 0 ? zh : language_->currentIndex() == 1 ? cn : en);
}
void MainWindow::updateLanguage()
{
	loadSource_->setText(text("讀取來源影像", "读取源图像", "Open source image"));
	loadTemplate_->setText(text("讀取模板影像", "读取模板图像", "Open template"));
	parameters_->setTitle(text("比對設定", "匹配设置", "Match settings"));
	engineLabel_->setText(text("運算方式", "运算方式", "Engine"));
	engine_->setItemText(0,text("CPU 原有流程", "CPU 原有流程", "CPU original"));
	if(engine_->count()>1) engine_->setItemText(1,text("CPU 尺度搜尋", "CPU 尺度搜索", "CPU scale search"));
	if(engine_->count()>2) engine_->setItemText(2,text("Metal 尺度搜尋", "Metal 尺度搜索", "Metal scale search"));
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
	labels_[0]->setText(text("目標數量", "目标数量", "Max targets"));
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
	execute_->setText(busy_ ? text("比對中…", "匹配中…", "Matching…") : text("執行比對", "执行匹配", "Run match"));
	fit_->setText(text("影像符合視窗", "图像适应窗口", "Fit image to view"));
	export_->setText(text("匯出 JSON 結果", "导出 JSON 结果", "Export results as JSON"));
	table_->setHorizontalHeaderLabels({text("索引", "索引", "Index"), text("分數", "分数", "Score"), text("角度 °", "角度 °", "Angle °"), "X", "Y", text("尺度", "尺度", "Scale")});
}
void MainWindow::clearResults()
{
	results_.clear(); resultScales_.clear(); provenance_={}; hasCompleted_=false; table_->setRowCount(0); view_->setResults({}); export_->setEnabled(false); status_->clear();
}
bool MainWindow::loadSource(const QString& path)
{
	if (busy_) return false;
	const auto image = readGray(path);
	if (image.isNull()) { status_->setText(text("無法讀取影像：", "无法读取图像：", "Cannot read image: ") + path); return false; }
	source_ = image; sourcePath_=path;
#ifdef FPM_GUI_RESEARCH
	session_.reset();sessionKey_.clear();
#endif
	clearResults(); view_->setImage(source_, imageScroll_->viewport()->size());
	sourceInfo_->setText(QString("%1 × %2 px").arg(source_.width()).arg(source_.height())); setBusy(false); return true;
}
bool MainWindow::loadTemplate(const QString& path)
{
	if (busy_) return false;
	const auto image = readGray(path);
	if (image.isNull()) { status_->setText(text("無法讀取模板：", "无法读取模板：", "Cannot read template: ") + path); return false; }
	templ_ = image; templatePath_=path; pattern_.reset();
#ifdef FPM_GUI_RESEARCH
	session_.reset();sessionKey_.clear();
#endif
	clearResults();
	templatePreview_->setPixmap(QPixmap::fromImage(templ_).scaled(290, 135, Qt::KeepAspectRatio, Qt::SmoothTransformation));
	templateInfo_->setText(QString("%1 × %2 px").arg(templ_.width()).arg(templ_.height())); setBusy(false); return true;
}
void MainWindow::setBusy(bool busy)
{
	busy_ = busy; parameters_->setEnabled(!busy); loadSource_->setEnabled(!busy); loadTemplate_->setEnabled(!busy);
	execute_->setEnabled(!busy && !source_.isNull() && !templ_.isNull()); export_->setEnabled(!busy && hasCompleted_);
	updateLanguage();
}
void MainWindow::updateEngineControls()
{
	const bool research=engine_->currentData().toInt()!=0,metal=engine_->currentData().toInt()==2;
	for(auto* spin:{scaleMin_,scaleMax_,scaleStep_}) spin->setEnabled(research);
	continuous_->setEnabled(research);nccMode_->setEnabled(metal);coarseOverlap_->setEnabled(metal);
}
void MainWindow::execute()
{
	if (busy_ || source_.isNull() || templ_.isNull()) return;
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
	QJsonArray angleRanges;for(auto r:config.angleRanges) angleRanges.append(QJsonArray{r.min,r.max});
	QJsonObject provenance{{"engine",engine==0?"cpu_original":engine==1?"cpu_research":"metal_research"},
		{"source",sourcePath_},{"template",templatePath_},{"source_width",source.width()},{"source_height",source.height()},
		{"max_targets",config.maxTargets},{"minimum_score",config.minScore},{"max_overlap",config.maxOverlap},
		{"rotation_tolerance",config.toleranceAngle},{"angle_ranges",angleRanges},{"subpixel",config.subpixel},
		{"invert_source",config.invertSource},{"stop_layer",config.stopLayer},{"min_reduced_area",area}};
#ifdef FPM_GUI_RESEARCH
	fpm::research::Settings settings;settings.backend=engine==2?1:0;settings.minReducedArea=area;settings.scales.clear();
	fpm::research::Options options;options.match=config;options.nccMode=engine==2?nccMode_->currentData().toInt():0;
	options.overlap=engine==2 && coarseOverlap_->isChecked();options.refine=continuous_->isChecked();
	if(engine) {
		const double lo=scaleMin_->value(),hi=scaleMax_->value(),step=scaleStep_->value();
		if(lo>hi || (hi-lo)/step>63+1e-9) {
			clearResults();status_->setText(text("尺度起點須小於等於終點，且最多 64 個尺度。", "尺度起点须小于等于终点，且最多 64 个尺度。", "Scale from must not exceed to; at most 64 scales."));emit matchingFinished(false);return;
		}
		for(int i=0;i<64 && lo+i*step<hi-1e-9;++i) settings.scales.push_back(lo+i*step);
		settings.scales.push_back(hi);
	}
	QJsonArray scales;for(double scale:settings.scales) scales.append(scale);
	provenance["proposal_scales"]=scales;provenance["ncc_mode"]=options.nccMode;provenance["coarse_overlap"]=options.overlap;
	provenance["continuous_refinement"]=engine && options.refine;provenance["cpu_final_verification"]=engine && options.refine;
	const QString key=QString::number(engine)+":"+QString::number(area)+":"+QString::fromUtf8(QJsonDocument(scales).toJson(QJsonDocument::Compact));
	const auto cachedSession=key==sessionKey_?session_:nullptr;
#endif
	clearResults(); setBusy(true); status_->setText(text("正在比對…可繼續縮放檢視影像。", "正在匹配…可继续缩放查看图像。", "Matching… You can continue zooming the image."));
	watcher_.setFuture(QtConcurrent::run([source, templ, config, area, cached, engine, provenance
#ifdef FPM_GUI_RESEARCH
		,settings,options,key,cachedSession
#endif
	] {
		MatchJobResult result;result.provenance=provenance;
		try {
			QElapsedTimer timer; timer.start();
#ifdef FPM_GUI_RESEARCH
			if(engine) {
				result.session=cachedSession?cachedSession:std::make_shared<fpm::research::Session>(imageView(source),imageView(templ),settings);
				result.sessionKey=key;result.learnMs=timer.nsecsElapsed()/1e6;
				const auto output=result.session->run(options);result.matches=output.matches;result.scales=output.scales;result.matchMs=output.totalMs;
				result.provenance["frame_generation"]=double(output.frame.generation);
				result.provenance["prepare_ms"]=output.prepareMs;result.provenance["proposal_ms"]=output.proposalMs;
				result.provenance["refine_ms"]=output.refineMs;result.provenance["nms_ms"]=output.nmsMs;
				result.provenance["frame_gpu_retained_bytes"]=double(output.frame.coarse_gpu_bytes+output.frame.fine_gpu_bytes);
				result.provenance["pose_gpu_retained_bytes"]=double(output.pose.gpu_buffer_bytes);
				result.provenance["rescued_seeds"]=double(output.rescue.selected_seeds);
				result.provenance["refinement_backend"]=output.rescue.selected_backend;
			} else
#endif
			{
				result.pattern = cached ? cached : std::make_shared<const fpm::Pattern>(imageView(templ), area);
				result.learnMs = timer.nsecsElapsed() / 1e6; timer.restart();
				result.matches = fpm::match(imageView(source), *result.pattern, config);
				result.matchMs = timer.nsecsElapsed() / 1e6;result.scales.assign(result.matches.size(),1.);
			}
			result.provenance["match_ms"]=result.matchMs;result.provenance["setup_ms"]=result.learnMs;
		} catch (const std::exception& e) { result.error = QString::fromUtf8(e.what()); }
		catch (...) { result.error = "Unknown matching error"; }
		return result;
	}));
}

void MainWindow::complete()
{
	const auto result = watcher_.result();
	if (result.error.isEmpty()) {
		pattern_ = result.pattern; learnedArea_ = minArea_->value(); results_ = result.matches;
		resultScales_=result.scales;provenance_=result.provenance;hasCompleted_=true;
#ifdef FPM_GUI_RESEARCH
		session_=result.session;sessionKey_=result.sessionKey;
#endif
		table_->setRowCount(int(results_.size()));
		for (int i = 0; i < int(results_.size()); ++i) {
			const auto& r = results_[i];
			const QStringList values{QString::number(i), QString::number(r.score, 'f', 6), QString::number(r.angle, 'f', 3), QString::number(r.center.x, 'f', 3), QString::number(r.center.y, 'f', 3), QString::number(resultScales_[i], 'f', 5)};
			for (int col = 0; col < 6; ++col) table_->setItem(i, col, new QTableWidgetItem(values[col]));
		}
		view_->setResults(results_);
		status_->setText(text("找到 %1 個目標 · 比對 %2 ms · 學習 %3 ms", "找到 %1 个目标 · 匹配 %2 ms · 学习 %3 ms", "%1 matches · Match %2 ms · Learn %3 ms").arg(results_.size()).arg(result.matchMs, 0, 'f', 1).arg(result.learnMs, 0, 'f', 1));
	} else status_->setText(result.error);
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
	if (!busy_ && event->mimeData()->hasUrls()) event->acceptProposedAction();
}
void MainWindow::dropEvent(QDropEvent* event)
{
	if (busy_ || event->mimeData()->urls().isEmpty()) return;
	const auto path = event->mimeData()->urls().first().toLocalFile();
	if (path.isEmpty()) return;
	if (event->position().x() < imageScroll_->mapTo(this, QPoint(imageScroll_->width(), 0)).x()) loadSource(path);
	else loadTemplate(path);
}
