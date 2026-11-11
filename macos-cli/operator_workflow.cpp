// SPDX-License-Identifier: BSD-2-Clause
#include "mainwindow.h"
#include <QBuffer>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDesktopServices>
#include <QUrl>
#include <QCloseEvent>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QJsonDocument>
#include <QJsonArray>
#include <QLabel>
#include <QPushButton>
#include <QSaveFile>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTimer>
#include <QVBoxLayout>
#include <cmath>

namespace {
bool writeJson(const QString& path,const QByteArray& bytes)
{
	QSaveFile file(path);
	return file.open(QIODevice::WriteOnly) && file.write(bytes)==bytes.size() && file.commit();
}
QJsonObject pngAsset(const QImage& image,const QString& name)
{
	QBuffer data;
	const auto pixels=image.format()==QImage::Format_Grayscale8?image:image.convertToFormat(QImage::Format_RGB888);
	if(!data.open(QIODevice::WriteOnly) || !pixels.save(&data,"PNG")) return {};
	return {{"name",QFileInfo(name).fileName()},{"png",QString::fromLatin1(data.data().toBase64())}};
}
QImage decodeAsset(const QJsonObject& asset)
{
	if(asset["name"].toString().isEmpty()) return {};
	return QImage::fromData(QByteArray::fromBase64(asset["png"].toString().toLatin1(),QByteArray::AbortOnBase64DecodingErrors),"PNG");
}
QString digest(const QByteArray& bytes)
{
	return QString::fromLatin1(QCryptographicHash::hash(bytes,QCryptographicHash::Sha256).toHex());
}
}

void MainWindow::buildOperatorPanel(QVBoxLayout* side,QVBoxLayout* controls)
{
	recipeInfo_=new QLabel;recipeInfo_->setWordWrap(true);recipeInfo_->setTextFormat(Qt::PlainText);recipeInfo_->setObjectName("recipeInfo");
	auto font=recipeInfo_->font();font.setBold(true);recipeInfo_->setFont(font);
	side->addWidget(recipeInfo_);
	openRecipe_=new QPushButton;openRecipe_->setObjectName("openRecipe");side->addWidget(openRecipe_);
	editSettings_=new QPushButton;editSettings_->setObjectName("editSettings");editSettings_->setCheckable(true);
	saveRecipe_=new QPushButton;saveRecipe_->setObjectName("saveRecipe");
	batchButton_=new QPushButton;batchButton_->setObjectName("inspectBatch");
	batchInfo_=new QLabel;batchInfo_->setWordWrap(true);batchInfo_->setTextFormat(Qt::PlainText);batchInfo_->setObjectName("batchInfo");
	inspection_=new QLabel;inspection_->setWordWrap(true);inspection_->setObjectName("inspection");
	font.setPointSizeF(font.pointSizeF()+2);inspection_->setFont(font);
	controls->addWidget(inspection_);controls->addWidget(batchInfo_);
	auto* actions=new QHBoxLayout;controls->addLayout(actions);
	resumeBatch_=new QPushButton;resumeBatch_->setObjectName("continueBatch");actions->addWidget(resumeBatch_);
	stopBatch_=new QPushButton;stopBatch_->setObjectName("stopBatch");actions->addWidget(stopBatch_);
	openBatchFolder_=new QPushButton;openBatchFolder_->setObjectName("openBatchFolder");controls->addWidget(openBatchFolder_);
	connect(openBatchFolder_,&QPushButton::clicked,this,[this]{
		if(!QDesktopServices::openUrl(QUrl::fromLocalFile(batchDir_))) status_->setText(QDir::toNativeSeparators(batchDir_));
	});
	batchTimer_=new QTimer(this);batchTimer_->setSingleShot(true);
	connect(batchTimer_,&QTimer::timeout,this,&MainWindow::runBatchImage);
	connect(this,&MainWindow::matchingFinished,this,[this](bool ok){if(batchActive()) recordBatchImage(ok);});
	connect(openRecipe_,&QPushButton::clicked,this,[this]{
		const auto path=QFileDialog::getOpenFileName(this,openRecipe_->text(),{},"MatchTool (*.fpmrecipe)");
		if(!path.isEmpty()) loadRecipe(path);
	});
	connect(saveRecipe_,&QPushButton::clicked,this,[this]{
		auto path=QFileDialog::getSaveFileName(this,saveRecipe_->text(),recipePath_,"MatchTool (*.fpmrecipe)");
		if(path.isEmpty()) return;
		if(!path.endsWith(".fpmrecipe",Qt::CaseInsensitive)) path+=".fpmrecipe";
		saveRecipe(path);
	});
	connect(batchButton_,&QPushButton::clicked,this,&MainWindow::pickBatch);
	connect(resumeBatch_,&QPushButton::clicked,this,[this] {
		if(batchSummaryFailed_ && !batchActive()) {saveBatchSummary();return;}
		if(!batchActive() || busy_) return;
		if(batchWriteFailed_) {recordBatchImage(hasCompleted_);return;}
		batchPaused_=false;++batchIndex_;runBatchImage();
	});
	connect(stopBatch_,&QPushButton::clicked,this,[this] {
		stopRequested_=true;
		if(busy_) refreshBatch(); else finishBatch();
	});
}

QList<QWidget*> MainWindow::recipeFields() const
{
	QList<QWidget*> fields{engine_,searchColour_,expected_,targets_,score_,overlap_,angle_,minArea_,useRanges_,
		ranges_[0],ranges_[1],ranges_[2],ranges_[3],subpixel_,invert_,simd_,fast_,
		scaleMin_,scaleMax_,scaleStep_,nccMode_,coarseOverlap_,continuous_};
#ifdef FPM_GUI_RESEARCH
	fields.append({shortSide_,proposalScore_,verifySmall_,dedup_,checkColour_,colourThreshold_});
#endif
	return fields;
}

QJsonObject MainWindow::recipeSettings() const
{
	QJsonObject settings;
	for(auto* field:recipeFields()) {
		const auto key=field->objectName();
		if(auto* v=qobject_cast<QSpinBox*>(field)) settings[key]=v->value();
		else if(auto* v=qobject_cast<QDoubleSpinBox*>(field)) settings[key]=v->value();
		else if(auto* v=qobject_cast<QComboBox*>(field)) settings[key]=v->currentData().toInt();
		else if(auto* v=qobject_cast<QCheckBox*>(field)) settings[key]=v->isChecked();
	}
	return settings;
}

void MainWindow::refreshRecipe()
{
	QString label;
	if(recipePath_.isEmpty()) label=templ_.isNull()?text("尚未選擇料號","尚未选择料号","No part selected"):
		text("使用單獨模板","使用单独模板","Using a standalone template");
	else {
		label=text("料號配方 · ","料号配方 · ","Part recipe · ")+QFileInfo(recipePath_).completeBaseName();
		if(assetsChanged_ || savedSettings_!=recipeSettings()) label+=text("（已修改）","（已修改）"," (modified)");
	}
	recipeInfo_->setText(label);recipeInfo_->setToolTip(recipePath_);
}

QString MainWindow::configError() const
{
	if(expected_->value()>0 && targets_->value()<=expected_->value())
		return text("目標數量上限必須大於預期件數，才能檢出多件。","目标数量上限必须大于预期件数，才能检出多件。","Set max targets above the expected count to detect extra parts.");
	return {};
}

bool MainWindow::saveRecipe(const QString& path)
{
	if(live_ || busy_ || batchActive() || templ_.isNull()) return false;
	const auto settings=recipeSettings();
	const auto error=configError();
	if(!error.isEmpty()) {status_->setText(error);return false;}
	if((engine_->currentData().toInt()!=0 && (scaleMin_->value()>scaleMax_->value() ||
		(scaleMax_->value()-scaleMin_->value())/scaleStep_->value()>63+1e-9)) ||
		(useRanges_->isChecked() && (ranges_[0]->value()>ranges_[1]->value() || ranges_[2]->value()>ranges_[3]->value()))) {
		status_->setText(text("請檢查尺度或角度範圍。","请检查尺度或角度范围。","Check the scale or angle range."));return false;
	}
	const auto schema=searchColour_->isChecked()?"fpm.part-recipe.v2":"fpm.part-recipe.v1";
	QJsonObject recipe{{"schema",schema},{"settings",settings}};
#ifdef FPM_GUI_RESEARCH
	recipe["research"]=true;
	recipe["template"]=pngAsset(templateColour_,templatePath_);
	if(!inspectionMask_.colour.isNull()) {
		QImage rgb(templ_.size(),QImage::Format_RGB888);
		if(rgb.isNull()) return false;
		for(int y=0;y<rgb.height();++y) {
			auto* row=rgb.scanLine(y);
			for(int x=0;x<rgb.width();++x) {
				row[3*x]=inspectionMask_.colour.constScanLine(y)[x];
				row[3*x+1]=inspectionMask_.reference.constScanLine(y)[x];
				row[3*x+2]=inspectionMask_.dark.isNull()?0:inspectionMask_.dark.constScanLine(y)[x];
			}
		}
		recipe["mask"]=pngAsset(rgb,inspectionMask_.path);
	}
#else
	recipe["research"]=false;recipe["template"]=pngAsset(templateColour_,templatePath_);
#endif
	const auto bytes=QJsonDocument(recipe).toJson(QJsonDocument::Compact);
	if(recipe["template"].toObject().isEmpty() || (recipe.contains("mask") && recipe["mask"].toObject().isEmpty()) || !writeJson(path,bytes)) {
		status_->setText(text("無法儲存料號配方。","无法保存料号配方。","Could not save the part recipe."));return false;
	}
	recipePath_=path;recipeDigest_=digest(bytes);savedSettings_=settings;assetsChanged_=false;
	refreshRecipe();status_->setText(text("已儲存料號配方。","已保存料号配方。","Part recipe saved."));return true;
}

bool MainWindow::loadRecipe(const QString& path)
{
	if(live_ || busy_ || batchActive()) return false;
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly)) {status_->setText(file.errorString());return false;}
	const auto bytes=file.readAll();
	const auto recipe=QJsonDocument::fromJson(bytes).object();
	auto invalid=[this] {
		status_->setText(text("無法套用配方，請確認檔案、參數與本機後端。","无法套用配方，请确认文件、参数与本机后端。","Cannot apply recipe. Check its file, settings and available engine."));return false;
	};
	const bool legacy=recipe["schema"]=="fpm.part-recipe.v1";
	if((!legacy && recipe["schema"]!="fpm.part-recipe.v2") || !recipe["settings"].isObject() || !recipe["research"].isBool()) return invalid();
	auto settings=recipe["settings"].toObject();
	if(legacy && !settings.contains("searchColour")) settings["searchColour"]=false; // v1 grayscale recipes
	const auto fields=recipeFields();
	for(auto* field:fields) {
		const auto key=field->objectName();
#ifdef FPM_GUI_RESEARCH
		if(!recipe["research"].toBool() && !settings.contains(key) &&
			(field==shortSide_ || field==proposalScore_ || field==verifySmall_ || field==dedup_ || field==checkColour_ || field==colourThreshold_))
			settings[key]=defaultSettings_.value(key);
#endif
		const auto value=settings.value(key);
		const double n=value.toDouble();
		if(auto* v=qobject_cast<QSpinBox*>(field)) {
			if(!value.isDouble() || n!=std::floor(n) || n<v->minimum() || n>v->maximum()) return invalid();
		} else if(auto* v=qobject_cast<QDoubleSpinBox*>(field)) {
			const double unit=std::pow(10.,v->decimals());
			if(!value.isDouble() || n<v->minimum() || n>v->maximum() || std::abs(n*unit-std::round(n*unit))>1.e-7) return invalid();
		} else if(auto* v=qobject_cast<QComboBox*>(field)) {
			if(!value.isDouble() || n!=value.toInt(-1) || v->findData(value.toInt(-1))<0) return invalid();
		} else if(!value.isBool()) return invalid();
	}
	const auto badRange=[&](const char* a,const char* b){return settings[a].toDouble()>settings[b].toDouble();};
	if(settings["expectedCount"].toInt()>0 && settings["expectedCount"].toInt()>=settings["targetCount"].toInt()) return invalid();
	if(settings["engine"].toInt()!=0 && (badRange("scaleMin","scaleMax") ||
		(settings["scaleMax"].toDouble()-settings["scaleMin"].toDouble())/settings["scaleStep"].toDouble()>63+1.e-9)) return invalid();
	if(settings["useRanges"].toBool() && (badRange("range0","range1") || badRange("range2","range3"))) return invalid();
	const auto asset=recipe["template"].toObject();const auto original=decodeAsset(asset);
	const auto gray=recipeGray(original);
	if(gray.isNull()) return invalid();
#ifdef FPM_GUI_RESEARCH
	InspectionMask mask;
	if(recipe.contains("mask")) {
		const auto entry=recipe["mask"].toObject();
		try {mask=InspectionMask::fromImage(decodeAsset(entry),original,entry["name"].toString());}
		catch(const std::exception& e) {status_->setText(QString::fromUtf8(e.what()));return false;}
	}
	if(settings["checkColour"].toBool() && mask.colour.isNull()) return invalid();
#else
	if(recipe.contains("mask") || settings["checkColour"].toBool()) return invalid();
#endif
	// All assets and settings have been checked before replacing the current job.
	clearSource();installTemplate(gray,original,asset["name"].toString());
#ifdef FPM_GUI_RESEARCH
	inspectionMask_=std::move(mask);checkColour_->setEnabled(!inspectionMask_.colour.isNull());
#endif
	for(auto* field:fields) {
		const auto value=settings.value(field->objectName());
		if(auto* v=qobject_cast<QSpinBox*>(field)) v->setValue(value.toInt());
		else if(auto* v=qobject_cast<QDoubleSpinBox*>(field)) v->setValue(value.toDouble());
		else if(auto* v=qobject_cast<QComboBox*>(field)) v->setCurrentIndex(v->findData(value.toInt()));
		else if(auto* v=qobject_cast<QCheckBox*>(field)) v->setChecked(value.toBool());
	}
	recipePath_=path;recipeDigest_=digest(bytes);assetsChanged_=false;savedSettings_=recipeSettings();
	editSettings_->setChecked(false);setBusy(false);loadSource_->setFocus(Qt::OtherFocusReason);status_->clear();return true;
}

void MainWindow::clearSource()
{
	source_={};frameInfo_={};sourcePath_.clear();sourceInfo_->clear();sourceInfo_->setToolTip({});
sourceColour_={};
	clearResults();view_->setImage({},{});setBusy(false);
}

void MainWindow::refreshInspection()
{
	QString label;
	if(busy_) label=text("檢查中…","检查中…","Inspecting…");
	else if(inspectionFailed_) label=text("檢查未完成","检查未完成","Inspection incomplete");
	else if(!hasCompleted_) {
		if(templ_.isNull()) label=text("開啟配方，或在料號設定載入模板","打开配方，或在料号设置中载入模板","Open a recipe or load a template in part settings");
		else if(source_.isNull()) label=text("等待待檢影像","等待待检图像","Waiting for an image");
		else label=text("影像就緒，可執行檢查","图像就绪，可执行检查","Image ready to inspect");
	} else {
		const auto s=provenance_["inspection"].toObject();
		const int found=s["found"].toInt(), expected=s["expected"].toInt();
		label=s["needs_review"].toBool()?text("需複核","需复核","Review required"):
			(expected?text("件數符合","件数符合","Count matches"):text("定位完成","定位完成","Located"));
		label+=text(" · 找到 %1 件"," · 找到 %1 件"," · %1 found").arg(found);
		if(expected) label+=text("／預期 %1 件","／预期 %1 件"," / %1 expected").arg(expected);
		if(s["colour_outside"].toInt()) label+=text(" · %1 件色度超限"," · %1 件色度超限"," · %1 outside colour limit").arg(s["colour_outside"].toInt());
		if(s["colour_review"].toInt()) label+=text(" · %1 件無法驗色"," · %1 件无法验色"," · %1 colour inconclusive").arg(s["colour_review"].toInt());
	}
	const bool attention=inspectionFailed_ || (hasCompleted_ && provenance_["inspection"].toObject()["needs_review"].toBool());
	inspection_->setStyleSheet(attention?"border-left: 4px solid #b77913; padding: 5px 8px;":"padding: 5px 0px;");
	inspection_->setText(label);
}

void MainWindow::refreshBatch()
{
	openBatchFolder_->setText(text("開啟本批結果資料夾","打开本批结果文件夹","Open batch results folder"));
	openBatchFolder_->setVisible(!batchDir_.isEmpty());batchInfo_->setVisible(!batchDir_.isEmpty());batchInfo_->setToolTip(batchDir_);
	resumeBatch_->setVisible((batchActive() && (batchPaused_ || batchWriteFailed_)) || batchSummaryFailed_);
	resumeBatch_->setText((batchWriteFailed_ || batchSummaryFailed_)?text("重試儲存","重试保存","Retry saving"):text("繼續下一張","继续下一张","Continue to next image"));
	stopBatch_->setVisible(batchActive());stopBatch_->setEnabled(!stopRequested_);
	if(batchActive()) {
		QString msg=text("本批 %1／%2 · %3","本批 %1／%2 · %3","Batch %1/%2 · %3")
			.arg(batchIndex_+1).arg(batchPaths_.size()).arg(QFileInfo(batchPaths_.at(batchIndex_)).fileName());
		if(stopRequested_) msg+=text(" · 本張完成後停止"," · 本张完成后停止"," · stopping after this image");
		if(batchWriteFailed_) msg+=text(" · 結果尚未存檔"," · 结果尚未存档"," · result not saved");
		batchInfo_->setText(msg);
	} else if(!batchSummary_.isEmpty()) {
		const auto title=batchSummary_["stopped"].toBool()?text("本批已停止","本批已停止","Batch stopped"):text("本批已完成","本批已完成","Batch finished");
		QString msg=title+text(" · 已存 %1／%2 張 · %3 張需複核"," · 已存 %1／%2 张 · %3 张需复核"," · %1/%2 saved · %3 require review")
			.arg(batchSaved_).arg(batchSummary_["total"].toInt()).arg(reviewIndices_.size());
		if(!failedIndices_.isEmpty()) msg+=text("（含 %1 張檢查失敗）","（含 %1 张检查失败）"," (%1 inspections failed)").arg(failedIndices_.size());
		if(batchSummaryFailed_) msg+=text(" · 彙總檔尚未儲存"," · 汇总文件尚未保存"," · summary not saved");
		batchInfo_->setText(msg);
	}
}

void MainWindow::pickBatch()
{
	const auto paths=QFileDialog::getOpenFileNames(this,batchButton_->text(),{},"Images (*.bmp *.png *.jpg *.jpeg *.pgm);;All files (*)");
	if(!paths.isEmpty()) chooseBatchOutput(paths);
}

void MainWindow::chooseBatchOutput(const QStringList& paths)
{
	const auto dir=QFileDialog::getExistingDirectory(this,text("選擇本批結果的儲存位置","选择本批结果的保存位置","Choose where to save batch results"));
	if(!dir.isEmpty()) startBatch(paths,dir);
}

bool MainWindow::startBatch(const QStringList& paths,const QString& outputRoot)
{
	if(live_ || busy_ || batchActive() || paths.isEmpty() || templ_.isNull()) return false;
	const auto error=configError();
	if(!error.isEmpty()) {status_->setText(error);return false;}
	const auto prefix=QString("MatchTool-%1-XXXXXX").arg(QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss"));
	QTemporaryDir dir(QDir(outputRoot).filePath(prefix));
	QJsonArray inputs;QStringList files;
	for(const auto& path:paths) {files.append(QFileInfo(path).absoluteFilePath());inputs.append(files.last());}
	const QJsonObject manifest{{"schema","fpm.batch.v1"},{"sources",inputs},{"recipe_sha256",recipeDigest_},
		{"recipe",QFileInfo(recipePath_).completeBaseName()},{"recipe_modified",!recipePath_.isEmpty() && (assetsChanged_ || savedSettings_!=recipeSettings())},{"settings",recipeSettings()},{"template",templatePath_}};
	if(outputRoot.isEmpty() || !dir.isValid() || !writeJson(dir.filePath("batch.json"),QJsonDocument(manifest).toJson())) {
		status_->setText(text("無法建立本批結果目錄。","无法创建本批结果目录。","Cannot create the batch output folder."));return false;
	}
	dir.setAutoRemove(false);batchDir_=dir.path();batchPaths_=files;batchIndex_=0;batchSaved_=0;reviewIndices_={};failedIndices_={};batchSummary_={};batchSummaryFailed_=false;
	batchPaused_=batchWriteFailed_=stopRequested_=false;editSettings_->setChecked(false);
	setBusy(false);runBatchImage();return true;
}

void MainWindow::runBatchImage()
{
	if(!batchActive() || busy_) return;
	batchError_.clear();
	if(!readSource(batchPaths_.at(batchIndex_))) {recordBatchImage(false);return;}
	execute();
}

void MainWindow::recordBatchImage(bool success)
{
	if(!batchActive()) return;
	if(!success && !batchWriteFailed_) batchError_=status_->text();
	const QString file=QDir(batchDir_).filePath(QString("%1.json").arg(batchIndex_+1,5,10,QChar('0')));
	const QJsonObject failure{{"schema","fpm.inspection-error.v1"},{"needs_review",true},
		{"source",batchPaths_.at(batchIndex_)},{"recipe_sha256",recipeDigest_},{"recorded_utc",QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},{"error",batchError_}};
	const bool saved=success?exportResults(file):writeJson(file,QJsonDocument(failure).toJson());
	batchWriteFailed_=!saved;
	if(!saved) {
		closeAfterBatch_=false;stopRequested_=false;batchPaused_=true;status_->setText(text("寫入失敗，可重試儲存。","写入失败，可重试保存。","Write failed. Retry saving.")+" "+file);
		refreshBatch();return;
	}
	++batchSaved_;
	batchPaused_=!success || provenance_["inspection"].toObject()["needs_review"].toBool();
	if(batchPaused_) reviewIndices_.append(batchIndex_+1);
	if(!success) failedIndices_.append(batchIndex_+1);
	if(stopRequested_ || batchIndex_+1==batchPaths_.size()) {finishBatch();return;}
	if(!batchPaused_) {++batchIndex_;batchTimer_->start(0);}
	refreshBatch();
	if(batchPaused_) resumeBatch_->setFocus(Qt::OtherFocusReason);
}

void MainWindow::finishBatch()
{
	batchTimer_->stop();
	batchSummary_={{"schema","fpm.batch-summary.v1"},{"stopped",stopRequested_},{"total",int(batchPaths_.size())},
		{"saved",batchSaved_},{"review_indices",reviewIndices_},{"error_indices",failedIndices_},
		{"finished_utc",QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)}};
	batchIndex_=-1;batchPaths_.clear();batchPaused_=stopRequested_=batchWriteFailed_=false;
	saveBatchSummary();
	setBusy(false);
	if(closeAfterBatch_) {closeAfterBatch_=false;QTimer::singleShot(0,this,&QWidget::close);}
}

void MainWindow::saveBatchSummary()
{
	batchSummaryFailed_=!writeJson(QDir(batchDir_).filePath("summary.json"),QJsonDocument(batchSummary_).toJson());
	if(batchSummaryFailed_) closeAfterBatch_=false;
	refreshBatch();
}

void MainWindow::closeEvent(QCloseEvent* event)
{
#ifdef FPM_GUI_MULTIMEDIA
	if(live_) stopLive();
#endif
	if(batchActive() && busy_) {
		stopRequested_=true;closeAfterBatch_=true;refreshBatch();event->ignore();return;
	}
	if(batchActive()) {stopRequested_=true;finishBatch();}
	QMainWindow::closeEvent(event);
}
