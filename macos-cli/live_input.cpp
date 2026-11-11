// SPDX-License-Identifier: BSD-2-Clause
#include "mainwindow.h"
#include <QApplication>
#include <QCamera>
#include <QPermissions>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMediaCaptureSession>
#include <QMediaDevices>
#include <QMediaPlayer>
#include <QPushButton>
#include <QSaveFile>
#include <QSpinBox>
#include <QTimer>
#include <QVideoSink>

// The sink owns the latest frame. Sampling it on the GUI thread avoids a
// queued image per camera callback while the asynchronous matcher is busy.
class LiveInput : public QObject {
public:
	explicit LiveInput(QObject* parent) : QObject(parent) {}
	QVideoSink sink;
	QTimer timer;
	QVideoFrame previous;
	QMediaPlayer* player=nullptr;
	QCamera* camera=nullptr;
	QString name;
	int frames=0;
	bool cameraStarted=false;
	void stop() {
		timer.stop();
		if(player) player->stop();
		if(camera) camera->stop();
	}
};

void MainWindow::chooseLiveInput()
{
	if(live_) {stopLive();return;}
	if(busy_ || batchActive()) return;
	QDialog dialog(this);dialog.setWindowTitle(text("即時偵測來源","实时检测来源","Live source"));
	QFormLayout form(&dialog);QComboBox input;input.addItems({text("本機影片","本机视频","Local video"),text("本機相機","本机相机","Local camera")});
	form.addRow(text("來源","来源","Source"),&input);
	QSpinBox interval;interval.setRange(20,60000);interval.setSuffix(" ms");interval.setValue(liveInterval_);form.addRow(text("取樣間隔","采样间隔","Sample interval"),&interval);
	QDialogButtonBox buttons(QDialogButtonBox::Ok|QDialogButtonBox::Cancel);
	buttons.button(QDialogButtonBox::Ok)->setText(text("開啟","打开","Open"));buttons.button(QDialogButtonBox::Cancel)->setText(text("取消","取消","Cancel"));
	form.addRow(&buttons);connect(&buttons,&QDialogButtonBox::accepted,&dialog,&QDialog::accept);connect(&buttons,&QDialogButtonBox::rejected,&dialog,&QDialog::reject);
	if(dialog.exec()!=QDialog::Accepted) return;
	if(input.currentIndex()==0) {
		const auto path=QFileDialog::getOpenFileName(this,text("選擇本機影片","选择本机视频","Open local video"),{},"Video (*.mp4 *.mov *.m4v *.mkv *.avi *.webm);;All files (*)");
		if(!path.isEmpty()) startLiveVideo(path,interval.value());
	} else startLocalCamera(interval.value());
}

LiveInput* MainWindow::prepareLive(int interval)
{
	if(busy_ || batchActive() || live_ || interval<20 || interval>60000) return nullptr;
	const auto error=configError();
	if(!templ_.isNull() && !error.isEmpty()) {status_->setText(error);return nullptr;}
	liveInterval_=interval;live_=new LiveInput(this);
	auto* input=live_;input->timer.setInterval(interval);
	clearSource();liveInfo_->setText(text("等候第一張影格…","等候第一帧…","Waiting for the first frame…"));
	liveInfo_->show();setBusy(false);
	connect(&input->timer,&QTimer::timeout,this,[this,input] {
		if(live_!=input || busy_) return;
		const auto frame=input->sink.videoFrame();
		if(!frame.isValid() || frame==input->previous) return;
		input->previous=frame;
		auto rgb=frame.toImage();
		if(rgb.isNull()) {stopLive(text("無法解碼影格","无法解码帧","Could not decode the video frame"));return;}
		auto gray=recipeGray(rgb);
		if(gray.isNull()) {stopLive(text("無法準備影格","无法准备帧","Could not prepare the video frame"));return;}
		++input->frames;
		installSource(gray,std::move(rgb),input->name,true);
		frameInfo_={{"source",input->name},{"sample",input->frames},{"interval_ms",liveInterval_}};
		if(frame.startTime()>=0) frameInfo_["media_time_us"]=double(frame.startTime());
		if(input->player) frameInfo_["playback_position_ms"]=double(input->player->position());
		liveInfo_->setText(QString("%1 · %2 ms · #%3").arg(QFileInfo(input->name).fileName()).arg(liveInterval_).arg(input->frames));
		if(!templ_.isNull()) execute();
	});
	return input;
}

bool MainWindow::startLiveVideo(const QString& path,int interval)
{
	if(!QFileInfo(path).isFile()) {status_->setText(text("找不到本機影片","找不到本机视频","Local video not found"));return false;}
	auto* input=prepareLive(interval);if(!input) return false;
	input->name=QFileInfo(path).absoluteFilePath();input->player=new QMediaPlayer(input);
	input->player->setVideoSink(&input->sink);
	connect(input->player,&QMediaPlayer::errorOccurred,this,[this,input](QMediaPlayer::Error,const QString& why){if(live_==input) stopLive(why);});
	connect(input->player,&QMediaPlayer::mediaStatusChanged,this,[this,input](QMediaPlayer::MediaStatus state) {
		if(live_==input && state==QMediaPlayer::EndOfMedia)
			stopLive(input->frames?text("影片播放完畢","视频播放完毕","Video finished"):text("影片結束，未取到影格","视频结束，未取到帧","Video ended before a frame was sampled"));
	});
	input->player->setSource(QUrl::fromLocalFile(input->name));
	if(live_!=input) return false;
	input->timer.start();input->player->play();return live_==input;
}

void MainWindow::startLocalCamera(int interval)
{
	auto* input=prepareLive(interval);if(!input) return;
	auto open=[this,input] {
		if(live_!=input) return;
		const auto devices=QMediaDevices::videoInputs();
		if(devices.isEmpty()) {stopLive(text("找不到本機相機","找不到本机相机","No local camera found"));return;}
		int index=0;
		if(devices.size()>1) {
			QStringList names;
			for(int i=0;i<devices.size();++i) names.append(QString("%1 · %2").arg(i+1).arg(devices[i].description()));
			bool ok=false;const auto name=QInputDialog::getItem(this,text("選擇相機","选择相机","Choose camera"),{},names,0,false,&ok);
			if(live_!=input) return;
			if(!ok) {stopLive();return;}
			index=names.indexOf(name);
		}
		input->name=devices[index].description();input->camera=new QCamera(devices[index],input);
		auto* capture=new QMediaCaptureSession(input);capture->setCamera(input->camera);capture->setVideoSink(&input->sink);
		connect(input->camera,&QCamera::errorOccurred,this,[this,input](QCamera::Error,const QString& why){if(live_==input) stopLive(why);});
		connect(input->camera,&QCamera::activeChanged,this,[this,input](bool active){
			if(live_!=input) return;
			if(active) input->cameraStarted=true;
			else if(input->cameraStarted) stopLive(text("相機已中斷","相机已中断","Camera disconnected"));
		});
		input->timer.start();input->camera->start();
	};
#if defined(Q_OS_MACOS) && QT_CONFIG(permissions)
	const QCameraPermission permission;
	if(qApp->checkPermission(permission)!=Qt::PermissionStatus::Granted) {
		qApp->requestPermission(permission,input,[this,input,open](const QPermission& p){
			if(live_!=input) return;
			if(p.status()==Qt::PermissionStatus::Granted) open();
			else stopLive(text("請在系統設定允許 MatchTool 使用相機。","请在系统设置允许 MatchTool 使用相机。","Allow MatchTool camera access in System Settings."));
		});
		return;
	}
#endif
	open();
}

bool MainWindow::saveSampledFrame(const QString& path) const
{
	if(busy_ || frameInfo_.isEmpty() || sourceColour_.isNull()) return false;
	QSaveFile file(path);
	return file.open(QIODevice::WriteOnly) && sourceColour_.save(&file,"PNG") && file.commit();
}

void MainWindow::stopLive(const QString& reason)
{
	if(!live_) return;
	auto* old=live_;live_=nullptr;old->stop();old->deleteLater();
	liveInfo_->setText(reason.isEmpty()?text("已停止，保留最後取樣影格","已停止，保留最后采样帧","Stopped on the last sampled frame"):reason);
	setBusy(busy_);
}
