#include "mainwindow.h"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QImageReader>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QLabel>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QThreadPool>
#include <QtTest>
#include <cmath>

class GuiTest : public QObject {
	Q_OBJECT
private slots:
	void grayHighDpiPreview()
	{
		QTemporaryDir dir; QVERIFY(dir.isValid());
		QImage gray(31,23,QImage::Format_Grayscale8);
		for(int y=0;y<gray.height();++y) for(int x=0;x<gray.width();++x)
			gray.scanLine(y)[x]=uchar((17*x+31*y)%256);
		const auto path=dir.filePath("template@2x.png"); QVERIFY(gray.save(path));
		QImageReader reader(path); reader.setAutoTransform(false);
		const auto decoded=reader.read(); QCOMPARE(decoded.format(),QImage::Format_Grayscale8);
		QCOMPARE(decoded.devicePixelRatio(),qreal(2));
		// The original GUI allocated a new gray image, dropping DPR/color tags.
		const auto rgb=decoded.convertToFormat(QImage::Format_RGB888);
		QImage reference(rgb.size(),QImage::Format_Grayscale8);
		for(int y=0;y<rgb.height();++y) for(int x=0;x<rgb.width();++x) {
			const auto* p=rgb.constScanLine(y)+3*x;
			reference.scanLine(y)[x]=fpm::grayFromRgb(p[0],p[1],p[2]);
		}
		const auto expected=QPixmap::fromImage(reference).scaled(290,135,Qt::KeepAspectRatio,Qt::SmoothTransformation);
		MainWindow window; QVERIFY(window.loadTemplate(path));
		const auto preview=window.findChild<QLabel*>("templatePreview"); QVERIFY(preview);
		const auto actual=preview->pixmap(); QCOMPARE(actual.devicePixelRatio(),qreal(1));
		QCOMPARE(actual.size(),expected.size()); QCOMPARE(actual.toImage(),expected.toImage());
	}
	void rotorRotationAndOverlap()
	{
		MainWindow window;
		QVERIFY(window.loadSource(QStringLiteral(FPM_FIXTURE_DIR "/Src7.bmp")));
		QVERIFY(window.loadTemplate(QStringLiteral(FPM_FIXTURE_DIR "/Dst7.bmp")));
		auto* button=window.findChild<QPushButton*>("execute");
		auto* table=window.findChild<QTableWidget*>("results");
		auto* angle=window.findChild<QDoubleSpinBox*>("angle");
		auto* overlap=window.findChild<QDoubleSpinBox*>("overlap");
		auto* score=window.findChild<QDoubleSpinBox*>("score");
		auto* subpixel=window.findChild<QCheckBox*>("subpixel");
		QVERIFY(button && table && angle && overlap && score && subpixel);
		QSignalSpy finished(&window,&MainWindow::matchingFinished);
		// Same image, independent effects of angle, overlap suppression and score.
		struct Case { double angle, overlap, score; int count; bool subpixel=false; };
		for (const auto c : {Case{8,0,.5,1}, Case{180,0,.5,3}, Case{180,.1,.5,6}, Case{180,.1,.8,3},
			Case{180,.2,.5,6,true}, Case{180,.2,.8,3,true}}) {
			angle->setValue(c.angle); overlap->setValue(c.overlap); score->setValue(c.score);
			subpixel->setChecked(c.subpixel);
			button->click();
			QVERIFY(finished.wait(30000)); QVERIFY(finished.takeFirst()[0].toBool());
			QCOMPARE(table->rowCount(),c.count);
			if (c.count==6) {
				// Verify six distinct poses, not six duplicate boxes around one target.
				struct Pose { double x,y,angle; };
				const Pose expected[]={{1726,1045.5,0},{1769.4,2098.2,120.1},{2662.4,1537.8,-119.95},
					{2380.9,2071,-179.42},{1446.8,1588.1,60.88},{2329.8,1024,-58.94}};
				bool seen[6]{};
				for (int row=0;row<6;++row) {
					const double x=table->item(row,3)->text().toDouble(), y=table->item(row,4)->text().toDouble();
					const double a=table->item(row,2)->text().toDouble();
					int found=-1;
					for (int i=0;i<6;++i)
						if (!seen[i] && std::hypot(x-expected[i].x,y-expected[i].y)<2 &&
							std::abs(std::remainder(a-expected[i].angle,360.0))<1) { found=i; break; }
					QVERIFY2(found>=0,"unexpected or duplicate rotor pose"); seen[found]=true;
					const double value=table->item(row,1)->text().toDouble();
					QVERIFY(found<3 ? value>.98 : value>.63 && value<.67);
				}
			}
		}
	}

#ifdef FPM_GUI_RESEARCH
	void nativeScaleWorkflow()
	{
		QTemporaryDir dir;QVERIFY(dir.isValid());
		QImage templ(32,24,QImage::Format_Grayscale8),source(256,192,QImage::Format_Grayscale8);source.fill(20);
		for(int y=0;y<24;++y) for(int x=0;x<32;++x) templ.scanLine(y)[x]=uchar((x*23+y*47+x*y*11)%256);
		auto small=fpm::research::scaledTemplate({templ.constBits(),32,24,size_t(templ.bytesPerLine())},.75);
		for(int y=0;y<24;++y) for(int x=0;x<32;++x) source.scanLine(30+y)[40+x]=templ.scanLine(y)[x];
		for(int y=0;y<small.height();++y) for(int x=0;x<small.width();++x) source.scanLine(110+y)[160+x]=small.row(y)[x];
		const auto sp=dir.filePath("source.png"),tp=dir.filePath("template.png"),out=dir.filePath("out.json");QVERIFY(source.save(sp));QVERIFY(templ.save(tp));
		MainWindow window;window.show();QVERIFY(window.loadSource(sp));QVERIFY(window.loadTemplate(tp));
		auto engine=window.findChild<QComboBox*>("engine");auto mode=window.findChild<QComboBox*>("nccMode");
		auto lo=window.findChild<QDoubleSpinBox*>("scaleMin");auto hi=window.findChild<QDoubleSpinBox*>("scaleMax");
		auto overlap=window.findChild<QCheckBox*>("coarseOverlap");auto refine=window.findChild<QCheckBox*>("continuousRefinement");
		auto button=window.findChild<QPushButton*>("execute");auto table=window.findChild<QTableWidget*>("results");
		QSignalSpy finished(&window,&MainWindow::matchingFinished);
		window.findChild<QDoubleSpinBox*>("score")->setValue(.9);window.findChild<QDoubleSpinBox*>("overlap")->setValue(.2);
		window.findChild<QSpinBox*>("minArea")->setValue(10000);window.findChild<QDoubleSpinBox*>("angle")->setValue(180);
		lo->setValue(.75);hi->setValue(1);refine->setChecked(false);
		auto run=[&]() {button->click();return finished.wait(30000) && finished.takeFirst()[0].toBool();};
		for(int route=1;route<engine->count();++route) {
			engine->setCurrentIndex(route);mode->setCurrentIndex(1);overlap->setChecked(true);
			QVERIFY(run());QCOMPARE(table->rowCount(),2);QVERIFY(window.exportResults(out));
			QFile file(out);QVERIFY(file.open(QIODevice::ReadOnly));auto data=QJsonDocument::fromJson(file.readAll()).object();
			QCOMPARE(data["schema_version"].toInt(),2);auto config=data["run"].toObject();QCOMPARE(config["ncc_mode"].toInt(),route==2?2:0);
			QCOMPARE(config["coarse_overlap"].toBool(),route==2);QCOMPARE(config["frame_generation"].toInt(),1);
			bool one=false,scaled=false;for(auto m:data["matches"].toArray()) {double scale=m.toObject()["scale"].toDouble();one|=scale==1.;scaled|=scale==.75;}QVERIFY(one && scaled);
			QVERIFY(run());QVERIFY(window.exportResults(out));file.close();QVERIFY(file.open(QIODevice::ReadOnly));
			QCOMPARE(QJsonDocument::fromJson(file.readAll()).object()["run"].toObject()["frame_generation"].toInt(),2);
			if(route==engine->count()-1 && !qEnvironmentVariable("FPM_GUI_SCREENSHOT").isEmpty()) { QCoreApplication::processEvents();QVERIFY(window.grab().save(qEnvironmentVariable("FPM_GUI_SCREENSHOT"))); }
			refine->setChecked(true);QVERIFY(run());QVERIFY(table->rowCount()>=1);QVERIFY(window.exportResults(out));
			file.close();QVERIFY(file.open(QIODevice::ReadOnly));QCOMPARE(QJsonDocument::fromJson(file.readAll()).object()["run"].toObject()["refinement_backend"].toInt(),route==2?1:0);
			refine->setChecked(false);
		}
		// Invalid scale is a recoverable immediate failure, and clears old output.
		lo->setValue(1.5);hi->setValue(1);button->click();QVERIFY(!finished.isEmpty());QVERIFY(!finished.takeFirst()[0].toBool());QCOMPARE(table->rowCount(),0);QVERIFY(!window.exportResults(out));
		lo->setValue(.75);QVERIFY(run());
		// Export provenance belongs to the completed job even after controls change.
		hi->setValue(1.25);QVERIFY(window.exportResults(out));
		QFile snapshot(out);QVERIFY(snapshot.open(QIODevice::ReadOnly));QCOMPARE(QJsonDocument::fromJson(snapshot.readAll()).object()["run"].toObject()["proposal_scales"].toArray().last().toDouble(),1.);hi->setValue(1);
		// Changing the image invalidates the source snapshot, even at same size.
		source.fill(20);QVERIFY(source.save(sp));QVERIFY(window.loadSource(sp));QVERIFY(run());QCOMPARE(table->rowCount(),0);QVERIFY(window.exportResults(out));
		QFile empty(out);QVERIFY(empty.open(QIODevice::ReadOnly));auto e=QJsonDocument::fromJson(empty.readAll()).object();QCOMPARE(e["run"].toObject()["frame_generation"].toInt(),1);QCOMPARE(e["matches"].toArray().size(),0);
		// Closing a busy window must leave the worker with owned image/session snapshots.
		auto closing=std::make_unique<MainWindow>();QVERIFY(closing->loadSource(sp));QVERIFY(closing->loadTemplate(tp));
		closing->findChild<QComboBox*>("engine")->setCurrentIndex(engine->count()-1);
		closing->findChild<QPushButton*>("execute")->click();closing.reset();QVERIFY(QThreadPool::globalInstance()->waitForDone(30000));
	}
#endif
	void workflow()
	{
		QTemporaryDir dir; QVERIFY(dir.isValid());
		QImage source(256,200,QImage::Format_Grayscale8); source.fill(0);
		QImage templ(31,23,QImage::Format_Grayscale8);
		for(int y=0;y<23;++y) for(int x=0;x<31;++x) { const auto p=uchar(((x/3)*17+(y/3)*31+(x/3)*(y/3)*7)%256); templ.scanLine(y)[x]=p; source.scanLine(60+y)[70+x]=p; }
		const auto src=dir.filePath("來源.png"), dst=dir.filePath("模板.png");
		QVERIFY(source.save(src)); QVERIFY(templ.save(dst));
		MainWindow window; window.show();
		auto* button=window.findChild<QPushButton*>("execute"); QVERIFY(button); QVERIFY(!button->isEnabled());
		QVERIFY(window.loadSource(src)); QVERIFY(window.loadTemplate(dst));
		window.findChild<QDoubleSpinBox*>("score")->setValue(.95);
		window.findChild<QSpinBox*>("targetCount")->setValue(1);
		QSignalSpy finished(&window,&MainWindow::matchingFinished);
		bool eventLoopRan=false;
		QTest::mouseClick(button,Qt::LeftButton);
		QVERIFY(!button->isEnabled()); QVERIFY(!window.loadSource(src));
		QTimer::singleShot(0,&window,[&] { eventLoopRan=true; });
		QVERIFY(finished.wait(10000)); QVERIFY(finished.takeFirst()[0].toBool()); QVERIFY(eventLoopRan);
		auto* table=window.findChild<QTableWidget*>("results"); QCOMPARE(table->rowCount(),1);
		QCOMPARE(table->item(0,3)->text().toDouble(),85.5); QCOMPARE(table->item(0,4)->text().toDouble(),71.5);
		const auto output=dir.filePath("results.json"); QVERIFY(window.exportResults(output));
		QFile file(output); QVERIFY(file.open(QIODevice::ReadOnly));
		QCOMPARE(QJsonDocument::fromJson(file.readAll()).object()["matches"].toArray().size(),1);
		// Relearn after a depth change; invalidate stale results on image changes.
		window.findChild<QSpinBox*>("minArea")->setValue(10000);
		window.findChild<QCheckBox*>("subpixel")->setChecked(true);
		QTest::mouseClick(button,Qt::LeftButton); QVERIFY(finished.wait(10000)); QVERIFY(finished.takeFirst()[0].toBool()); QCOMPARE(table->rowCount(),1);
		QVERIFY(window.loadTemplate(dst)); QCOMPARE(table->rowCount(),0);
		// Invalid custom range produces a recoverable error instead of a crash.
		window.findChild<QCheckBox*>("useRanges")->setChecked(true);
		window.findChild<QDoubleSpinBox*>("range0")->setValue(70);
		window.findChild<QDoubleSpinBox*>("range1")->setValue(40);
		QTest::mouseClick(button,Qt::LeftButton); QVERIFY(finished.wait(10000)); QVERIFY(!finished.takeFirst()[0].toBool());
		QVERIFY(button->isEnabled()); QVERIFY(!window.findChild<QLabel*>("status")->text().isEmpty());
	}
};
QTEST_MAIN(GuiTest)
#include "gui_test.moc"
