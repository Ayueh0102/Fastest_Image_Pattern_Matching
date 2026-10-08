#include "mainwindow.h"
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFile>
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
#include <QtTest>
#include <cmath>

class GuiTest : public QObject {
	Q_OBJECT
private slots:
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
