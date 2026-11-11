#include "mainwindow.h"
#include "template_editor.h"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QPainter>
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
	void cropRotateSplit()
	{
		QTemporaryDir dir;QVERIFY(dir.isValid());
		QImage image(101,77,QImage::Format_RGB32);
		for(int y=0;y<image.height();++y) for(int x=0;x<image.width();++x)
			image.setPixel(x,y,qRgb(x,y,(x*11+y*23)%256));
		TemplateEditor e(image,2);
		auto set=[&](const char* key,int value){e.findChild<QSpinBox*>(key)->setValue(value);};
		set("cropWidth",7);set("cropHeight",5);set("cropX",20);set("cropY",18);
		set("tileRows",2);set("tileCols",3);set("activeTile",6);
		QCOMPARE(e.selectedTile(),image.copy(24,20,3,3));
		const QDir tiles(e.exportTiles(dir.path()));
		QCOMPARE(tiles.entryList({"*.png"}).size(),6);
		QCOMPARE(QImage(tiles.filePath("row1-col2.png")).convertToFormat(image.format()),image.copy(22,18,2,2));
		set("tileRows",1);set("tileCols",1);set("cropWidth",6);set("cropHeight",4);
		e.findChild<QDoubleSpinBox*>("cropAngle")->setValue(90);
		const auto turned=e.selectedTile();QCOMPARE(turned.size(),QSize(6,4));
		QCOMPARE(turned.pixel(0,0),image.pixel(24,17));QCOMPARE(turned.pixel(5,3),image.pixel(21,22));
		set("cropX",100);QVERIFY(!e.findChild<QPushButton*>("useCrop")->isEnabled());QVERIFY(e.selectedTile().isNull());
		e.show();QCoreApplication::processEvents();
		auto* canvas=e.findChild<QWidget*>("cropCanvas");const auto mid=canvas->rect().center();
		QTest::mousePress(canvas,Qt::LeftButton,{},mid+QPoint(50,30));
		QTest::mouseRelease(canvas,Qt::LeftButton,{},mid-QPoint(50,30));
		QVERIFY(e.findChild<QPushButton*>("useCrop")->isEnabled());
		QVERIFY(e.selectedTile().width()>5);QCOMPARE(e.findChild<QDoubleSpinBox*>("cropAngle")->value(),0.);

		MainWindow w;const auto path=dir.filePath("frame.png");QVERIFY(image.save(path));QVERIFY(w.loadSource(path));
		auto* create=w.findChild<QPushButton*>("makeTemplate");
		QTimer::singleShot(0,&w,[&]{QApplication::activeModalWidget()->close();});create->click();
		QVERIFY(!w.saveRecipe(dir.filePath("empty.fpmrecipe")));
		QTimer::singleShot(0,&w,[&]{
			auto* modal=QApplication::activeModalWidget();
			modal->findChild<QSpinBox*>("cropWidth")->setValue(17);modal->findChild<QSpinBox*>("cropHeight")->setValue(18);
			modal->findChild<QSpinBox*>("cropX")->setValue(14);modal->findChild<QSpinBox*>("cropY")->setValue(20);
			modal->findChild<QPushButton*>("useCrop")->click();
		});
		create->click();const auto recipe=dir.filePath("cropped.fpmrecipe");QVERIFY(w.saveRecipe(recipe));
		QFile f(recipe);QVERIFY(f.open(QIODevice::ReadOnly));
		const auto png=QJsonDocument::fromJson(f.readAll()).object()["template"].toObject()["png"].toString().toLatin1();
		QCOMPARE(QImage::fromData(QByteArray::fromBase64(png)).convertToFormat(image.format()),image.copy(14,20,17,18));
	}
#ifdef FPM_GUI_MULTIMEDIA
	void localLiveVideo()
	{
		QTemporaryDir dir;QVERIFY(dir.isValid());MainWindow w;
		const auto movie=QStringLiteral(FPM_FIXTURE_DIR "/../cpp-core/tests/data/live-mark.mp4");
		QVERIFY(!w.startLiveVideo(dir.filePath("missing.mp4"),40));
		QVERIFY(!w.startLiveVideo(movie,0));
		QImage t(32,24,QImage::Format_RGB32);t.fill(Qt::white);
		QPainter p(&t);p.fillRect(5,4,6,15,Qt::black);p.fillRect(9,14,17,5,Qt::black);p.end();
		const auto tp=dir.filePath("mark.png");QVERIFY(t.save(tp));QVERIFY(w.loadTemplate(tp));
		w.findChild<QDoubleSpinBox*>("score")->setValue(.92);
		w.findChild<QSpinBox*>("targetCount")->setValue(2);
		QStringList problems;QSet<int> centres;QSet<int> samples;
		connect(&w,&MainWindow::matchingFinished,&w,[&](bool ok){
			const auto path=dir.filePath("frame.json");
			if(!ok || !w.exportResults(path)) {problems<<"match";return;}
			QFile f(path);if(!f.open(QIODevice::ReadOnly)) {problems<<"file";return;}
			const auto result=QJsonDocument::fromJson(f.readAll()).object();
			const auto frame=result["run"].toObject()["input_frame"].toObject();
			if(frame["interval_ms"].toInt()!=60 || frame["sample"].toInt()<1) problems<<"frame";
			if(samples.contains(frame["sample"].toInt())) problems<<"duplicate";
			samples.insert(frame["sample"].toInt());
			const auto m=result["matches"].toArray();
			if(m.size()!=1) {problems<<"count";return;}
			centres.insert(qRound(m[0].toObject()["pos_x"].toDouble()));
		});
		QVERIFY(w.startLiveVideo(movie,60));QVERIFY(!w.loadTemplate(tp));QVERIFY(!w.startLiveVideo(movie,60));
		QTRY_VERIFY_WITH_TIMEOUT(w.findChild<QLabel*>("liveInfo")->text().contains("播放完畢"),12000);
		QTRY_VERIFY_WITH_TIMEOUT(w.findChild<QPushButton*>("loadSource")->isEnabled(),5000);
		QVERIFY2(problems.isEmpty(),qPrintable(problems.join(",")));QVERIFY(centres.contains(36));QVERIFY(centres.contains(112));
		const auto png=dir.filePath("sample.png");QVERIFY(w.saveSampledFrame(png));QCOMPARE(QImage(png).size(),QSize(160,120));
		QVERIFY(w.startLiveVideo(movie,60));w.findChild<QPushButton*>("liveInput")->click();
		QVERIFY(w.findChild<QPushButton*>("loadSource")->isEnabled());
	}
#endif
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


	void locateByColour()
	{
		QTemporaryDir dir; QVERIFY(dir.isValid());
		QImage target(48,40,QImage::Format_RGB888), other(target.size(),target.format());
		QVERIFY(fpm::grayFromRgb(190,52,80)==97 && fpm::grayFromRgb(31,135,80)==97);
		for(int y=0;y<40;++y) for(int x=0;x<48;++x) {
			const bool ink=(x>=7 && x<17 && y>=5 && y<32) || (x>=15 && x<38 && y>=24 && y<32)
				|| (x>=29 && x<36 && y>=6 && y<15);
			const int t=((x/3)*17+(y/3)*31+(x/3)*(y/3)*7)%19-9;
			target.setPixelColor(x,y,ink?QColor(190+t,52+t,80+t):QColor(97+t,97+t,97+t));
			other.setPixelColor(x,y,ink?QColor(31+t,135+t,80+t):QColor(97+t,97+t,97+t));
		}
		const auto tp=dir.filePath("template.png"),sp=dir.filePath("pair.png"),np=dir.filePath("other.png");
		QVERIFY(target.save(tp));
		QImage source(270,150,QImage::Format_RGB888); source.fill(QColor(97,97,97));
		QPainter painter(&source); painter.drawImage(28,30,target); painter.drawImage(170,75,other); painter.end();
		QVERIFY(source.save(sp));
		painter.begin(&source); painter.drawImage(28,30,other); painter.end(); QVERIFY(source.save(np));
		MainWindow w; QVERIFY(w.loadSource(sp)); QVERIFY(w.loadTemplate(tp));
		const auto mode=w.findChild<QCheckBox*>("searchColour");
		const auto engine=w.findChild<QComboBox*>("engine");
		const auto button=w.findChild<QPushButton*>("execute");
		const auto table=w.findChild<QTableWidget*>("results");
		w.findChild<QDoubleSpinBox*>("score")->setValue(.9);
		w.findChild<QSpinBox*>("targetCount")->setValue(8);
		QSignalSpy done(&w,&MainWindow::matchingFinished);
		auto run=[&] { button->click(); return done.wait(20000) && done.takeFirst()[0].toBool(); };
		auto read=[](const QString& path) {
			QFile file(path); if(!file.open(QIODevice::ReadOnly)) return QJsonObject{};
			return QJsonDocument::fromJson(file.readAll()).object();
		};
		QVERIFY(run()); QCOMPARE(table->rowCount(),2);
		const auto legacy=dir.filePath("old.fpmrecipe"); QVERIFY(w.saveRecipe(legacy));
		auto old=read(legacy); auto oldSettings=old["settings"].toObject();
		oldSettings.remove("searchColour"); old["settings"]=oldSettings;
		QFile oldFile(legacy); QVERIFY(oldFile.open(QIODevice::WriteOnly|QIODevice::Truncate));
		oldFile.write(QJsonDocument(old).toJson()); oldFile.close();
		mode->setChecked(true); QVERIFY(run()); QCOMPARE(table->rowCount(),1);
		QCOMPARE(table->item(0,3)->text().toDouble(),52.); QCOMPARE(table->item(0,4)->text().toDouble(),50.);
		const auto recipe=dir.filePath("part.fpmrecipe"); QVERIFY(w.saveRecipe(recipe));
		QCOMPARE(read(recipe)["schema"].toString(),QString("fpm.part-recipe.v2"));
		QVERIFY(w.loadRecipe(legacy)); QVERIFY(!mode->isChecked()); QVERIFY(w.loadSource(sp)); QVERIFY(run()); QCOMPARE(table->rowCount(),2);
		QVERIFY(QFile::remove(tp)); QVERIFY(w.loadRecipe(recipe)); QVERIFY(mode->isChecked());
		w.findChild<QCheckBox*>("continuousRefinement")->setChecked(false);
		for(const int route:{0,4}) {
			if(engine->findData(route)<0) continue;
			engine->setCurrentIndex(engine->findData(route)); QVERIFY(w.loadSource(sp)); QVERIFY(run());
			QCOMPARE(table->rowCount(),1); QVERIFY(std::abs(table->item(0,3)->text().toDouble()-52)<.1);
			QVERIFY(std::abs(table->item(0,4)->text().toDouble()-50)<.1);
			const auto out=dir.filePath("result.json"); QVERIFY(w.exportResults(out));
			const auto record=read(out)["run"].toObject();
			QCOMPARE(record["search_feature"].toString(),QString("chroma_cocg"));
			QVERIFY(record["feature_ms"].isDouble());
			if(route==4) QVERIFY(record["full_gpu_retained_bytes"].toDouble()>0);
			QVERIFY(w.loadSource(np)); QVERIFY(run()); QCOMPARE(table->rowCount(),0);
		}
		QImage flat(32,30,QImage::Format_RGB888); flat.fill(Qt::red); QVERIFY(flat.save(tp));
		QVERIFY(w.loadTemplate(tp)); QVERIFY(!run()); QCOMPARE(table->rowCount(),0);
		QVERIFY(w.findChild<QLabel*>("status")->text().contains("模板"));
	}

#ifdef FPM_GUI_RESEARCH
	void calibratedSceneAndColour()
	{
		QTemporaryDir dir; QVERIFY(dir.isValid());
		MainWindow window;
		QImage templ(48,40,QImage::Format_RGB888), mask(templ.size(),QImage::Format_RGB888);
		QImage source(144,112,QImage::Format_RGB888); source.fill(Qt::black); mask.fill(Qt::black);
		for(int y=0;y<templ.height();++y) for(int x=0;x<templ.width();++x) {
			const int gray=35+(17*x+29*y+x*y*3)%180;
			QColor pixel(gray,gray,gray);
			if(y>=8 && y<16) {
				if(x>=8 && x<16) {mask.setPixelColor(x,y,Qt::red);pixel={150,55,70};}
				if(x>=20 && x<28) {mask.setPixelColor(x,y,Qt::green);pixel={190,190,190};}
				if(x>=32 && x<40) {mask.setPixelColor(x,y,Qt::blue);pixel={60,60,60};}
			}
			templ.setPixelColor(x,y,pixel);
			source.setPixelColor(40+x,32+y,mask.pixelColor(x,y)==Qt::green?QColor(Qt::white):pixel);
		}
		const auto tp=dir.filePath("template.png"),sp=dir.filePath("source.png"),mp=dir.filePath("mask.png");
		QVERIFY(templ.save(tp));QVERIFY(source.save(sp));QVERIFY(mask.save(mp));
		QVERIFY(window.loadTemplate(tp));QVERIFY(window.loadSource(sp));
		QVERIFY(!window.loadInspectionMask(sp));QVERIFY(window.loadInspectionMask(mp));
		const auto colourRecipe=dir.filePath("colour-part.fpmrecipe");
		QVERIFY(window.saveRecipe(colourRecipe));QVERIFY(QFile::remove(mp));QVERIFY(window.loadRecipe(colourRecipe));
		QVERIFY(window.findChild<QCheckBox*>("checkColour")->isChecked());QVERIFY(window.loadSource(sp));
		auto* engine=window.findChild<QComboBox*>("engine");auto* preset=window.findChild<QComboBox*>("scenePreset");
		auto* floor=window.findChild<QSpinBox*>("minTopShortSide");
		QVERIFY(engine && preset && floor);
		engine->setCurrentIndex(engine->findData(1));preset->activated(2);QCOMPARE(floor->value(),12);
		preset->activated(1);QCOMPARE(floor->value(),0);
		QFile recipe(dir.filePath("recipe.json"));QVERIFY(recipe.open(QIODevice::WriteOnly));
		QJsonObject job{{"angle",0},{"limit",5},{"score",.8},{"proposal_score",.5},{"overlap",.2},
			{"scales",QJsonArray{1.}},{"min_area",256},{"stop_layer",0}};
		QJsonObject arm{{"min_top_short_side",12}};
		QJsonObject report{{"schema","fpm.scene-tuning.v1"},{"selected_on_dev","short12"},{"backend",0},
			{"arms",QJsonObject{{"short12",arm}}},{"baseline",job}};
		// A final fractional scale step is representable without rounding the bank.
		job["scales"]=QJsonArray{.9,1.,1.07};report["baseline"]=job;
		recipe.write(QJsonDocument(report).toJson());recipe.close();QVERIFY(window.loadTuning(recipe.fileName()));
		QCOMPARE(window.findChild<QDoubleSpinBox*>("scaleMin")->value(),.9);
		QCOMPARE(window.findChild<QDoubleSpinBox*>("scaleMax")->value(),1.07);
		QCOMPARE(window.findChild<QDoubleSpinBox*>("scaleStep")->value(),.1);
		job["scales"]=QJsonArray{1.};report["baseline"]=job;
		QVERIFY(recipe.open(QIODevice::WriteOnly|QIODevice::Truncate));
		recipe.write(QJsonDocument(report).toJson());recipe.close();QVERIFY(window.loadTuning(recipe.fileName()));
		QCOMPARE(floor->value(),12);QCOMPARE(engine->currentData().toInt(),1);
		QVERIFY(!window.loadTuning(mp));QCOMPARE(floor->value(),12);
		auto* button=window.findChild<QPushButton*>("execute");auto* table=window.findChild<QTableWidget*>("results");
		QSignalSpy done(&window,&MainWindow::matchingFinished);
		auto run=[&] { button->click();return done.wait(30000) && done.takeFirst()[0].toBool(); };
		auto result=[&]() {
			const auto path=dir.filePath("out.json");if(!window.exportResults(path)) return QJsonObject();
			QFile file(path);if(!file.open(QIODevice::ReadOnly)) return QJsonObject();
			return QJsonDocument::fromJson(file.readAll()).object()["run"].toObject();
		};
		QVERIFY(run());QCOMPARE(table->rowCount(),1);QCOMPARE(table->columnCount(),7);
		auto measured=result();QCOMPARE(measured["min_top_short_side"].toInt(),12);
		auto colour=measured["colour_results"].toArray().first().toObject();
		QCOMPARE(colour["status"].toInt(),0);QCOMPARE(colour["reference"].toString(),QString("dark"));
		QCOMPARE(colour["decision"].toString(),QString("measured"));
		QVERIFY(window.loadTemplate(tp));QVERIFY(!window.findChild<QCheckBox*>("checkColour")->isChecked());
		if(engine->findData(4)>=0) {
			engine->setCurrentIndex(engine->findData(4));preset->activated(3);
			QVERIFY(window.findChild<QCheckBox*>("dedup")->isChecked());
			// Keep the small single-object fixture and its runtime bounded.
			window.findChild<QDoubleSpinBox*>("angle")->setValue(0);window.findChild<QSpinBox*>("targetCount")->setValue(5);
			QVERIFY(run());QCOMPARE(table->rowCount(),1);QCOMPARE(table->columnCount(),6);
			auto full=result();QCOMPARE(full["engine"].toString(),QString("full_metal"));
			QCOMPARE(full["full_gpu_peak_mode"].toInt(),2);QVERIFY(full["full_gpu_retained_bytes"].toDouble()>0);
		}
	}
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
			if(engine->itemData(route).toInt()>2) continue; // New dedicated routes have their own configuration.
			engine->setCurrentIndex(route);mode->setCurrentIndex(1);overlap->setChecked(true);
			QVERIFY(run());QCOMPARE(table->rowCount(),2);QVERIFY(window.exportResults(out));
			QFile file(out);QVERIFY(file.open(QIODevice::ReadOnly));auto data=QJsonDocument::fromJson(file.readAll()).object();
			QCOMPARE(data["schema_version"].toInt(),2);auto config=data["run"].toObject();QCOMPARE(config["ncc_mode"].toInt(),route==2?2:0);
			QCOMPARE(config["coarse_overlap"].toBool(),route==2);QCOMPARE(config["frame_generation"].toInt(),1);
			bool one=false,scaled=false;for(auto m:data["matches"].toArray()) {double scale=m.toObject()["scale"].toDouble();one|=scale==1.;scaled|=scale==.75;}QVERIFY(one && scaled);
			QVERIFY(run());QVERIFY(window.exportResults(out));file.close();QVERIFY(file.open(QIODevice::ReadOnly));
			QCOMPARE(QJsonDocument::fromJson(file.readAll()).object()["run"].toObject()["frame_generation"].toInt(),2);
			if((engine->findData(2)<0 || route==engine->findData(2)) && !qEnvironmentVariable("FPM_GUI_SCREENSHOT").isEmpty()) { QCoreApplication::processEvents();QVERIFY(window.grab().save(qEnvironmentVariable("FPM_GUI_SCREENSHOT"))); }
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
		QFile empty(out);QVERIFY(empty.open(QIODevice::ReadOnly));auto e=QJsonDocument::fromJson(empty.readAll()).object();QVERIFY(e["run"].toObject()["source_updated"].toBool());QVERIFY(e["run"].toObject()["session_reused"].toBool());QCOMPARE(e["matches"].toArray().size(),0);
		// Closing a busy window must leave the worker with owned image/session snapshots.
		auto closing=std::make_unique<MainWindow>();QVERIFY(closing->loadSource(sp));QVERIFY(closing->loadTemplate(tp));
		closing->findChild<QComboBox*>("engine")->setCurrentIndex(engine->count()-1);
		closing->findChild<QPushButton*>("execute")->click();closing.reset();QVERIFY(QThreadPool::globalInstance()->waitForDone(30000));
	}
#endif
	void operatorRecipeAndBatch()
	{
		QTemporaryDir dir;QVERIFY(dir.isValid());
		QImage print(QStringLiteral(FPM_FIXTURE_DIR "/Dst3.bmp"));QVERIFY(!print.isNull());
		const auto tp=dir.filePath("定位標記.png");QVERIFY(print.save(tp));
		auto frame=[&](const QString& name,int count) {
			QImage image(600,330,QImage::Format_RGB32);image.fill(QColor(182,182,182));
			QPainter painter(&image);
			const QPoint points[]={{28,55},{228,153},{430,68}};
			for(int i=0;i<count;++i) painter.drawImage(points[i],print);
			painter.end();const auto path=dir.filePath(name);return image.save(path)?path:QString();
		};
		const auto three=frame("工件-001.png",3),two=frame("工件-002.png",2);
		QVERIFY(!three.isEmpty() && !two.isEmpty());
		const auto saved=dir.filePath("定位標記-A.fpmrecipe");
		auto capture=[](MainWindow& w,const QString& name) {
			const auto folder=qEnvironmentVariable("FPM_OPERATOR_SCREENSHOTS");
			if(folder.isEmpty()) return true;
			QCoreApplication::processEvents();return w.grab().save(QDir(folder).filePath(name));
		};
		MainWindow setup;setup.show();
		setup.findChild<QPushButton*>("editSettings")->click();
		QVERIFY(setup.loadTemplate(tp));
		setup.findChild<QSpinBox*>("expectedCount")->setValue(3);
		setup.findChild<QSpinBox*>("targetCount")->setValue(3);
		QVERIFY(!setup.saveRecipe(saved));
		setup.findChild<QSpinBox*>("targetCount")->setValue(10);
		setup.findChild<QSpinBox*>("minArea")->setValue(20000);
		setup.findChild<QDoubleSpinBox*>("score")->setValue(.95);
		QVERIFY(setup.saveRecipe(saved));setup.close();
		const auto screenshots=qEnvironmentVariable("FPM_OPERATOR_SCREENSHOTS");
		if(!screenshots.isEmpty()) {
			const auto demo=QDir(screenshots).filePath("demo");QVERIFY(QDir().mkpath(demo));
			for(const auto& asset:{saved,three,two}) {
				const auto dest=QDir(demo).filePath(QFileInfo(asset).fileName());
				if(QFile::exists(dest)) QVERIFY(QFile::remove(dest));
				QVERIFY(QFile::copy(asset,dest));
			}
		}
		QVERIFY(QFile::remove(tp)); // A moved recipe owns its template.
		MainWindow w;w.show();QVERIFY(w.loadRecipe(saved));QVERIFY(capture(w,"02-ready.png"));
		auto* expected=w.findChild<QSpinBox*>("expectedCount");
		auto* recipe=w.findChild<QLabel*>("recipeInfo");
		QCOMPARE(expected->value(),3);QVERIFY(recipe->text().contains("定位標記-A"));
		QVERIFY(!w.findChild<QPushButton*>("editSettings")->isChecked());
		expected->setValue(2);QVERIFY(recipe->text().contains("已修改"));
		expected->setValue(3);QVERIFY(!recipe->text().contains("已修改"));
		auto read=[](const QString& path) {
			QFile f(path);if(!f.open(QIODevice::ReadOnly)) return QJsonObject();
			return QJsonDocument::fromJson(f.readAll()).object();
		};
		auto broken=read(saved);broken["template"]=QJsonObject{};
		QFile bad(dir.filePath("bad.fpmrecipe"));QVERIFY(bad.open(QIODevice::WriteOnly));
		bad.write(QJsonDocument(broken).toJson());bad.close();
		QVERIFY(!w.loadRecipe(bad.fileName()));QVERIFY(recipe->text().contains("定位標記-A"));
		const auto missing=dir.filePath("無法讀取.png");
		QVERIFY(w.startBatch({three,two,missing,three},dir.path()));
		auto* next=w.findChild<QPushButton*>("continueBatch");auto* status=w.findChild<QLabel*>("inspection");
		QTRY_VERIFY_WITH_TIMEOUT(next->isVisible(),15000);
		QVERIFY(status->text().contains("需複核"));QVERIFY(status->text().contains("2"));
		QVERIFY(capture(w,"03-review.png"));
		QVERIFY(!w.loadRecipe(saved));QVERIFY(!w.loadSource(three));
		const auto dirs=QDir(dir.path()).entryList({"MatchTool-*"},QDir::Dirs);
		QCOMPARE(dirs.size(),1);QDir out(dir.filePath(dirs.front()));
		auto normal=read(out.filePath("00001.json"))["run"].toObject();
		QCOMPARE(normal["inspection"].toObject()["found"].toInt(),3);QVERIFY(!normal["inspection"].toObject()["needs_review"].toBool());
		QCOMPARE(normal["recipe_sha256"].toString().size(),64);QVERIFY(!normal["recipe_modified"].toBool());
		QVERIFY(read(out.filePath("00002.json"))["run"].toObject()["inspection"].toObject()["needs_review"].toBool());
		next->click();QTRY_VERIFY(next->isVisible());QVERIFY(status->text().contains("未完成"));
		QCOMPARE(read(out.filePath("00003.json"))["source"].toString(),missing);
		QCOMPARE(w.findChild<QTableWidget*>("results")->rowCount(),0);
		next->click();QTRY_VERIFY_WITH_TIMEOUT(QFile::exists(out.filePath("00004.json")),10000);
		QVERIFY(!next->isVisible());QVERIFY(capture(w,"04-finished.png"));
		const auto summary=read(out.filePath("summary.json"));
		QCOMPARE(summary["saved"].toInt(),4);QCOMPARE(summary["review_indices"].toArray(),QJsonArray({2,3}));
		QCOMPARE(summary["error_indices"].toArray(),QJsonArray({3}));
		QVERIFY(w.findChild<QLabel*>("batchInfo")->text().contains("2 張需複核"));
		QVERIFY(w.findChild<QPushButton*>("openBatchFolder")->isVisible());
		// A failed archive keeps the current image until a deliberate retry or stop.
		QVERIFY(w.startBatch({three,two},dir.path()));
		QDir retryFolder(w.findChild<QLabel*>("batchInfo")->toolTip());
		QVERIFY(retryFolder.mkdir("00001.json"));QTRY_VERIFY(next->isVisible());
		QVERIFY(next->text().contains("重試"));QVERIFY(retryFolder.rmdir("00001.json"));
		next->click();QTRY_VERIFY_WITH_TIMEOUT(QFile::exists(retryFolder.filePath("00002.json")),10000);
		QVERIFY(!read(retryFolder.filePath("00001.json"))["matches"].toArray().isEmpty());
		QVERIFY(w.startBatch({three,three},dir.path()));
		QDir stopFolder(w.findChild<QLabel*>("batchInfo")->toolTip());
		w.findChild<QPushButton*>("stopBatch")->click();
		QTRY_VERIFY_WITH_TIMEOUT(QFile::exists(stopFolder.filePath("summary.json")),10000);
		const auto stopped=read(stopFolder.filePath("summary.json"));
		QVERIFY(stopped["stopped"].toBool());QCOMPARE(stopped["saved"].toInt(),1);
		QVERIFY(!QFile::exists(stopFolder.filePath("00002.json")));
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
