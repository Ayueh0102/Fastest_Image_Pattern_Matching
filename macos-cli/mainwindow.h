#pragma once
#include "fpm/matcher.hpp"
#include <QMainWindow>
#include <QImage>
#include <QWidget>
#include <QFutureWatcher>
#include <QJsonObject>
#ifdef FPM_GUI_RESEARCH
#include "research_pipeline.hpp"
#endif
#include <memory>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QPushButton;
class QScrollArea;
class QSpinBox;
class QTableWidget;

class ImageView : public QWidget {
	Q_OBJECT
public:
	explicit ImageView(QWidget* parent = nullptr);
	void setImage(const QImage& image, const QSize& viewport);
	void setResults(const std::vector<fpm::Match>& results);
	void selectResult(int index);
	void fit(const QSize& viewport);
	signals:
	void mouseMoved(int x, int y);
protected:
	void paintEvent(QPaintEvent*) override;
	void wheelEvent(QWheelEvent*) override;
	void mouseMoveEvent(QMouseEvent*) override;
private:
	void resizeImage();
	QImage image_;
	std::vector<fpm::Match> results_;
	double scale_ = 1, fitScale_ = 1;
	int selected_ = -1;
};

struct MatchJobResult {
	std::vector<fpm::Match> matches;
	std::shared_ptr<const fpm::Pattern> pattern;
	QString error;
	double learnMs = 0, matchMs = 0;
	std::vector<double> scales;
	QJsonObject provenance;
#ifdef FPM_GUI_RESEARCH
	std::shared_ptr<fpm::research::Session> session;
	QString sessionKey;
#endif
};

class MainWindow : public QMainWindow {
	Q_OBJECT
public:
	explicit MainWindow(QWidget* parent = nullptr);
	bool loadSource(const QString& path);
	bool loadTemplate(const QString& path);
	bool exportResults(const QString& path) const;
	signals:
	void matchingFinished(bool success);
protected:
	void dragEnterEvent(QDragEnterEvent*) override;
	void dropEvent(QDropEvent*) override;
private slots:
	void execute();
	void complete();
	void updateLanguage();
private:
	void clearResults();
	void setBusy(bool busy);
	void updateEngineControls();
	QString text(const char* traditional, const char* simplified, const char* english) const;
	QImage source_, templ_;
	std::shared_ptr<const fpm::Pattern> pattern_;
	int learnedArea_ = 0;
	std::vector<fpm::Match> results_;
	QFutureWatcher<MatchJobResult> watcher_;
	bool busy_ = false, hasCompleted_ = false;
	std::vector<double> resultScales_;
	QJsonObject provenance_;
	QString sourcePath_, templatePath_;
#ifdef FPM_GUI_RESEARCH
	std::shared_ptr<fpm::research::Session> session_;
	QString sessionKey_;
#endif
	ImageView* view_;
	QScrollArea* imageScroll_;
	QLabel *templatePreview_, *sourceInfo_, *templateInfo_, *status_;
	QPushButton *loadSource_, *loadTemplate_, *execute_, *export_, *fit_;
	QSpinBox *targets_, *minArea_;
	QDoubleSpinBox *score_, *overlap_, *angle_, *ranges_[4];
	QCheckBox *useRanges_, *subpixel_, *invert_, *simd_, *fast_;
	QComboBox* language_;
	QComboBox *engine_, *nccMode_;
	QDoubleSpinBox *scaleMin_, *scaleMax_, *scaleStep_;
	QCheckBox *coarseOverlap_, *continuous_;
	QLabel *engineLabel_, *scaleLabel_, *nccLabel_;
	QTableWidget* table_;
	QGroupBox* parameters_;
	QLabel* labels_[5];
};
