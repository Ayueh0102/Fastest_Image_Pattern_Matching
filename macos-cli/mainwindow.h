#pragma once
#include "fpm/chroma.hpp"
#include <QMainWindow>
#include <QImage>
#include <QWidget>
#include <QFutureWatcher>
#include <QJsonObject>
#include <QJsonArray>
#include <QStringList>
#ifdef FPM_GUI_RESEARCH
#include "research_pipeline.hpp"
#include "color_inspection.h"
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
class QVBoxLayout;
class QTimer;
class LiveInput;

class ImageView : public QWidget {
	Q_OBJECT
public:
	explicit ImageView(QWidget* parent = nullptr);
	void setImage(const QImage& image, const QSize& viewport,bool keepZoom=false);
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
	std::shared_ptr<const fpm::ChromaTemplate> chroma;
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
	bool loadRecipe(const QString& path);
	bool saveRecipe(const QString& path);
#ifdef FPM_GUI_MULTIMEDIA
	bool saveSampledFrame(const QString& path) const;
	bool startLiveVideo(const QString& path,int interval);
#endif
	bool startBatch(const QStringList& paths, const QString& outputRoot);
	bool loadTemplate(const QString& path);
	bool exportResults(const QString& path) const;
#ifdef FPM_GUI_RESEARCH
	bool loadTuning(const QString& path);
	bool loadInspectionMask(const QString& path);
#endif
	signals:
	void matchingFinished(bool success);
protected:
	void closeEvent(QCloseEvent*) override;
	void dragEnterEvent(QDragEnterEvent*) override;
	void dropEvent(QDropEvent*) override;
private slots:
	void execute();
	void complete();
	void updateLanguage();
private:
	void clearResults();
	static QImage recipeGray(const QImage& image);
	void installTemplate(QImage gray, QImage original, const QString& name);
	void clearSource();
	void installSource(QImage gray,QImage original,const QString& name,bool keepZoom);
	void makeTemplate();
	LiveInput* live_=nullptr;
	QPushButton* crop_;
#ifdef FPM_GUI_MULTIMEDIA
	QPushButton *liveButton_,*saveFrame_;
	QLabel* liveInfo_;
	int liveInterval_=200;
	LiveInput* prepareLive(int interval);
	void startLocalCamera(int interval);
	void chooseLiveInput();
	void stopLive(const QString& reason={});
#endif
	QJsonObject frameInfo_;
	bool readSource(const QString& path);
	QList<QWidget*> recipeFields() const;
	QJsonObject recipeSettings() const;
	void refreshRecipe();
	void refreshInspection();
	void refreshBatch();
	void runBatchImage();
	void recordBatchImage(bool success);
	void finishBatch();
	void pickBatch();
	void chooseBatchOutput(const QStringList& paths);
	void buildOperatorPanel(QVBoxLayout* side, QVBoxLayout* controls);
	QString configError() const;
	QString recipePath_, recipeDigest_;
	QJsonObject savedSettings_, defaultSettings_;
	QTimer* batchTimer_;
	bool assetsChanged_ = false, inspectionFailed_ = false;
	QPushButton *openRecipe_, *saveRecipe_, *editSettings_, *batchButton_, *resumeBatch_, *stopBatch_, *openBatchFolder_;
	QLabel *recipeInfo_, *inspection_, *batchInfo_, *expectedLabel_;
	QSpinBox* expected_;
	QStringList batchPaths_;
	QString batchDir_, batchError_;
	int batchIndex_ = -1, batchSaved_ = 0;
	QJsonArray reviewIndices_, failedIndices_;
	QJsonObject batchSummary_;
	bool batchSummaryFailed_ = false;
	void saveBatchSummary();
	bool batchPaused_ = false, batchWriteFailed_ = false, stopRequested_ = false, closeAfterBatch_ = false;
	bool batchActive() const { return batchIndex_>=0; }
	void setBusy(bool busy);
	void updateEngineControls();
	QString text(const char* traditional, const char* simplified, const char* english) const;
	QImage source_, templ_, sourceColour_, templateColour_;
	std::shared_ptr<const fpm::ChromaTemplate> chroma_;
	QCheckBox* searchColour_;
	void updateTemplatePreview();
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
	bool sessionSourceDirty_ = false;
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
#ifdef FPM_GUI_RESEARCH
	void applyScene(int index);
	QComboBox* scene_;
	QSpinBox* shortSide_;
	QDoubleSpinBox* proposalScore_;
	QLabel *shortSideLabel_, *proposalLabel_;
	QCheckBox *verifySmall_, *dedup_;
	QPushButton* loadTuning_;
	QPushButton* loadMask_;
	QCheckBox* checkColour_;
	QDoubleSpinBox* colourThreshold_;
	QLabel* colourThresholdLabel_;
	InspectionMask inspectionMask_;
#endif
	QDoubleSpinBox *scaleMin_, *scaleMax_, *scaleStep_;
	QCheckBox *coarseOverlap_, *continuous_;
	QLabel *engineLabel_, *scaleLabel_, *nccLabel_;
	QTableWidget* table_;
	QGroupBox* parameters_;
	QLabel* labels_[5];
};
