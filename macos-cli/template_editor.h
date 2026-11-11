// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include <QDialog>
#include <QImage>

class CropCanvas;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSpinBox;

class TemplateEditor : public QDialog {
public:
	TemplateEditor(QImage source,int language,QWidget* parent=nullptr);
	QImage selectedTile() const;
	QString exportTiles(const QString& directory) const;
private:
	void refresh();
	QString label(const char* traditional,const char* simplified,const char* english) const;
	QRect region() const;
	QRect cell(int index) const;
	QTransform toSource() const;
	bool fits() const;
	QImage render(const QRect& tile,QSize output) const;
	QImage source_;
	int language_;
	CropCanvas* canvas_;
	QSpinBox *x_,*y_,*width_,*height_,*rows_,*cols_,*selected_;
	QDoubleSpinBox* angle_;
	QLabel *preview_,*hint_;
	QPushButton *apply_,*export_;
};
