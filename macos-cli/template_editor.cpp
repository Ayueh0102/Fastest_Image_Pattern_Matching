// SPDX-License-Identifier: BSD-2-Clause
#include "template_editor.h"
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <functional>

class CropCanvas : public QWidget {
public:
	explicit CropCanvas(const QImage& image) : image_(image) {
		setMinimumSize(300,240); setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
		setCursor(Qt::CrossCursor); setObjectName("cropCanvas");
	}
	std::function<void(QRect)> onSelection;
	QTransform pose;
	QSize extent;
	int rows=1,cols=1;
	QRect selected;
protected:
	void paintEvent(QPaintEvent*) override {
		QPainter p(this); p.fillRect(rect(),QColor("#25292d"));
		p.setTransform(display()); p.drawImage(QRectF(QPointF(),image_.size()),image_);
		p.setWorldTransform(pose,true);
		QPen pen(QColor("#31e4c0")); pen.setCosmetic(true); pen.setWidth(2); p.setPen(pen);
		p.drawRect(QRectF(QPointF(),extent));
		for(int i=1;i<cols;++i) {int x=qint64(i)*extent.width()/cols;p.drawLine(x,0,x,extent.height());}
		for(int i=1;i<rows;++i) {int y=qint64(i)*extent.height()/rows;p.drawLine(0,y,extent.width(),y);}
		if(rows*cols>1) {p.fillRect(selected,QColor(255,202,68,55));pen.setColor(QColor("#ffca44"));p.setPen(pen);p.drawRect(QRectF(selected));}
	}
	void mousePressEvent(QMouseEvent* e) override {
		if(e->button()!=Qt::LeftButton) return;
		const auto pt=display().inverted().map(e->position());
		if(!QRectF(QPointF(),image_.size()).contains(pt)) return;
		start_=pixel(pt); dragging_=true;
	}
	void mouseMoveEvent(QMouseEvent* e) override { if(dragging_) select(e->position()); }
	void mouseReleaseEvent(QMouseEvent* e) override {
		if(dragging_ && e->button()==Qt::LeftButton) {dragging_=false; select(e->position());}
	}
private:
	QTransform display() const {
		const double s=std::min(double(width())/image_.width(),double(height())/image_.height());
		QTransform t; t.translate((width()-s*image_.width())/2,(height()-s*image_.height())/2);t.scale(s,s);return t;
	}
	QPoint pixel(QPointF p) const {return {std::clamp(int(std::floor(p.x())),0,image_.width()-1),std::clamp(int(std::floor(p.y())),0,image_.height()-1)};}
	void select(QPointF pos) {
		const auto end=pixel(display().inverted().map(pos));
		const QPoint lo(std::min(start_.x(),end.x()),std::min(start_.y(),end.y()));
		const QPoint hi(std::max(start_.x(),end.x()),std::max(start_.y(),end.y()));
		if(onSelection) onSelection(QRect(lo,hi));
	}
	QImage image_;
	QPoint start_;
	bool dragging_=false;
};

QString TemplateEditor::label(const char* tw,const char* cn,const char* en) const
{
	return QString::fromUtf8(language_==0?tw:language_==1?cn:en);
}

TemplateEditor::TemplateEditor(QImage source,int language,QWidget* parent)
	: QDialog(parent),source_(std::move(source)),language_(language)
{
	source_.setDevicePixelRatio(1);
	setWindowTitle(label("從影像製作模板","从图像制作模板","Create template from image"));resize(920,580);
	auto* layout=new QHBoxLayout(this);canvas_=new CropCanvas(source_);layout->addWidget(canvas_,1);
	auto* controls=new QVBoxLayout;layout->addLayout(controls);
	auto* help=new QLabel(label("拖曳框選工件，轉動選框對齊工件邊緣。","拖动框选工件，转动选框对齐工件边缘。","Drag over the part. Rotate the selection to align its edges."));
	help->setWordWrap(true);help->setMaximumWidth(250);controls->addWidget(help);
	auto* form=new QFormLayout;controls->addLayout(form);
	auto spin=[&](const char* id,int min,int max,int value) {
		auto* s=new QSpinBox; s->setObjectName(id);s->setRange(min,max);s->setValue(value);return s;
	};
	x_=spin("cropX",0,source_.width()-1,0); y_=spin("cropY",0,source_.height()-1,0);
	width_=spin("cropWidth",1,source_.width(),source_.width());height_=spin("cropHeight",1,source_.height(),source_.height());
	auto pair=[&](const QString& name,QSpinBox* a,QSpinBox* b){
		auto* row=new QHBoxLayout;row->addWidget(a);row->addWidget(b);form->addRow(name,row);
	};
	pair(label("起點 X、Y","起点 X、Y","Origin X, Y"),x_,y_);
	pair(label("寬、高 (px)","宽、高 (px)","Width, height (px)"),width_,height_);
	angle_=new QDoubleSpinBox;angle_->setObjectName("cropAngle");angle_->setRange(-180,180);angle_->setDecimals(1);angle_->setSuffix("°");
	form->addRow(label("選框順時針角度","选框顺时针角度","Clockwise angle"),angle_);
	rows_=spin("tileRows",1,16,1);cols_=spin("tileCols",1,16,1);
	pair(label("拆分列數、欄數","拆分行数、列数","Grid rows, columns"),rows_,cols_);
	selected_=spin("activeTile",1,1,1);form->addRow(label("使用第幾格","使用第几格","Active tile"),selected_);
	preview_=new QLabel;preview_->setFixedSize(240,150);preview_->setAlignment(Qt::AlignCenter);controls->addWidget(preview_);
	hint_=new QLabel;hint_->setWordWrap(true);hint_->setMaximumWidth(250);controls->addWidget(hint_);
	controls->addStretch();
	export_=new QPushButton(label("匯出全部分格 PNG…","导出全部分格 PNG…","Export all tiles as PNG…"));controls->addWidget(export_);
	auto* buttons=new QDialogButtonBox(QDialogButtonBox::Cancel);
	apply_=buttons->addButton(label("使用這個模板","使用这个模板","Use this template"),QDialogButtonBox::AcceptRole);
	apply_->setObjectName("useCrop");buttons->button(QDialogButtonBox::Cancel)->setText(label("取消","取消","Cancel"));controls->addWidget(buttons);
	connect(buttons,&QDialogButtonBox::accepted,this,&QDialog::accept);connect(buttons,&QDialogButtonBox::rejected,this,&QDialog::reject);
	for(auto* s:{x_,y_,width_,height_,rows_,cols_,selected_}) connect(s,&QSpinBox::valueChanged,this,[this]{refresh();});
	connect(angle_,&QDoubleSpinBox::valueChanged,this,[this]{refresh();});
	canvas_->onSelection=[this](QRect r) {
		const QSignalBlocker bx(x_),by(y_),bw(width_),bh(height_),ba(angle_);
		x_->setValue(r.x()); y_->setValue(r.y());width_->setValue(r.width());height_->setValue(r.height());angle_->setValue(0);refresh();
	};
	connect(export_,&QPushButton::clicked,this,[this] {
		const auto path=QFileDialog::getExistingDirectory(this,export_->text());if(path.isEmpty()) return;
		const auto saved=exportTiles(path);
		hint_->setText(saved.isEmpty()?label("無法儲存分格影像","无法保存分格图像","Could not save tiles"):QDir::toNativeSeparators(saved));
	});
	refresh();
}
QRect TemplateEditor::region() const {return {x_->value(),y_->value(),width_->value(),height_->value()};}
QRect TemplateEditor::cell(int i) const
{
	const int row=i/cols_->value(),col=i%cols_->value();
	auto edge=[](int i,int size,int count){return int(qint64(i)*size/count);};
	const int x=edge(col,width_->value(),cols_->value()),y=edge(row,height_->value(),rows_->value());
	return {x,y,edge(col+1,width_->value(),cols_->value())-x,edge(row+1,height_->value(),rows_->value())-y};
}
QTransform TemplateEditor::toSource() const
{
	const auto r=QRectF(region()); QTransform t;t.translate(r.center().x(),r.center().y());t.rotate(angle_->value());t.translate(-r.width()/2,-r.height()/2);return t;
}
bool TemplateEditor::fits() const
{
	const QRectF bounds=toSource().mapRect(QRectF(QPointF(),region().size()));
	return QRectF(QPointF(),source_.size()).adjusted(-1e-7,-1e-7,1e-7,1e-7).contains(bounds);
}
QImage TemplateEditor::render(const QRect& tile,QSize output) const
{
	if(!fits()) return {};
	if(angle_->value()==0 && tile.size()==output) return source_.copy(tile.translated(region().topLeft()));
	QImage dst(output,QImage::Format_RGB32);if(dst.isNull()) return {};
	dst.fill(Qt::black);QPainter p(&dst);p.setRenderHint(QPainter::SmoothPixmapTransform);
	p.scale(double(output.width())/tile.width(),double(output.height())/tile.height());
	p.translate(-tile.x(),-tile.y());p.setWorldTransform(toSource().inverted(),true);
	p.drawImage(QRectF(QPointF(),source_.size()),source_);p.end();return dst;
}
QImage TemplateEditor::selectedTile() const {const auto tile=cell(selected_->value()-1);return render(tile,tile.size());}
QString TemplateEditor::exportTiles(const QString& directory) const
{
	if(!fits() || !QDir(directory).exists()) return {};
	QTemporaryDir out(QDir(directory).filePath("MatchTool-tiles-XXXXXX"));if(!out.isValid()) return {};
	for(int i=0;i<rows_->value()*cols_->value();++i){
		const auto tile=cell(i);const auto name=QString("row%1-col%2.png").arg(i/cols_->value()+1).arg(i%cols_->value()+1);
		if(!render(tile,tile.size()).save(out.filePath(name),"PNG")) return {};
	}
	out.setAutoRemove(false);return out.path();
}
void TemplateEditor::refresh()
{
	const QSignalBlocker br(rows_),bc(cols_),bs(selected_);
	rows_->setMaximum(std::min(16,height_->value()));cols_->setMaximum(std::min(16,width_->value()));
	selected_->setMaximum(rows_->value()*cols_->value());selected_->setEnabled(selected_->maximum()>1);
	const auto tile=cell(selected_->value()-1);
	canvas_->pose=toSource();canvas_->extent=region().size();canvas_->rows=rows_->value();canvas_->cols=cols_->value();canvas_->selected=tile;canvas_->update();
	const bool valid=fits();apply_->setEnabled(valid);export_->setEnabled(valid);
	if(valid) {
		const auto size=tile.size().scaled(preview_->size(),Qt::KeepAspectRatio).expandedTo(QSize(1,1));
		preview_->setPixmap(QPixmap::fromImage(render(tile,size)));
		hint_->setText(QString("%1 × %2 px").arg(tile.width()).arg(tile.height()));
	} else {
		preview_->clear();hint_->setText(label("選框超出影像，請移動或縮小。","选框超出图像，请移动或缩小。","Move or shrink the selection to fit inside the image."));
	}
}
