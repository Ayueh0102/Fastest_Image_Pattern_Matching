#include "mainwindow.h"
#include "cli.hpp"
#include <QApplication>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
	bool gui = argc == 1;
	for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--gui") gui = true;
	if (!gui) return runPortableCli(argc, argv);
	QApplication app(argc, argv);
	QApplication::setApplicationName("Fastest Image Pattern Matching");
	MainWindow window;
	window.show();
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "--gui") continue;
		if ((arg == "--src" || arg == "--tmpl") && i + 1 < argc) {
			const QString path = QString::fromLocal8Bit(argv[++i]);
			const bool ok = arg == "--src" ? window.loadSource(path) : window.loadTemplate(path);
			if (!ok) return 1;
		} else {
			std::cerr << "GUI options: --gui [--src image] [--tmpl template]\n";
			return 1;
		}
	}
	return app.exec();
}
