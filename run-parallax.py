#!/usr/bin/env python
import argparse
from PyQt5.QtWidgets import QApplication
from parallax.model import Model
from parallax.main_window import MainWindow

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help="Configuration file to load")

args = parser.parse_args()

app = QApplication([])

model = Model()
main_window = MainWindow(model)

if args.config is not None:
    model.load_config(args.config)

main_window.show()

app.exec()

model.clean()
