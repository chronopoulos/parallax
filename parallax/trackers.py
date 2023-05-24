import cv2
import numpy as np
import random
from time import perf_counter
import os

from PyQt5.QtWidgets import QWidget, QLabel, QSlider, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog
from PyQt5.QtWidgets import QComboBox, QSpinBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon

from . import get_image_file, data_dir
from .helper import FONT_BOLD

"""
Trackers are objects that process the frames from a Camera, and return
the pixel coordinates of the object they are tracking. Trackers are similar to
detectors, but they need to be initialized with a bounding box. 
"""

class OpenCVTracker(QWidget):

    tracked = pyqtSignal(tuple)

    name = 'OpenCV Tracker'

    algos = {
                'BOOSTING' : cv2.TrackerBoosting_create,
                'MIL' : cv2.TrackerMIL_create,
                'KCF' : cv2.TrackerKCF_create,
                'TLD' : cv2.TrackerTLD_create,
                'MOSSE' : cv2.TrackerMOSSE_create,
                'CSRT' : cv2.TrackerCSRT_create
            }

    def __init__(self, model, target_screen):
        QWidget.__init__(self)
        self.model = model
        self.screen = target_screen
        self.tracker = None
        self.camera = None

        self.instruction_label = QLabel("Double click to select Tip and ROI")

        from .screen_widget import ScreenWidget
        self.roi_screen = ScreenWidget(model=self.model)
        self.roi_screen.set_camera(self.screen.camera)

        self.algo_label = QLabel('Algorithm:')
        self.algo_drop = QComboBox()
        for k in self.algos.keys():
            self.algo_drop.addItem(k)
        self.algo_drop.currentTextChanged.connect(self.handle_algo)

        self.init_button = QPushButton('Initialize')
        self.init_button.clicked.connect(self.initialize)
        
        self.refresh_button = QPushButton('Refresh Frame')
        self.refresh_button.clicked.connect(self.refresh_frame)

        self.status_label = QLabel('Status: Not Initialized')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(FONT_BOLD)
        
        layout = QGridLayout()
        layout.addWidget(self.instruction_label, 0,0,1,2)
        layout.addWidget(self.roi_screen, 1,0,6,2)
        layout.addWidget(self.algo_label, 7,0,1,1)
        layout.addWidget(self.algo_drop, 7,1,1,1)
        layout.addWidget(self.init_button, 8,0,1,1)
        layout.addWidget(self.refresh_button, 8,1,1,1)
        layout.addWidget(self.status_label, 9,0,1,2)
        self.setLayout(layout)

        self.setWindowTitle('OpenCV Tracker Control')
        self.setWindowIcon(QIcon(get_image_file('sextant.png')))
        self.setMinimumWidth(300)

    def refresh_frame(self):
        self.roi_screen.refresh()

    def initialize(self):
        pos, size, _ = self.roi_screen.get_roi()
        sel = self.roi_screen.get_selected()
        self.offset = sel[0] - pos[0], sel[1] - pos[1]

        if self.tracker is not None:
            self.tracker.clear()
            self.tracker = None

        try:
            algo_name = self.algo_drop.currentText()
            tracker_create = self.algos[algo_name]
            self.tracker = tracker_create()
        except KeyError as e:
            print('No such tracker algorithm found: ', e)
            return

        if (self.tracker is not None) and (self.screen.camera is not None):
            frame = self.screen.camera.get_last_image_data()
            bbox = (*pos, *size)
            ok = self.tracker.init(frame, bbox)

    def process(self):
        if (self.tracker is not None) and (self.screen.camera is not None):
            frame = self.screen.camera.get_last_image_data()
            t1 = perf_counter()
            ok, bbox = self.tracker.update(frame)
            t2 = perf_counter()
            dt = t2 - t1
            if ok:
                xb, yb = bbox[:2]
                xo, yo = self.offset
                x,y = xb+xo, yb+yo
                self.status_label.setText('Status: Tracking (%.2f FPS)' % (1/dt))
            else:
                x,y = 0,0
                self.status_label.setText('Status: Lost')
            pos = (x,y)
            self.tracked.emit(pos)

    def handle_algo(self, name):
        try:
            self.tracker_class = self.algos[name]
        except KeyError as e:
            print('No such tracker algorithm found: ', e)

