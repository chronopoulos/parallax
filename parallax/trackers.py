import cv2
import numpy as np
import random
import time
import os

from PyQt5.QtWidgets import QWidget, QLabel, QSlider, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog
from PyQt5.QtWidgets import QComboBox, QSpinBox, QSpacerItem
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QObject
from PyQt5.QtGui import QIcon

from . import get_image_file, data_dir
from .helper import FONT_BOLD

"""
Trackers are objects that process the frames from a Camera, and return
the pixel coordinates of the object they are tracking. Trackers are similar to
detectors, but they need to be initialized with a bounding box. 
"""


class NoTracker(QObject):

    name = 'None'

    tracked = pyqtSignal(tuple)

    def __init__(self, model, target_screen):
        QObject.__init__(self)

    def show(self):
        pass

    def process(self, frame):
        pass


class OpenCVTracker(QWidget):

    tracked = pyqtSignal(tuple)

    name = 'OpenCV Tracker'

    algos = {
                'KCF' : cv2.TrackerKCF_create,
                'MOSSE' : cv2.TrackerMOSSE_create,
                'MIL' : cv2.TrackerMIL_create,
                'CSRT' : cv2.TrackerCSRT_create,
                'MedianFlow' : cv2.TrackerMedianFlow_create,
                #'GOTURN' : cv2.TrackerGOTURN_create,       # bug in opencv 4.2.0
                'Boosting' : cv2.TrackerBoosting_create,    # crashy
                'TLD' : cv2.TrackerTLD_create               # drunk
            }

    class CVWorker(QObject):
        """
        Inner class to conform with object inspection when populating ScreenWidget menu
        """
        finished = pyqtSignal()
        tracked = pyqtSignal(tuple)
        status_updated = pyqtSignal(bool, float)

        def __init__(self):
            QObject.__init__(self)
            self.new = False
            self.tracker = None
            self.running = True

        def update_frame(self, frame):
            self.frame = frame
            self.new = True

        def initialize_tracker(self, tracker, bbox, offset):
            tracker.init(self.frame, bbox)
            self.tracker = tracker
            self.offset = offset

        def process(self, frame):
            if self.tracker is not None:
                t1 = time.perf_counter()
                ok, bbox = self.tracker.update(frame)
                t2 = time.perf_counter()
                self.status_updated.emit(ok, 1./(t2-t1))
                if ok:
                    xb, yb = bbox[:2]
                    xo, yo = self.offset
                    x,y = xb+xo, yb+yo
                else:
                    x,y = 0,0
                pos = (x,y)
                self.tracked.emit(pos)

        def quit(self):
            self.running = False

        def run(self):
            while self.running:
                if self.tracker is not None:
                    if self.new:
                        self.process(self.frame)
                        self.new = False
                time.sleep(0.001)
            self.finished.emit()

    def __init__(self, model, target_screen):
        QWidget.__init__(self)
        self.model = model
        self.screen = target_screen
        self.tracker = None

        # CV worker and thread
        self.cv_thread = QThread()
        self.cv_worker = self.CVWorker()
        self.cv_worker.moveToThread(self.cv_thread)
        self.cv_thread.started.connect(self.cv_worker.run)
        self.cv_worker.finished.connect(self.cv_thread.quit, Qt.DirectConnection)
        self.cv_worker.finished.connect(self.cv_worker.deleteLater)
        self.cv_thread.finished.connect(self.cv_thread.deleteLater)
        self.cv_worker.tracked.connect(self.tracked)
        self.cv_worker.status_updated.connect(self.update_status)
        self.cv_thread.start()

        self.instruction_label = QLabel("Double click to select Tip and ROI")

        from .screen_widget import ScreenWidget
        self.roi_screen = ScreenWidget(model=self.model) # don't need the model?

        self.algo_label = QLabel('Algorithm:')
        self.algo_label.setAlignment(Qt.AlignCenter)
        self.algo_drop = QComboBox()
        for k in self.algos.keys():
            self.algo_drop.addItem(k)

        self.init_button = QPushButton('Initialize')
        self.init_button.clicked.connect(self.initialize)
        
        self.status_label = QLabel('Status: Not Initialized')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(FONT_BOLD)
        
        layout = QGridLayout()
        layout.addWidget(self.instruction_label, 0,0,1,4)
        layout.addWidget(self.roi_screen, 1,0,6,4)
        layout.addWidget(self.algo_label, 7,0,1,1)
        layout.addWidget(self.algo_drop, 7,1,1,1)
        layout.addWidget(self.init_button, 7,2,1,1)
        layout.addWidget(self.status_label, 8,0,1,4)
        self.setLayout(layout)

        self.setWindowTitle('OpenCV Tracker Control')
        self.setWindowIcon(QIcon(get_image_file('sextant.png')))
        self.setMinimumWidth(300)

    def update_status(self, ok, fps):
        if ok:
            self.status_label.setText('Status: Tracking (%.2f FPS)' % fps)
        else:
            self.status_label.setText('Status: Lost')

    def refresh_frame(self):
        self.roi_screen.refresh()

    def initialize(self):
        pos, size, _ = self.roi_screen.get_roi()
        sel = self.roi_screen.get_selected()
        offset = sel[0] - pos[0], sel[1] - pos[1]
        try:
            algo_name = self.algo_drop.currentText()
            tracker_create = self.algos[algo_name]
            tracker = tracker_create()
        except KeyError as e:
            print('No such tracker algorithm found: ', e)
            tracker = None
            return
        bbox = (*pos, *size)
        self.cv_worker.initialize_tracker(tracker, bbox, offset)

    def process(self, frame):
        self.roi_screen.set_data(frame)
        self.cv_worker.update_frame(frame)

    def clean(self):
        self.cv_worker.quit()
        self.cv_thread.wait()

