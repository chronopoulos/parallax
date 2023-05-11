from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QGridLayout, QLineEdit
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QComboBox
from PyQt5.QtCore import pyqtSignal, Qt, QModelIndex, QMimeData, QTimer
from PyQt5.QtCore import QThread, QObject
from PyQt5.QtGui import QIcon, QDoubleValidator, QPixmap

import numpy as np
import time
from enum import Enum

from . import get_image_file
from .helper import FONT_BOLD
from .stage_dropdown import StageDropdown
from .control_panel import JogMode, AxisControl, PositionWorker

class ClosedLoopProbeDetector(QWidget):
    msg_posted = pyqtSignal(str)

    def __init__(self, model, screens):
        QWidget.__init__(self)
        self.model = model
        self.screens = screens

        # widgets

        self.main_label = QLabel('Closed Loop Probe Detector')
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_label.setFont(FONT_BOLD)

        self.stage_dropdown = StageDropdown(self.model)
        self.stage_dropdown.activated.connect(self.handle_stage_selection)

        self.halt_button = QPushButton()
        self.halt_button.setIcon(QIcon(get_image_file('stop-sign.png')))
        self.halt_button.setToolTip('Halt This Stage')
        self.halt_button.clicked.connect(self.halt)

        self.xcontrol = AxisControl('x')
        self.xcontrol.jog_requested.connect(self.jog)
        self.ycontrol = AxisControl('y')
        self.ycontrol.jog_requested.connect(self.jog)
        self.zcontrol = AxisControl('z')
        self.zcontrol.jog_requested.connect(self.jog)

        self.screen_dropdown = QComboBox()
        self.screen_dropdown.addItem('Screen 1')
        self.screen_dropdown.addItem('Screen 2')
        self.screen_dropdown.setToolTip('Select a screen')
        self.screen_dropdown.activated.connect(self.handle_screen_selection)

        # layout
        main_layout = QGridLayout()
        main_layout.addWidget(self.main_label, 0,0, 1,4)
        main_layout.addWidget(self.stage_dropdown, 1,0, 1,2)
        main_layout.addWidget(self.halt_button, 1,3, 1,2)
        main_layout.addWidget(self.screen_dropdown, 2,0, 1,4)
        main_layout.addWidget(self.xcontrol, 3,0, 1,1)
        main_layout.addWidget(self.ycontrol, 3,1, 1,1)
        main_layout.addWidget(self.zcontrol, 3,2, 1,1)
        self.setLayout(main_layout)

        self.stage = None
        self.jog_default = 50e-6
        self.jog_fine = 10e-6
        self.jog_coarse = 250e-6

        self.setWindowTitle('Closed Loop Probe Detector')
        self.setWindowIcon(QIcon(get_image_file('sextant.png')))

        # position worker and thread
        self.pos_thread = QThread()
        self.pos_worker = PositionWorker()
        self.pos_worker.moveToThread(self.pos_thread)
        self.pos_thread.started.connect(self.pos_worker.run)
        self.pos_worker.finished.connect(self.pos_thread.quit)
        self.pos_worker.finished.connect(self.pos_worker.deleteLater)
        self.pos_thread.finished.connect(self.pos_thread.deleteLater)

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_coordinates)
        self.refresh_timer.start(100)

        self.handle_screen_selection(0)

    def handle_stage_selection(self, index):
        stage_name = self.stage_dropdown.currentText()
        self.set_stage(self.model.stages[stage_name])

    def handle_screen_selection(self, index):
        self.screen = self.screens[index]
        self.camera = self.screen.camera

    def update_coordinates(self, *args):
        if self.stage is not None:
            if self.pos_worker.pos_cached is not None:
                x, y, z = self.pos_worker.pos_cached
                self.xcontrol.set_value(x)
                self.ycontrol.set_value(y)
                self.zcontrol.set_value(z)

    def set_stage(self, stage):
        self.stage = stage
        self.pos_worker.set_stage(self.stage)
        self.pos_thread.start()

    def jog(self, axis, forward, jog_mode):
        if self.stage is not None:
            if jog_mode == JogMode.FINE:
                distance = self.jog_fine
            elif jog_mode == JogMode.COARSE:
                distance = self.jog_coarse
            else:
                distance = self.jog_default
            if not forward:
                distance = (-1) * distance
            if self.screen and self.camera:
                frame1 = self.camera.get_last_image_data()
                self.stage.move_relative_1d(axis, distance*1e6)
                while not self.stage.get_1d_done(axis):
                    time.sleep(0.1)
                frame2 = self.camera.get_last_image_data()
                x,y = self.detect_probe_from_move(frame1, frame2)
                self.screen.select((x,y))

    def detect_probe_from_move(self, f1, f2):
        print('detecting')
        #diff = f2 - f1
        diff = f1 - f2
        threshold = np.min(diff) * 0.6
        mask = np.zeros(diff.shape, dtype='uint8')
        mask[diff < threshold] = 1
        print('sum of mask: ', np.sum(mask))
        x = np.random.uniform(0,4000)
        y = np.random.uniform(0,3000)
        print('detection done')
        return x,y

    def halt(self):
        self.stage.halt()

