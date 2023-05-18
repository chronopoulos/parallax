from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QGridLayout, QHBoxLayout, QLineEdit
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QComboBox, QSlider
from PyQt5.QtCore import pyqtSignal, Qt, QModelIndex, QMimeData, QTimer
from PyQt5.QtCore import QThread, QObject
from PyQt5.QtGui import QIcon, QDoubleValidator, QPixmap

import numpy as np
import time
from enum import Enum
from skimage.transform import hough_line
from scipy.ndimage import gaussian_filter1d
import pyqtgraph as pg
import cv2

from . import get_image_file
from .helper import FONT_BOLD
from .stage_dropdown import StageDropdown
from .control_panel import JogMode, AxisControl, PositionWorker
from .screen_widget import ScreenWidget

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

        self.threshold = -40
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(-255)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.setToolTip('Threshold')
        self.threshold_slider.sliderMoved.connect(self.set_threshold)

        self.lscreen = ScreenWidget(model=self.model)
        self.rscreen = ScreenWidget(model=self.model)
        self.screens_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.lscreen)
        layout.addWidget(self.rscreen)
        self.screens_widget.setLayout(layout)

        self.refresh_lscreen_timer = QTimer()
        self.refresh_lscreen_timer.timeout.connect(self.lscreen.refresh)
        self.refresh_lscreen_timer.start(125)

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.histoplot = self.graph_widget.addPlot(row=0, col=0)
        self.histoplot.setLabel('bottom', 'Pixel Value Histogram')
        self.stdplot = self.graph_widget.addPlot(row=0, col=1)
        self.stdplot.setLabel('bottom', 'STD along Line')
        self.hitsplot = self.graph_widget.addPlot(row=0, col=2)
        self.hitsplot.setLabel('bottom', 'Hits along Line')

        # layout
        main_layout = QGridLayout()
        main_layout.addWidget(self.main_label, 0,0, 1,3)
        main_layout.addWidget(self.stage_dropdown, 1,0, 1,3)
        main_layout.addWidget(self.xcontrol, 3,0, 1,1)
        main_layout.addWidget(self.ycontrol, 3,1, 1,1)
        main_layout.addWidget(self.zcontrol, 3,2, 1,1)
        main_layout.addWidget(self.threshold_slider, 4,0, 1,3)
        main_layout.addWidget(self.screens_widget, 5,0, 4,3)
        main_layout.addWidget(self.graph_widget, 9,0, 4,3)
        self.setLayout(main_layout)

        self.stage = None
        self.jog_default = 300e-6
        self.jog_fine = 50e-6
        self.jog_coarse = 500e-6

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

    def set_threshold(self, val):
        self.threshold = val
        print(val)

    def handle_stage_selection(self, index):
        stage_name = self.stage_dropdown.currentText()
        self.set_stage(self.model.stages[stage_name])

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
            if self.lscreen.camera:
                frame1 = self.lscreen.camera.get_last_image_data()
                self.stage.move_relative_1d(axis, distance*1e6)
                while not self.stage.get_1d_done(axis):
                    time.sleep(0.1)
                frame2 = self.lscreen.camera.get_last_image_data()
                x,y = self.detect_probe_from_move(frame1, frame2)
                self.lscreen.select((x,y))

    def detect_probe_from_move(self, f1, f2):
        # upcast to float32
        f1 = np.array(f1, dtype='float32')
        f2 = np.array(f2, dtype='float32')
        # convert to mono
        f1 = np.mean(f1, axis=2)
        f2 = np.mean(f2, axis=2)
        # take difference
        diff = f2 - f1
        hist, bins = np.histogram(diff, bins=256)
        bargraph = pg.BarGraphItem(x0=bins[:-1], x1=bins[1:], height=hist, brush ='r')
        self.histoplot.clear()
        self.histoplot.addItem(bargraph)
        self.histoplot.setXRange(-30, 30)
        # take threshold
        mask = np.zeros(diff.shape, dtype='uint8')
        if self.threshold < 0:
            mask[diff <= self.threshold] = 255
        else:
            mask[diff >= self.threshold] = 255
        # do hough transform
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        hh, tt, dd = hough_line(mask, theta=tested_angles)
        # find best fit distance (d) and theta (t)
        ii, jj = np.where(hh == np.max(hh))
        i,j = ii[0], jj[0]
        d = dd[i]
        t = tt[j]
        # define 2 points on that line
        p1 = (d/np.cos(t)), 0
        p2 = (d-3000*np.sin(t))/np.cos(t), 3000
        # find std along that line
        (x_intercept, y_intercept) = d * np.array([np.cos(t), np.sin(t)])
        y = np.arange(3000)
        x = x_intercept - (y - y_intercept) * np.tan(t)
        probe_width = 30
        x_probe = np.concatenate([x.astype('int') + i for i in range(-probe_width//2,probe_width//2)])
        y_probe = np.concatenate([y.astype('int') for i in range(-probe_width//2,probe_width//2)])
        diffs_along_probe = np.reshape(diff[y_probe, x_probe],(probe_width,3000))
        std_along_probe = np.std(diffs_along_probe, axis=0)
        std_along_probe_smoothed = gaussian_filter1d(std_along_probe,5)
        linegraph_std = pg.PlotDataItem(std_along_probe_smoothed)
        self.stdplot.clear()
        self.stdplot.addItem(linegraph_std)
        # find number of probe points in vicinity from mask
        hits_along_probe = np.reshape(mask[y_probe, x_probe], (probe_width,3000))
        sumhits_along_probe = np.sum(hits_along_probe, axis=0)
        sumhits_along_probe_smoothed = gaussian_filter1d(sumhits_along_probe, 5)
        linegraph_hits = pg.PlotDataItem(sumhits_along_probe_smoothed)
        self.hitsplot.clear()
        self.hitsplot.addItem(linegraph_hits)
        # use hitscount theshold crossing as probe tip
        where = np.where(sumhits_along_probe_smoothed > 4000)
        itip = where[0][0]
        xtip, ytip = x[itip], y[itip]
        # draw a line and marker on the mask, show on screen_widget
        cv2.line(mask, tuple(int(i) for i in p1), tuple(int(i) for i in p2), 255, thickness=1)
        cv2.drawMarker(mask, (int(xtip), int(ytip)), (255,255,255), cv2.MARKER_CROSS, 15, 5)
        self.rscreen.set_data(mask)
        return xtip, ytip

    def halt(self):
        self.stage.halt()

