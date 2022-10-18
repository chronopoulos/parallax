from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QFrame, QFileDialog
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import os
import numpy as np
import time
import cv2
import pickle

from Calibration import Calibration
from lib import *
from Helper import *
from Dialogs import CalibrationDialog


class TriangulationPanel(QFrame):
    msgPosted = pyqtSignal(str)

    def __init__(self, model):
        QFrame.__init__(self)
        self.model = model

        # layout
        mainLayout = QVBoxLayout()
        self.mainLabel = QLabel('Triangulation')
        self.mainLabel.setAlignment(Qt.AlignCenter)
        self.mainLabel.setFont(FONT_BOLD)
        self.calButton = QPushButton('Run Calibration Routine')
        self.goButton = QPushButton('Triangulate Points')
        self.detectButton = QPushButton('Detect Probe')

        self.statusLabel = QLabel()
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setFont(FONT_BOLD)
        self.updateStatus()

        mainLayout.addWidget(self.mainLabel)
        mainLayout.addWidget(self.calButton)
        mainLayout.addWidget(self.statusLabel)
        mainLayout.addWidget(self.goButton)
        mainLayout.addWidget(self.detectButton)
        self.setLayout(mainLayout)

        # connections
        self.calButton.clicked.connect(self.launchCalibrationDialog)
        self.goButton.clicked.connect(self.triangulate)
        self.detectButton.clicked.connect(self.detectProbe)

        # frame border
        self.setFrameStyle(QFrame.Box | QFrame.Plain);
        self.setLineWidth(2);


    def detectProbe(self):

        # grab stage and cameras
        stages = list(self.model.stages.values())
        stage = stages[0]
        cameras = list(self.model.cameras.values())
        lcamera = cameras[0]
        rcamera = cameras[1]

        # capture before
        lcamera.capture()
        rcamera.capture()
        lbefore = lcamera.getLastImageData()
        rbefore = rcamera.getLastImageData()

        # retract slightly
        x,y,z = stage.getPosition_abs()
        stage.moveToTarget3d_abs(x, y, z+100)

        # capture after
        lcamera.capture()
        rcamera.capture()
        lafter = lcamera.getLastImageData()
        rafter = rcamera.getLastImageData()

        # un-retract
        stage.moveToTarget3d_abs(x, y, z)

        # TODO diff the images and save them
        ldiff = lafter - lbefore
        cv2.imwrite('ldiff.png', ldiff)
        rdiff = rafter - rbefore
        cv2.imwrite('rdiff.png', rdiff)

        # for now just save them
        data = (lbefore, rbefore, lafter, rafter)
        with open('diffs.pkl', 'wb') as f:
            pickle.dump(data, f)

    def triangulate(self):

        if not (self.model.calibration and self.model.lcorr and self.model.rcorr):
            self.msgPosted.emit('Error: please load a calibration, and select '
                                'correspondence points before attempting triangulation')
            return

        x,y,z = self.model.triangulate()
        self.msgPosted.emit('Reconstructed object point: '
                            '[{0:.2f}, {1:.2f}, {2:.2f}]'.format(x, y, z))

    def launchCalibrationDialog(self):

        dlg = CalibrationDialog(self.model)
        if dlg.exec_():

            intrinsicsLoad = dlg.getIntrinsicsLoad()
            stage = dlg.getStage()
            res = dlg.getResolution()
            extent = dlg.getExtent()

            self.model.setCalStage(stage)
            if intrinsicsLoad:
                print('TODO load intrinsics from file')
                return
            self.model.calFinished.connect(self.updateStatus)
            self.model.startCalibration(res, extent)

    def load(self):
        filename = QFileDialog.getOpenFileName(self, 'Load calibration file', '.',
                                                    'Pickle files (*.pkl)')[0]
        if filename:
            self.model.loadCalibration(filename)
            self.updateStatus()

    def save(self):

        if not self.model.calibration:
            self.msgPosted.emit('Error: no calibration loaded')
            return

        filename = QFileDialog.getSaveFileName(self, 'Save calibration file', '.',
                                                'Pickle files (*.pkl)')[0]
        if filename:
            self.model.saveCalibration(filename)
            self.msgPosted.emit('Saved calibration to: %s' % filename)

    def updateStatus(self):

        if self.model.calibration:
            x,y,z = self.model.calibration.getOrigin()
            msg = 'Calibration loaded.\nOrigin = [{0:.2f}, {1:.2f}, {2:.2f}]'.format(x,y,z)
            self.statusLabel.setText(msg)
        else:
            self.statusLabel.setText('No calibration loaded.')


