from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPainter, QPixmap, QImage, QColor, qRgb
from PyQt5.QtCore import Qt, pyqtSignal

import PySpin
import time, datetime
import numpy as np
import cv2 as cv

from Camera import Camera
from glue import *

xScreen = 500
yScreen = 375


class ScreenWidget(QLabel):

    clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        self.setMinimumSize(xScreen, yScreen)
        self.setMaximumSize(xScreen, yScreen)
        self.setData(np.zeros((HCV,WCV), dtype=np.uint8))

    def setData(self, data):

        self.data = data
        self.qimage = QImage(data, data.shape[1], data.shape[0], QImage.Format_Grayscale8)
        self.show()

    def show(self):

        pixmap = QPixmap(self.qimage.scaled(xScreen, yScreen, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        self.setPixmap(pixmap)
        self.update()

    def updatePixel(self, x, y):

        for i in range(x-8, x+8):
            for j in range(y-8, y+8):
                self.qimage.setPixel(i, j, qRgb(255, 255, 255))
        self.show()

    def mousePressEvent(self, e):

        if e.button() == Qt.LeftButton:
            self.xclicked = e.x()
            self.yclicked = e.y()
            pixel = self.data[self.yclicked*4, self.xclicked*4]
            self.updatePixel(self.xclicked*4, self.yclicked*4)
            self.clicked.emit(self.xclicked, self.yclicked)
    

class FlirTab(QWidget):

    def __init__(self, msgLog):
        QWidget.__init__(self)

        self.msgLog = msgLog
        self.ncameras = 0
        self.initGui()

        self.initialized = False
        self.lastImage = None

        self.projectionsDone = False
        self.lCorrDone = False
        self.rCorrDone = False

        self.initCameras()  # tmp

    def initCameras(self):

        self.msgLog.post('Initializing cameras...')

        self.instance = PySpin.System.GetInstance()

        self.libVer = self.instance.GetLibraryVersion()
        self.libVerString = 'Version %d.%d.%d.%d' % (self.libVer.major, self.libVer.minor,
                                                    self.libVer.type, self.libVer.build)
        self.cameras_pyspin = self.instance.GetCameras()
        self.ncameras = self.cameras_pyspin.GetSize()

        ncameras_string = '%d camera%s detected' % (self.ncameras, 's' if self.ncameras!=1 else '')
        self.msgLog.post(ncameras_string)
        if self.ncameras < 2:
            print('Error: need at least 2 cameras')
            return

        ###

        self.lcamera = Camera(self.cameras_pyspin.GetByIndex(0))
        self.rcamera = Camera(self.cameras_pyspin.GetByIndex(1))

        self.msgLog.post('FLIR Library Version is %s' % self.libVerString)

        self.captureButton.setEnabled(True)
        self.initializeButton.setEnabled(False)
        self.initialized = True

    def clean(self):

        if (self.initialized):
            print('cleaning up SpinSDK')
            time.sleep(1)
            self.lcamera.clean()
            self.rcamera.clean()
            self.cameras_pyspin.Clear()
            self.instance.ReleaseInstance()

    def initGui(self):

        mainLayout = QVBoxLayout()

        self.screens = QWidget()
        hlayout = QHBoxLayout()
        self.lscreen = ScreenWidget()
        self.lscreen.clicked.connect(self.handleLscreenClicked)
        hlayout.addWidget(self.lscreen)
        self.rscreen = ScreenWidget()
        self.rscreen.clicked.connect(self.handleRscreenClicked)
        hlayout.addWidget(self.rscreen)
        self.screens.setLayout(hlayout)

        self.initializeButton = QPushButton('Initialize Cameras')
        self.initializeButton.clicked.connect(self.initCameras)

        self.captureButton = QPushButton('Capture Frames')
        self.captureButton.setEnabled(False)
        self.captureButton.clicked.connect(self.capture)

        self.checkerboardButton = QPushButton('Find Checkerboards')
        self.checkerboardButton.setEnabled(False)
        self.checkerboardButton.clicked.connect(self.findCheckerboards)

        self.projectionButton = QPushButton('Compute Projection Matrices')
        self.projectionButton.setEnabled(False)
        self.projectionButton.clicked.connect(self.computeProjectionMatrices)

        self.triangulateButton = QPushButton('Triangulate')
        self.triangulateButton.setEnabled(False)
        self.triangulateButton.clicked.connect(self.triangulate)

        self.saveButton = QPushButton('Save Last Frame')
        self.saveButton.setEnabled(False)
        self.saveButton.clicked.connect(self.save)

        mainLayout.addWidget(self.initializeButton)
        mainLayout.addWidget(self.screens)
        mainLayout.addWidget(self.captureButton)
        mainLayout.addWidget(self.checkerboardButton)
        mainLayout.addWidget(self.projectionButton)
        mainLayout.addWidget(self.triangulateButton)
        mainLayout.addWidget(self.saveButton)

        self.setLayout(mainLayout)

    def capture(self):

        ts = time.time()
        dt = datetime.datetime.fromtimestamp(ts)
        strTime = '%04d%02d%02d-%02d%02d%02d' % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        self.lastStrTime= strTime

        self.lcamera.capture()
        self.lscreen.setData(cv.pyrDown(self.lcamera.getLastImageData()))

        self.rcamera.capture()
        self.rscreen.setData(cv.pyrDown(self.rcamera.getLastImageData()))

        self.saveButton.setEnabled(True)
        self.checkerboardButton.setEnabled(True)

    def computeProjectionMatrices(self):

        lcornerss = []
        for i in range(NUM_CAL_IMG):
            self.lcamera.capture()
            ret, corners = cv.findChessboardCorners(cv.pyrDown(self.lcamera.getLastImageData()), (NCW,NCH), None)
            if not ret:
                self.msgLog.post('left corners not found')
                return
            lcornerss.append(corners)
        self.lproj = getProjectionMatrix(lcornerss)

        rcornerss = []
        for i in range(NUM_CAL_IMG):
            self.rcamera.capture()
            ret, corners = cv.findChessboardCorners(cv.pyrDown(self.rcamera.getLastImageData()), (NCW,NCH), None)
            if not ret:
                self.msgLog.post('right corners not found')
                return
            rcornerss.append(corners)
        self.rproj = getProjectionMatrix(rcornerss)

        self.projectionsDone = True

    def findCheckerboards(self):

        self.ldata = cv.pyrDown(self.lcamera.getLastImageData()) # half-res
        self.lret, self.lcorners = cv.findChessboardCorners(self.ldata, (NCW,NCH), None)
        if self.lret:
            self.lscreen.setData(cv.drawChessboardCorners(self.ldata, (NCW,NCH), self.lcorners, self.lret))
        else:
            self.msgLog.post('Checkerboard corners not found in left frame')

        self.rdata = cv.pyrDown(self.rcamera.getLastImageData()) # half-res
        self.rret, self.rcorners = cv.findChessboardCorners(self.rdata, (NCW,NCH), None)
        if self.rret:
            self.rscreen.setData(cv.drawChessboardCorners(self.rdata, (NCW,NCH), self.rcorners, self.rret))
        else:
            self.msgLog.post('Checkerboard corners not found in left frame')

        self.projectionButton.setEnabled(True)

    def save(self):

        image_converted = self.lcamera.getLastImage().Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
        filename = 'lcamera_%s.jpg' % self.lastStrTime
        image_converted.Save(filename)
        self.msgLog.post('Saved %s' % filename)

        image_converted = self.rcamera.getLastImage().Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
        filename = 'rcamera_%s.jpg' % self.lastStrTime
        image_converted.Save(filename)
        self.msgLog.post('Saved %s' % filename)

    def handleLscreenClicked(self, xclicked, yclicked):

        self.lCorrPoint = np.array([[xclicked], [yclicked]])
        print('lCorrPoint = ', self.lCorrPoint)
        self.lCorrDone = True
        self.enableTriangulationMaybe()

    def handleRscreenClicked(self, xclicked, yclicked):

        self.rCorrPoint = np.array([[xclicked], [yclicked]])
        print('rCorrPoint = ', self.rCorrPoint)
        self.rCorrDone = True
        self.enableTriangulationMaybe()

    def enableTriangulationMaybe(self):

        if self.projectionsDone:
            if self.lCorrDone and self.rCorrDone:
                self.triangulateButton.setEnabled(True)

    def triangulate(self):

        result_homogeneous = cv.triangulatePoints(self.lproj, self.rproj, self.lCorrPoint, self.rCorrPoint)
        print('result_homogeneous = ', result_homogeneous)

        #result_euclidean = cv.convertPointsFromHomogeneous(np.array(result_homogeneous))
        # ok i'll do it myself
        xe = result_homogeneous[0] / result_homogeneous[3]
        ye = result_homogeneous[1] / result_homogeneous[3]
        ze = result_homogeneous[2] / result_homogeneous[3]
        print('result_euclidean = ', xe, ye, ze)

