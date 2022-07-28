from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QFrame, QMenu
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt

from Helper import *
from Dialogs import *
import time
import socket

JOG_SIZE_STEPS = 1000

class Dropdown(QComboBox):
    popupAboutToBeShown = pyqtSignal()

    def __init__(self):
        QComboBox.__init__(self)
        self.setFocusPolicy(Qt.NoFocus)

    def showPopup(self):
        self.popupAboutToBeShown.emit()
        super(Dropdown, self).showPopup()


class AxisControl(QLabel):
    jogRequested = pyqtSignal(bool)
    centerRequested = pyqtSignal()

    def __init__(self, text):
        QLabel.__init__(self)
        self.text = text

        self.setText(self.text)
        self.setAlignment(Qt.AlignCenter)

    def setValue(self, val):
        self.value = val
        self.setText('{0} = {1}'.format(self.text, self.value))

    def wheelEvent(self, e):
        e.accept()
        self.jogRequested.emit(e.angleDelta().y() > 0)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.centerRequested.emit()

class ControlPanel(QFrame):
    msgPosted = pyqtSignal(str)

    def __init__(self, model):
        QFrame.__init__(self)
        self.model = model

        # widgets
        self.dropdown = Dropdown()
        self.dropdown.popupAboutToBeShown.connect(self.populateDropdown)
        self.dropdown.activated.connect(self.handleStageSelection)

        self.xcontrol = AxisControl('X')
        self.xcontrol.jogRequested.connect(self.jogX)
        self.xcontrol.centerRequested.connect(self.centerX)
        self.ycontrol = AxisControl('Y')
        self.ycontrol.jogRequested.connect(self.jogY)
        self.ycontrol.centerRequested.connect(self.centerY)
        self.zcontrol = AxisControl('Z')
        self.zcontrol.jogRequested.connect(self.jogZ)
        self.zcontrol.centerRequested.connect(self.centerZ)

        self.zeroButton = QPushButton('Zero')
        self.zeroButton.clicked.connect(self.zero)

        self.moveTargetButton = QPushButton('Move to Target')
        self.moveTargetButton.clicked.connect(self.moveToTarget)

        self.moveRandomButton = QPushButton('Move Random')
        self.moveRandomButton.clicked.connect(self.moveRandom)

        # layout
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.dropdown, 0,0, 1,3)
        mainLayout.addWidget(self.xcontrol, 1,0, 1,1)
        mainLayout.addWidget(self.ycontrol, 1,1, 1,1)
        mainLayout.addWidget(self.zcontrol, 1,2, 1,1)
        mainLayout.addWidget(self.zeroButton, 2,0, 1,3)
        mainLayout.addWidget(self.moveTargetButton, 3,0, 1,3)
        mainLayout.addWidget(self.moveRandomButton, 4,0, 1,3)
        self.setLayout(mainLayout)

        # frame border
        self.setFrameStyle(QFrame.Box | QFrame.Plain);
        self.setLineWidth(2);

        self.stage = None

    def updateCoordinates(self):
        # could be faster if we separate out the 3 coordinates
        x, y, z = self.stage.getPosition_rel()
        self.xcontrol.setValue(x)
        self.ycontrol.setValue(y)
        self.zcontrol.setValue(z)

    def handleStageSelection(self, index):
        socket.setdefaulttimeout(0.100)  # 50 ms timeout
        ip = self.dropdown.currentText()
        self.setStage(self.model.stages[ip])
        self.updateCoordinates()

    def populateDropdown(self):
        self.dropdown.clear()
        for ip in self.model.stages.keys():
            self.dropdown.addItem(ip)

    def setStage(self, stage):
        self.stage = stage

    def setCenter(self):
        dlg = CenterDialog()
        if dlg.exec_():
            x,y,z = dlg.getParams()
            self.stage.setCenter(x,y,z)

    def moveToTarget(self):
        dlg = TargetDialog()
        if dlg.exec_():
            params = dlg.getParams()
            x = params['x']
            y = params['y']
            z = params['z']
            if self.stage:
                if params['relative']:
                    self.stage.moveToTarget3d_rel(x, y, z)
                else:
                    self.stage.moveToTarget3d_abs(x, y, z)
                self.msgPosted.emit('Moved to position: (%f, %f, %f) mm' % (x, y, z))
                self.updateCoordinates()

    def moveRandom(self):
        if self.stage:
            x = np.random.uniform(7000, 8000)
            y = np.random.uniform(7000, 8000)
            z = np.random.uniform(7000, 8000)
            self.stage.moveToTarget3d_abs(x, y, z)
            self.msgPosted.emit('Moved to position: (%f, %f, %f) mm' % (x, y, z))
            self.updateCoordinates()

    def jogX(self, forward):
        if self.stage:
            self.stage.selectAxis('x')
            direction = 'forward' if forward else 'backward'
            self.stage.moveClosedLoopStep(direction, JOG_SIZE_STEPS)
            self.stage.wait()
            self.updateCoordinates()

    def jogY(self, forward):
        if self.stage:
            self.stage.selectAxis('y')
            direction = 'forward' if forward else 'backward'
            self.stage.moveClosedLoopStep(direction, JOG_SIZE_STEPS)
            self.stage.wait()
            self.updateCoordinates()

    def jogZ(self, forward):
        if self.stage:
            self.stage.selectAxis('z')
            direction = 'forward' if forward else 'backward'
            self.stage.moveClosedLoopStep(direction, JOG_SIZE_STEPS)
            self.stage.wait()
            self.updateCoordinates()

    def centerX(self):
        if self.stage:
            self.stage.selectAxis('x')
            self.stage.moveToTarget(15000)
            self.stage.wait()
            self.updateCoordinates()

    def centerY(self):
        if self.stage:
            self.stage.selectAxis('y')
            self.stage.moveToTarget(15000)
            self.stage.wait()
            self.updateCoordinates()

    def centerZ(self):
        if self.stage:
            self.stage.selectAxis('z')
            self.stage.moveToTarget(15000)
            self.stage.wait()
            self.updateCoordinates()

    def zero(self):
        if self.stage:
            x, y, z = self.stage.getPosition_abs()
            self.stage.setOrigin(x, y, z)
            self.zeroButton.setText('Zero: (%d %d %d)' % (x, y, z))
            self.updateCoordinates()

    def halt(self):
        # doesn't actually work now because we need threading
        self.stage.halt()


