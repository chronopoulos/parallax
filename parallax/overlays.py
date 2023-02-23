import cv2
import numpy as np
import random

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import pyqtSignal, Qt

from .stage_dropdown import StageDropdown

class NoOverlay:

    name = "None"

    def __init__(self, model):
        self.model = model

    def set_model(self, model):
        pass

    def process(self, frame):
        return -1,-1,-1

    def launch_control_panel(self):
        pass

class CoordinateOverlay:

    name = 'Stage Coordinates'

    def __init__(self, model):
        self.model = model
        self.stage = None

    def handle_stage_selection(self, index):
        stage_name = self.dropdown.currentText()
        self.set_stage(self.model.stages[stage_name])
        self.update_coordinates()

    def set_stage(self, stage):
        self.stage = stage

    def process(self, frame):
        if self.stage is not None:
            pos = self.stage.get_position()
            return pos
        else:
            return -1,-1,-1

    def launch_control_panel(self):
        self.control_panel = QWidget()
        self.dropdown = StageDropdown(self.model)
        self.dropdown.activated.connect(self.handle_stage_selection)
        layout = QVBoxLayout()
        layout.addWidget(self.dropdown)
        self.control_panel.setLayout(layout)
        self.control_panel.setWindowTitle('Stage Coordinate Overlay')
        self.control_panel.setMinimumWidth(300)
        self.control_panel.show()

class BogusCoordinateOverlay:

    name = 'Bogus Coordinates'

    def __init__(self, model):
        self.model = model

    def process(self, frame):
        x = random.uniform(0,15000)
        y = random.uniform(0,15000)
        z = random.uniform(0,15000)
        pos = x,y,z
        return pos

    def launch_control_panel(self):
        pass

