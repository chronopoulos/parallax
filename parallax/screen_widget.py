import functools
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QAction, QSlider, QMenu
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import QtCore
import pyqtgraph as pg
import inspect
import importlib

from . import filters


class ScreenWidgetControl(QWidget):

    selected = pyqtSignal(int, int)
    cleared = pyqtSignal()

    def __init__(self, filename=None, model=None, parent=None):
        QWidget.__init__(self)
        self.screen_widget = ScreenWidget(filename, model, parent)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setValue(50)
        self.contrast_slider.setToolTip('Contrast')

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setToolTip('Brightness')

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.screen_widget)
        self.layout.addWidget(self.contrast_slider)
        self.layout.addWidget(self.brightness_slider)
        self.setLayout(self.layout)

        # connections
        self.screen_widget.selected.connect(self.selected)
        self.screen_widget.cleared.connect(self.cleared)

    def update_camera_menu(self):
        self.screen_widget.update_camera_menu()

    def update_focus_control_menu(self):
        self.screen_widget.update_focus_control_menu()

    def refresh(self):
        self.screen_widget.refresh()

    def zoom_out(self):
        self.screen_widget.zoom_out()

    def clear_selected(self):
        self.screen_widget.clear_selected()


class ScreenWidget(pg.GraphicsView):

    selected = pyqtSignal(int, int)
    cleared = pyqtSignal()

    def __init__(self, filename=None, model=None, parent=None):
        super().__init__(parent=parent)
        self.filename = filename
        self.model = model

        self.view_box = pg.ViewBox(defaultPadding=0)
        self.setCentralItem(self.view_box)
        self.view_box.setAspectLocked()
        self.view_box.invertY()

        self.image_item = ClickableImage()
        self.image_item.axisOrder = 'row-major'
        self.view_box.addItem(self.image_item)
        self.image_item.mouse_clicked.connect(self.image_clicked)

        self.click_target = pg.TargetItem()
        self.view_box.addItem(self.click_target)
        self.click_target.setVisible(False)

        self.camera_actions = []
        self.focochan_actions = []
        self.filter_actions = []

        # still needed?
        self.camera_action_separator = self.view_box.menu.insertSeparator(self.view_box.menu.actions()[0])

        if self.filename:
            self.set_data(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

        self.clear_selected()

        self.camera = None
        self.focochan = None
        self.filter = filters.NoFilter()

        # sub-menus
        self.parallax_menu = QMenu("Parallax", self.view_box.menu)
        self.camera_menu = self.parallax_menu.addMenu("Cameras")
        self.focochan_menu = self.parallax_menu.addMenu("Focus Controllers")
        self.filter_menu = self.parallax_menu.addMenu("Filters")
        self.view_box.menu.insertMenu(self.view_box.menu.actions()[0], self.parallax_menu)

        self.update_filter_menu()

    def refresh(self):
        if self.camera:
            # takes a 3000,4000 grayscale image straight from the camera
            self.camera.capture()
            self.set_data(self.camera.get_last_image_data())

    def clear_selected(self):
        self.click_target.setVisible(False)
        self.cleared.emit()

    def set_data(self, data):
        data = self.filter.process(data)
        self.image_item.setImage(data, autoLevels=False)

    def update_camera_menu(self):
        for act in self.camera_actions:
            act.triggered.disconnect(act.callback)
            self.camera_menu.removeAction(act)
        for camera in self.model.cameras:
            act = self.camera_menu.addAction(camera.name())
            act.callback = functools.partial(self.set_camera, camera)
            act.triggered.connect(act.callback)
            self.camera_actions.append(act)

    def update_focus_control_menu(self):
        for act in self.focochan_actions:
            self.focochan_menu.removeAction(act)
        for foco in self.model.focos:
            for chan in range(6):
                act = self.focochan_menu.addAction(foco.ser.port + ', channel %d' % chan)
                act.callback = functools.partial(self.set_focochan, foco, chan)
                act.triggered.connect(act.callback)
                self.focochan_actions.append(act)

    def update_filter_menu(self):
        for act in self.filter_actions:
            self.filter_menu.removeAction(act)
        for name, obj in inspect.getmembers(filters):
            if inspect.isclass(obj) and (obj.__module__ == 'parallax.filters'):
                act = self.filter_menu.addAction(obj.name)
                act.callback = functools.partial(self.set_filter, obj)
                act.triggered.connect(act.callback)
                self.filter_actions.append(act)

    def image_clicked(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:            
            self.click_target.setPos(event.pos())
            self.click_target.setVisible(True)
            self.selected.emit(*self.get_selected())
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:            
            self.zoom_out()

    def zoom_out(self):
        self.view_box.autoRange()

    def set_camera(self, camera):
        self.camera = camera
        self.refresh()

    def set_focochan(self, foco, chan):
        self.focochan = (foco, chan)

    def set_filter(self, filt):
        self.filter = filt()
        self.filter.launch_control_panel()

    def get_selected(self):
        if self.click_target.isVisible():
            pos = self.click_target.pos()
            return pos.x(), pos.y()
        else:
            return None

    def wheelEvent(self, e):
        forward = bool(e.angleDelta().y() > 0)
        control = bool(e.modifiers() & Qt.ControlModifier)
        if control:
            if self.focochan:
                foco, chan = self.focochan
                foco.time_move(chan, forward, 100, wait=True)
        else:
            super().wheelEvent(e)


class ClickableImage(pg.ImageItem):
    mouse_clicked = pyqtSignal(object)    
    def mouseClickEvent(self, ev):
        super().mouseClickEvent(ev)
        self.mouse_clicked.emit(ev)
