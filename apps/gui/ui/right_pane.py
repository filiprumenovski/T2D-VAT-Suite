from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QComboBox, QToolBar, QPlainTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap

class PlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.images = {} # name -> path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Toolbar
        self.toolbar = QToolBar()
        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self.on_combo_changed)
        self.toolbar.addWidget(QLabel("Figure: "))
        self.toolbar.addWidget(self.combo)
        layout.addWidget(self.toolbar)

        # Image Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel("No figure generated yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def update_images(self, image_paths):
        self.images = {p.split("/")[-1]: p for p in image_paths}
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(self.images.keys()))
        self.combo.blockSignals(False)
        
        if self.images:
            self.combo.setCurrentIndex(0)
            self.display_image(list(self.images.values())[0])

    def display_image(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            # Scale if too large, but keep aspect ratio
            # For now, just set pixmap, let scroll area handle it or scale to width
            # Better: Scale to fit width of scroll area if larger
            w = self.scroll_area.width() - 20
            if pixmap.width() > w:
                pixmap = pixmap.scaledToWidth(w, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText(f"Failed to load image: {path}")

    @Slot(int)
    def on_combo_changed(self, index):
        name = self.combo.currentText()
        if name in self.images:
            self.display_image(self.images[name])

    def select_image_by_path(self, path):
        name = path.split("/")[-1]
        index = self.combo.findText(name)
        if index >= 0:
            self.combo.setCurrentIndex(index)

class LogsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def log(self, message):
        self.text_edit.appendPlainText(message)

class RightPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.plots_tab = PlotsTab()
        self.logs_tab = LogsTab()
        
        self.tabs.addTab(self.plots_tab, "Plots")
        self.tabs.addTab(self.logs_tab, "Logs")
        
        layout.addWidget(self.tabs)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
