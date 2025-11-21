from PySide6.QtWidgets import (
    QWidget, QGroupBox, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QHBoxLayout
)
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

class ConfigPanel(QWidget):
    input_changed = Signal(str)
    output_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()
        
        group_box = QGroupBox("Configuration")
        gb_layout = QGridLayout()

        # Row 1: Input File
        gb_layout.addWidget(QLabel("Input File:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("Select input Excel file...")
        gb_layout.addWidget(self.input_edit, 0, 1)
        
        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.clicked.connect(self.browse_input)
        gb_layout.addWidget(self.browse_input_btn, 0, 2)

        # Row 2: Output Dir
        gb_layout.addWidget(QLabel("Output Dir:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Select output directory...")
        gb_layout.addWidget(self.output_edit, 1, 1)
        
        btn_layout = QHBoxLayout()
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output)
        btn_layout.addWidget(self.browse_output_btn)
        
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        btn_layout.addWidget(self.open_folder_btn)
        
        gb_layout.addLayout(btn_layout, 1, 2)

        # Row 3: Status
        gb_layout.addWidget(QLabel("Status:"), 2, 0)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        gb_layout.addWidget(self.status_label, 2, 1, 1, 2)

        group_box.setLayout(gb_layout)
        layout.addWidget(group_box)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    @Slot()
    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input Excel File", "", "Excel Files (*.xlsx *.xls)")
        if path:
            self.input_edit.setText(path)
            self.input_changed.emit(path)

    @Slot()
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_edit.setText(path)
            self.open_folder_btn.setEnabled(True)
            self.output_changed.emit(path)

    @Slot()
    def open_output_folder(self):
        path = self.output_edit.text()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def set_status(self, message, is_error=False):
        self.status_label.setText(message)
        if is_error:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: gray; font-style: italic;")

    def set_controls_enabled(self, enabled):
        self.browse_input_btn.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
