from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QMessageBox, QStatusBar
)
from PySide6.QtCore import Qt, QThreadPool

from apps.gui.ui.config_panel import ConfigPanel
from apps.gui.ui.left_pane import LeftPane
from apps.gui.ui.right_pane import RightPane
from apps.gui.workers import Worker
from apps.gui.logic import run_reproducer_task, run_injector_task

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T2D-VAT Suite")
        self.resize(1200, 800)
        
        self.input_path = ""
        self.output_dir = ""
        self.threadpool = QThreadPool()

        self.init_ui()

    def init_ui(self):
        # Menu Bar (Stub)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("Exit", self.close)
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("About", self.show_about)

        # Central Widget
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Config Panel
        self.config_panel = ConfigPanel()
        self.config_panel.input_changed.connect(self.set_input_path)
        self.config_panel.output_changed.connect(self.set_output_dir)
        main_layout.addWidget(self.config_panel)

        # Splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        self.left_pane = LeftPane()
        self.left_pane.run_reproducer_requested.connect(self.run_reproducer)
        self.left_pane.run_predictor_requested.connect(self.run_predictor)
        self.left_pane.show_image_requested.connect(self.show_image)
        
        self.right_pane = RightPane()
        
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.right_pane)
        self.splitter.setStretchFactor(1, 1) # Give more space to right pane
        
        main_layout.addWidget(self.splitter)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def set_input_path(self, path):
        self.input_path = path
        self.check_ready()

    def set_output_dir(self, path):
        self.output_dir = path
        self.check_ready()

    def check_ready(self):
        ready = bool(self.input_path and self.output_dir)
        # We don't strictly disable tabs, but we check before running
        pass

    def show_about(self):
        QMessageBox.about(self, "About T2D-VAT Suite", "T2D-VAT Suite\n\nA tool for proteomics analysis.")

    def show_image(self, path):
        self.right_pane.plots_tab.select_image_by_path(path)
        self.right_pane.tabs.setCurrentIndex(0) # Switch to Plots tab

    def set_busy(self, busy, message=""):
        self.config_panel.set_controls_enabled(not busy)
        self.left_pane.set_controls_enabled(not busy)
        if busy:
            self.config_panel.set_status(message)
            self.status_bar.showMessage(message)
        else:
            self.config_panel.set_status("Ready")
            self.status_bar.showMessage("Ready")

    def run_reproducer(self, top_n):
        if not self.input_path or not self.output_dir:
            QMessageBox.warning(self, "Configuration Error", "Please select input file and output directory.")
            return

        self.set_busy(True, "Running Reproducer...")
        self.right_pane.logs_tab.log(f"Starting Reproducer with Top N={top_n}...")

        worker = Worker(
            run_reproducer_task,
            self.input_path,
            self.output_dir,
            top_n
        )
        worker.signals.progress.connect(self.on_progress)
        worker.signals.result.connect(self.on_reproducer_finished)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(self.on_task_complete)
        
        self.threadpool.start(worker)

    def run_predictor(self, top_n):
        if not self.input_path or not self.output_dir:
            QMessageBox.warning(self, "Configuration Error", "Please select input file and output directory.")
            return

        self.set_busy(True, "Running Predictor...")
        self.right_pane.logs_tab.log(f"Starting Predictor with Top Features={top_n}...")

        worker = Worker(
            run_injector_task,
            self.input_path,
            self.output_dir,
            top_n
        )
        worker.signals.progress.connect(self.on_progress)
        worker.signals.result.connect(self.on_predictor_finished)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(self.on_task_complete)
        
        self.threadpool.start(worker)

    def on_progress(self, msg):
        self.status_bar.showMessage(msg)
        self.config_panel.set_status(msg)
        self.right_pane.logs_tab.log(f"[Progress] {msg}")

    def on_reproducer_finished(self, images):
        self.right_pane.logs_tab.log("Reproducer completed successfully.")
        self.left_pane.reproducer_tab.update_outputs(images)
        self.right_pane.plots_tab.update_images(images)
        self.status_bar.showMessage("Reproducer completed.")

    def on_predictor_finished(self, result):
        images, metrics, top_features = result
        self.right_pane.logs_tab.log("Predictor completed successfully.")
        self.left_pane.predictor_tab.update_results(metrics, top_features)
        self.right_pane.plots_tab.update_images(images)
        self.status_bar.showMessage("Predictor completed.")

    def on_error(self, err):
        self.right_pane.logs_tab.log(f"Error: {err}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err}")
        self.config_panel.set_status("Error", is_error=True)

    def on_task_complete(self):
        self.set_busy(False)
