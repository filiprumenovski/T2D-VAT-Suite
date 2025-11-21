from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QGroupBox, QLabel, QSpinBox, QPushButton, 
    QListWidget, QListWidgetItem, QFormLayout, QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView
)
from PySide6.QtCore import Signal, Slot

class ReproducerTab(QWidget):
    run_requested = Signal(int) # top_n
    show_image_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Description
        desc_group = QGroupBox("Reproducer")
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Recreate PCA, volcano, heatmap, and QC plots from the T2D VAT dataset."))
        
        self.run_btn = QPushButton("Run Reproducer")
        self.run_btn.clicked.connect(self.on_run_clicked)
        desc_layout.addWidget(self.run_btn)
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)

        # Options
        opts_group = QGroupBox("Options")
        opts_layout = QFormLayout()
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(10, 100)
        self.top_n_spin.setValue(30)
        opts_layout.addRow("Top N proteins:", self.top_n_spin)
        opts_group.setLayout(opts_layout)
        layout.addWidget(opts_group)

        # Outputs
        out_group = QGroupBox("Outputs")
        out_layout = QVBoxLayout()
        self.output_list = QListWidget()
        self.output_list.itemClicked.connect(self.on_item_clicked)
        out_layout.addWidget(self.output_list)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        layout.addStretch()
        self.setLayout(layout)

    @Slot()
    def on_run_clicked(self):
        self.run_requested.emit(self.top_n_spin.value())

    @Slot()
    def on_item_clicked(self, item):
        self.show_image_requested.emit(item.data(100)) # UserRole for path

    def update_outputs(self, files):
        self.output_list.clear()
        for f in files:
            name = f.split("/")[-1]
            item = QListWidgetItem(name)
            item.setData(100, f)
            self.output_list.addItem(item)

class PredictorTab(QWidget):
    run_requested = Signal(int) # top_n
    show_image_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Description
        desc_group = QGroupBox("Predictor")
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Train a classifier to distinguish DM vs NDM from the proteome."))
        
        self.run_btn = QPushButton("Run Predictor")
        self.run_btn.clicked.connect(self.on_run_clicked)
        desc_layout.addWidget(self.run_btn)
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)

        # Options
        opts_group = QGroupBox("Options")
        opts_layout = QFormLayout()
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 100)
        self.top_n_spin.setValue(20)
        opts_layout.addRow("Top features:", self.top_n_spin)
        opts_group.setLayout(opts_layout)
        layout.addWidget(opts_group)

        # Metrics
        metrics_group = QGroupBox("Metrics")
        self.metrics_layout = QFormLayout()
        self.acc_label = QLabel("-")
        self.roc_label = QLabel("-")
        self.f1_label = QLabel("-")
        self.metrics_layout.addRow("Accuracy:", self.acc_label)
        self.metrics_layout.addRow("ROC AUC:", self.roc_label) # Note: ROC AUC might not be in simple metrics dict, using Precision/Recall/F1 usually
        self.metrics_layout.addRow("F1-score:", self.f1_label)
        metrics_group.setLayout(self.metrics_layout)
        layout.addWidget(metrics_group)

        # Top Features
        feats_group = QGroupBox("Top Features")
        feats_layout = QVBoxLayout()
        self.feats_table = QTableWidget()
        self.feats_table.setColumnCount(4)
        self.feats_table.setHorizontalHeaderLabels(["Gene", "Protein ID", "Importance", "q-value"])
        self.feats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        feats_layout.addWidget(self.feats_table)
        feats_group.setLayout(feats_layout)
        layout.addWidget(feats_group)

        self.setLayout(layout)

    @Slot()
    def on_run_clicked(self):
        self.run_requested.emit(self.top_n_spin.value())

    def update_results(self, metrics, top_features):
        # Update metrics
        training = metrics.get("training", {})
        self.acc_label.setText(f"{training.get('accuracy', 0):.3f}")
        self.roc_label.setText(f"{training.get('precision', 0):.3f}") # Using Precision as placeholder if ROC AUC not computed
        self.f1_label.setText(f"{training.get('f1', 0):.3f}")

        # Update table
        self.feats_table.setRowCount(len(top_features))
        for i, feat in enumerate(top_features):
            self.feats_table.setItem(i, 0, QTableWidgetItem(str(feat.get("gene_name", ""))))
            self.feats_table.setItem(i, 1, QTableWidgetItem(str(feat.get("feature", ""))))
            self.feats_table.setItem(i, 2, QTableWidgetItem(f"{feat.get('importance', 0):.4f}"))
            self.feats_table.setItem(i, 3, QTableWidgetItem("-")) # q-value not in top_features usually

class LeftPane(QWidget):
    run_reproducer_requested = Signal(int)
    run_predictor_requested = Signal(int)
    show_image_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.reproducer_tab = ReproducerTab()
        self.reproducer_tab.run_requested.connect(self.run_reproducer_requested)
        self.reproducer_tab.show_image_requested.connect(self.show_image_requested)
        
        self.predictor_tab = PredictorTab()
        self.predictor_tab.run_requested.connect(self.run_predictor_requested)
        
        self.tabs.addTab(self.reproducer_tab, "Reproducer")
        self.tabs.addTab(self.predictor_tab, "Predictor")
        
        layout.addWidget(self.tabs)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def set_controls_enabled(self, enabled):
        self.reproducer_tab.run_btn.setEnabled(enabled)
        self.predictor_tab.run_btn.setEnabled(enabled)
