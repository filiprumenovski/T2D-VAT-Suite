# T2D-VAT Suite GUI

This is the graphical user interface for the T2D-VAT Suite, built with PySide6 and QML.

## Requirements

Ensure you have the GUI dependencies installed:

```bash
pip install ".[gui]"
# or
pip install -r requirements.txt
```

## Running the App

From the project root directory, run:

```bash
python -m apps.gui.main
```

## Features

- **Reproducer Tab**: Run the reproduction pipeline, view generated plots (QC, PCA, Volcano, Heatmap).
- **Injector Tab**: Run the ML pipeline, view metrics and ML-specific plots (ROC, Confusion Matrix, Feature Importance).
- **Threaded Execution**: Long-running tasks run in the background without freezing the UI.
