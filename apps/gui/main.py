import sys
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QUrl

from apps.gui.viewmodel import ViewModel

def main():
    app = QGuiApplication(sys.argv)
    app.setOrganizationName("T2D Lab")
    app.setOrganizationDomain("t2dlab.org")
    app.setApplicationName("T2D-VAT Suite")

    engine = QQmlApplicationEngine()
    
    # Create and register ViewModel
    view_model = ViewModel()
    engine.rootContext().setContextProperty("viewModel", view_model)

    # Load QML
    current_dir = Path(__file__).parent
    qml_file = current_dir / "resources" / "main.qml"
    
    engine.load(QUrl.fromLocalFile(str(qml_file)))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
