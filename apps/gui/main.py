import sys
import os
from PySide6.QtWidgets import QApplication
from apps.gui.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("T2D-VAT Suite")
    app.setOrganizationName("T2D-VAT")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
