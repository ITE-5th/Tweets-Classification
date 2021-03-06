from PyQt5 import QtWidgets
import sys
from ui import Ui_MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    with open("qdarkstyle/style.qss") as f:
        app.setStyleSheet(f.read())
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setWindowTitle("Homework")
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
