from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget

from predictor import Predictor


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Homework")
        MainWindow.resize(414, 285)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.scoreButton = QtWidgets.QPushButton(self.centralWidget)
        self.scoreButton.setGeometry(QtCore.QRect(280, 120, 89, 25))
        self.scoreButton.setObjectName("scoreButton")
        self.resultLabel = QtWidgets.QLabel(self.centralWidget)
        self.resultLabel.setGeometry(QtCore.QRect(180, 203, 64, 64))
        self.resultLabel.setObjectName("resultLabel")
        self.widget = QtWidgets.QWidget(self.centralWidget)
        self.widget.setGeometry(QtCore.QRect(30, 30, 201, 171))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.tweetTextEdit = QtWidgets.QPlainTextEdit(self.widget)
        self.tweetTextEdit.setObjectName("tweetTextEdit")
        self.verticalLayout.addWidget(self.tweetTextEdit)
        MainWindow.setCentralWidget(self.centralWidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # non-ui
        self.setupEvents()
        self.predictor = Predictor("models/predictor.pkl")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.scoreButton.setText(_translate("MainWindow", "Score"))
        self.resultLabel.setText(_translate("MainWindow", "Result"))
        self.label.setText(_translate("MainWindow", "Your Tweet:"))

    def setupEvents(self):
        self.scoreButton.clicked.connect(self.score)

    def score(self):
        tweet = self.tweetTextEdit.toPlainText()
        result = self.predictor.predict([tweet])[0]
        myPixmap = QtGui.QPixmap("images/{}-icon.png".format("like" if result == "yes" else "dislike"))
        # myPixmap = myPixmap.scaled(self.resultLabel.size(), Qt.KeepAspectRatio)
        self.resultLabel.setPixmap(myPixmap)


if __name__ == '__main__':
    from nltk.corpus import sentiwordnet as swn
    happ = swn.senti_synset('سِيءَ.a.03')
    print(happ.pos_score())
