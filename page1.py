from PyQt5 import QtCore, QtGui, QtWidgets
from INDOOR_OBJECT_DETECT import indoor_detect
from OUTDOOR_OBJECT_DETECT import outdoor_detect
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 587)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 801, 591))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("583_main.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 0, 681, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(252, 221, 186);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(60, 480, 211, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: rgb(252, 221, 186);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(350, 480, 221, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color: rgb(252, 221, 186);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(580, 280, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("background-color: rgb(252, 221, 186);")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(580, 330, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet("background-color: rgb(252, 221, 186);")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(580, 380, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setStyleSheet("background-color: rgb(252, 221, 186);")
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(580, 430, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setStyleSheet("background-color: rgb(252, 221, 186);")
        self.pushButton_7.setObjectName("pushButton_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.pushButton_2.clicked.connect(self.indoor_detect1)
        self.pushButton_3.clicked.connect(self.outdoor_detect1)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "BLIND OBJECT RECOGNITION"))
        self.pushButton_2.setText(_translate("MainWindow", "INDOOR \nDETECT"))
        self.pushButton_3.setText(_translate("MainWindow", "OUTDOOR \nDETECT"))
        self.pushButton_4.setText(_translate("MainWindow", "PRATHYUSHA.K.G"))
        self.pushButton_5.setText(_translate("MainWindow", "SHANMUGAPRIYA.S"))
        self.pushButton_6.setText(_translate("MainWindow", "YAMINI.D.K"))


    def indoor_detect1(self):
        print("yes")
        indoor_detect()

    def outdoor_detect1(self):
        outdoor_detect()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
