# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:/files/bbb/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap
import time
import mythread
import threading
import myyolo
import cv2 as cv
startexplore = False
pauseexplore = False
stopexplore = False


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(978, 600)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        MainWindow.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button = QtWidgets.QWidget(self.centralwidget)
        self.button.setGeometry(QtCore.QRect(20, 490, 531, 101))
        self.button.setObjectName("button")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.button)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.start = QtWidgets.QPushButton(self.button)
        self.start.setMinimumSize(QtCore.QSize(100, 50))
        self.start.setStyleSheet("QPushButton\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(14 , 150 , 254);\n"
                                 "    border-radius:5px;\n"
                                 "    font:25px \"微软雅黑\";\n"
                                 "}\n"
                                 "\n"
                                 "::hover\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(44 , 137 , 255);\n"
                                 "}\n"
                                 "\n"
                                 "::pressed\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(14 , 135 , 228);\n"
                                 "    padding-left:3px;\n"
                                 "    padding-top:3px;\n"
                                 "}")
        self.start.setObjectName("start")
        self.horizontalLayout.addWidget(self.start)
        self.stop = QtWidgets.QPushButton(self.button)
        self.stop.setMinimumSize(QtCore.QSize(100, 50))
        self.stop.setStyleSheet("QPushButton\n"
                                "{\n"
                                "    color:white;\n"
                                "    background-color:rgb(14 , 150 , 254);\n"
                                "    border-radius:5px;\n"
                                "    font:25px \"微软雅黑\";\n"
                                "}\n"
                                "\n"
                                "::hover\n"
                                "{\n"
                                "    color:white;\n"
                                "    background-color:rgb(44 , 137 , 255);\n"
                                "}\n"
                                "\n"
                                "::pressed\n"
                                "{\n"
                                "    color:white;\n"
                                "    background-color:rgb(14 , 135 , 228);\n"
                                "    padding-left:3px;\n"
                                "    padding-top:3px;\n"
                                "}")
        self.stop.setObjectName("stop")
        self.horizontalLayout.addWidget(self.stop)
        self.close = QtWidgets.QPushButton(self.button)
        self.close.setMinimumSize(QtCore.QSize(100, 50))
        self.close.setStyleSheet("QPushButton\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(14 , 150 , 254);\n"
                                 "    border-radius:5px;\n"
                                 "    font:25px \"微软雅黑\";\n"
                                 "}\n"
                                 "\n"
                                 "::hover\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(44 , 137 , 255);\n"
                                 "}\n"
                                 "\n"
                                 "::pressed\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(14 , 135 , 228);\n"
                                 "    padding-left:3px;\n"
                                 "    padding-top:3px;\n"
                                 "}")
        self.close.setObjectName("close")
        self.horizontalLayout.addWidget(self.close)
        self.single = QtWidgets.QLabel(self.centralwidget)
        self.single.setGeometry(QtCore.QRect(590, 10, 261, 231))
        self.single.setStyleSheet("background-color: rgb(243, 243, 243);")
        self.single.setText("")
        self.single.setObjectName("single")
        self.message = QtWidgets.QWidget(self.centralwidget)
        self.message.setGeometry(QtCore.QRect(580, 250, 141, 181))
        self.message.setObjectName("message")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.message)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.zhengfan = QtWidgets.QLabel(self.message)
        self.zhengfan.setStyleSheet("font: 14pt \"黑体\";\n"
                                    "color: rgb(0, 170, 255);")
        self.zhengfan.setObjectName("zhengfan")
        self.verticalLayout.addWidget(self.zhengfan)
        self.liushuixian = QtWidgets.QLabel(self.message)
        self.liushuixian.setStyleSheet("font: 14pt \"黑体\";\n"
                                       "color: rgb(0, 170, 255);")
        self.liushuixian.setObjectName("liushuixian")
        self.verticalLayout.addWidget(self.liushuixian)
        self.quexian = QtWidgets.QLabel(self.message)
        self.quexian.setStyleSheet("font: 14pt \"黑体\";\n"
                                   "color: rgb(0, 170, 255);")
        self.quexian.setObjectName("quexian")
        self.verticalLayout.addWidget(self.quexian)
        self.resultm = QtWidgets.QWidget(self.centralwidget)
        self.resultm.setGeometry(QtCore.QRect(740, 250, 181, 181))
        self.resultm.setObjectName("resultm")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.resultm)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.m1 = QtWidgets.QLabel(self.resultm)
        self.m1.setStyleSheet("font: 14pt \"黑体\";")
        self.m1.setText("")
        self.m1.setObjectName("m1")
        self.verticalLayout_2.addWidget(self.m1)
        self.m2 = QtWidgets.QLabel(self.resultm)
        self.m2.setStyleSheet("font: 14pt \"黑体\";")
        self.m2.setText("")
        self.m2.setObjectName("m2")
        self.verticalLayout_2.addWidget(self.m2)
        self.m3 = QtWidgets.QLabel(self.resultm)
        self.m3.setStyleSheet("font: 14pt \"黑体\";")
        self.m3.setText("")
        self.m3.setObjectName("m3")
        self.verticalLayout_2.addWidget(self.m3)
        self.now = QtWidgets.QWidget(self.centralwidget)
        self.now.setGeometry(QtCore.QRect(580, 470, 141, 111))
        self.now.setObjectName("now")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.now)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.zhanghu = QtWidgets.QLabel(self.now)
        self.zhanghu.setStyleSheet("font: 14pt \"黑体\";\n"
                                   "color: rgb(255, 170, 0);")
        self.zhanghu.setObjectName("zhanghu")
        self.verticalLayout_3.addWidget(self.zhanghu)
        self.shijian = QtWidgets.QLabel(self.now)
        self.shijian.setStyleSheet("font: 14pt \"黑体\";\n"
                                   "color: rgb(255, 170, 0);")
        self.shijian.setObjectName("shijian")
        self.verticalLayout_3.addWidget(self.shijian)
        self.results = QtWidgets.QWidget(self.centralwidget)
        self.results.setGeometry(QtCore.QRect(740, 470, 181, 111))
        self.results.setObjectName("results")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.results)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.s1 = QtWidgets.QLabel(self.results)
        self.s1.setStyleSheet("font: 14pt \"黑体\";")
        self.s1.setText("")
        self.s1.setObjectName("s1")
        self.verticalLayout_4.addWidget(self.s1)
        self.s2 = QtWidgets.QLabel(self.results)
        self.s2.setStyleSheet("font: 10pt \"黑体\";")
        self.s2.setText("")
        self.s2.setObjectName("s2")
        self.verticalLayout_4.addWidget(self.s2)
        self.video = QtWidgets.QLabel(self.centralwidget)
        self.video.setGeometry(QtCore.QRect(10, 10, 551, 461))
        self.video.setStyleSheet("background-color: rgb(243, 243, 243);")
        self.video.setText("")
        self.video.setObjectName("video")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.start.clicked.connect(lambda: self.startex())
        self.stop.clicked.connect(lambda: self.pauseex())
        self.close.clicked.connect(lambda: self.stopex())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.start.setText(_translate("MainWindow", "开始"))
        self.stop.setText(_translate("MainWindow", "暂停"))
        self.close.setText(_translate("MainWindow", "停止"))
        self.zhengfan.setText(_translate("MainWindow", "正反面："))
        self.liushuixian.setText(_translate("MainWindow", "盖帽编号："))
        self.quexian.setText(_translate("MainWindow", "缺陷类型："))
        self.zhanghu.setText(_translate("MainWindow", "用户名："))
        self.shijian.setText(_translate("MainWindow", "当前时间："))

    def startex(self):
        global startexplore
        startexplore = True
        t4 = threading.Thread(target=self.singleout)
        t5 = threading.Thread(target=self.wholeout)
        t4.start()
        t5.start()

    def pauseex(self):
        global pauseexplore
        pauseexplore = not pauseexplore

    def stopex(self):
        global stopexplore
        stopexplore = True

    def singleout(self):
        while not stopexplore:
            if not myyolo.oneround.empty() and not pauseexplore:
                rr = myyolo.oneround.get()
                url = rr[0]
                img = cv.imread(url)
                label = rr[1]
                temp = QtGui.QImage(
                    img, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pixmap_temp = QPixmap.fromImage(temp)
                self.single.setPixmap(pixmap_temp)
                self.single.setScaledContents(True)
                self.m1.setText("正面")
                self.m1.repaint()
                self.m2.setText(str(label[2]))
                self.m2.repaint()
                self.m3.setText(label[1])
                self.m3.repaint()
                self.s1.setText(label[0])
                self.s1.repaint()
                self.s2.setText(time.strftime("%Y-%m-%d %X", time.localtime()))
                self.s2.repaint()

    def wholeout(self):
        while not stopexplore:
            if not mythread.whole.empty() and not pauseexplore:
                img = mythread.whole.get()
                temp = QtGui.QImage(
                    img, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pixmap_temp = QPixmap.fromImage(temp)
                self.video.setPixmap(pixmap_temp)
                self.video.setScaledContents(True)


def showout():
    form2 = QtWidgets.QDialog()
    ui = Ui_MainWindow()
    ui.setupUi(form2)
    form2.setWindowTitle("电池盖帽生产线检测系统")
    form2.setWindowIcon(QIcon("./data/icon.png"))
    form2.show()
    form2.exec_()
