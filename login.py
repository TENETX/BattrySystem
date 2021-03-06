# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
import SQLb as s
from PyQt5.QtWidgets import QMessageBox
import mainwindow
import sys
import register
import change
people = "user"


class Ui_login(object):
    def setupUi(self, login):
        login.setObjectName("login")
        login.resize(495, 382)
        login.setStyleSheet("background-color: rgb(245, 245, 245);")
        login.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.button = QtWidgets.QWidget(login)
        self.button.setGeometry(QtCore.QRect(30, 280, 441, 91))
        self.button.setObjectName("button")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.button)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.yes = QtWidgets.QPushButton(self.button)
        self.yes.setMinimumSize(QtCore.QSize(150, 50))
        self.yes.setStyleSheet("QPushButton\n"
                               "{\n"
                               "    color:white;\n"
                               "    background-color:rgb(14 , 150 , 254);\n"
                               "    border-radius:5px;\n"
                               "    font:20px \"微软雅黑\";\n"
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
        self.yes.setObjectName("yes")
        self.horizontalLayout.addWidget(self.yes)
        spacerItem = QtWidgets.QSpacerItem(
            82, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.close = QtWidgets.QPushButton(self.button)
        self.close.setMinimumSize(QtCore.QSize(150, 50))
        self.close.setStyleSheet("QPushButton\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(14 , 150 , 254);\n"
                                 "    border-radius:5px;\n"
                                 "    font:20px \"微软雅黑\";\n"
                                 "}\n"
                                 "\n"
                                 ":hover\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(44 , 137 , 255);\n"
                                 "}\n"
                                 "\n"
                                 ":pressed\n"
                                 "{\n"
                                 "    color:white;\n"
                                 "    background-color:rgb(14 , 135 , 228);\n"
                                 "    padding-left:3px;\n"
                                 "    padding-top:3px;\n"
                                 "}")
        self.close.setObjectName("close")
        self.horizontalLayout.addWidget(self.close)
        self.first = QtWidgets.QWidget(login)
        self.first.setGeometry(QtCore.QRect(20, 100, 451, 71))
        self.first.setObjectName("first")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.first)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.first)
        self.label_2.setStyleSheet("font:15px \"微软雅黑\";")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.number = QtWidgets.QLineEdit(self.first)
        self.number.setMinimumSize(QtCore.QSize(0, 30))
        self.number.setStyleSheet("background:white;\n"
                                  "    padding-left:5px ;\n"
                                  "    padding-top:1px ;\n"
                                  "    border-bottom-left-radius:3px;\n"
                                  "    border-bottom-right-radius:3px;\n"
                                  "    border: 1px solid rgb(209 , 209 , 209);\n"
                                  "    border-top:transparent;")
        self.number.setObjectName("number")
        self.horizontalLayout_2.addWidget(self.number)
        self.create = QtWidgets.QPushButton(self.first)
        self.create.setStyleSheet("QPushButton\n"
                                  "{\n"
                                  "    color:rgb(38 , 133 , 227);\n"
                                  "    background-color:transparent;\n"
                                  "}\n"
                                  "\n"
                                  ":hover\n"
                                  "{\n"
                                  "    color:rgb(97 , 179 , 246);\n"
                                  "}\n"
                                  "\n"
                                  ":pressed\n"
                                  "{\n"
                                  "    color:rgb(0 , 109 , 176);\n"
                                  "}")
        self.create.setObjectName("create")
        self.horizontalLayout_2.addWidget(self.create)
        self.second = QtWidgets.QWidget(login)
        self.second.setGeometry(QtCore.QRect(20, 190, 451, 71))
        self.second.setObjectName("second")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.second)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.second)
        self.label_3.setStyleSheet("font:15px \"微软雅黑\";")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.secret = QtWidgets.QLineEdit(self.second)
        self.secret.setMinimumSize(QtCore.QSize(0, 30))
        self.secret.setToolTip("")
        self.secret.setStyleSheet("background:white;\n"
                                  "    padding-left:5px ;\n"
                                  "    padding-top:1px ;\n"
                                  "    border-bottom-left-radius:3px;\n"
                                  "    border-bottom-right-radius:3px;\n"
                                  "    border: 1px solid rgb(209 , 209 , 209);\n"
                                  "    border-top:transparent;")
        self.secret.setEchoMode(QtWidgets.QLineEdit.Password)
        self.secret.setObjectName("secret")
        self.horizontalLayout_3.addWidget(self.secret)
        self.change = QtWidgets.QPushButton(self.second)
        self.change.setStyleSheet("QPushButton\n"
                                  "{\n"
                                  "    color:rgb(38 , 133 , 227);\n"
                                  "    background-color:transparent;\n"
                                  "}\n"
                                  "\n"
                                  ":hover\n"
                                  "{\n"
                                  "    color:rgb(97 , 179 , 246);\n"
                                  "}\n"
                                  "\n"
                                  ":pressed\n"
                                  "{\n"
                                  "    color:rgb(0 , 109 , 176);\n"
                                  "}")
        self.change.setObjectName("forget")
        self.horizontalLayout_3.addWidget(self.change)
        self.retranslateUi(login)
        QtCore.QMetaObject.connectSlotsByName(login)
        self.yes.clicked.connect(lambda: self.yescon(login))
        self.close.clicked.connect(lambda: self.nocon(login))
        self.create.clicked.connect(lambda: self.createcon(login))
        self.change.clicked.connect(lambda: self.changecon(login))

    def retranslateUi(self, login):
        _translate = QtCore.QCoreApplication.translate
        login.setWindowTitle(_translate("login", "Form"))
        self.yes.setText(_translate("login", "确        定"))
        self.close.setText(_translate("login", "取        消"))
        self.label_2.setText(_translate("login", "账     户："))
        self.create.setText(_translate("login", "创建账户"))
        self.label_3.setText(_translate("login", "密     码："))
        self.change.setText(_translate("login", "修改密码"))

    def nocon(self, login):
        mainwindow.stopexplore = True
        login.close()

    def yescon(self, login):
        f = "SELECT passwd FROM `users` WHERE id = '" + self.number.text() + "'"
        s.createone()
        c = s.con()
        c.execute(f)
        values = c.fetchall()
        if values:
            if str(values[0][0]) == self.secret.text():
                print("success")
                global people
                people = self.number.text()
                login.close()
                mainwindow.showout()
            else:
                QMessageBox.critical(
                    None, "登陆失败", "密码或账号错误", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            QMessageBox.critical(
                None, "登陆失败", "请输入账号密码！", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

    def createcon(self, login):
        register.showout()

    def changecon(self, login):
        change.showout()


def showout():
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QMainWindow()
    ui = Ui_login()
    ui.setupUi(Form)
    Form.setWindowIcon(QIcon("./data/icon.png"))
    Form.setWindowTitle("登录")
    Form.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    showout()
