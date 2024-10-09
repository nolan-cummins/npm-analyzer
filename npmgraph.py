# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'npmgraphAmgHOS.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QLabel,
    QSizePolicy, QWidget)

from pyqtgraph import PlotWidget

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(1040, 790)
        self.frame_4 = QFrame(Dialog)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setGeometry(QRect(10, 10, 1021, 771))
        self.frame_4.setFrameShape(QFrame.Shape.WinPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.frame_7 = QFrame(self.frame_4)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setGeometry(QRect(10, 10, 1001, 31))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setFrameShape(QFrame.Shape.WinPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Sunken)
        self.consoleOutputTitle = QLabel(self.frame_7)
        self.consoleOutputTitle.setObjectName(u"consoleOutputTitle")
        self.consoleOutputTitle.setGeometry(QRect(0, 0, 1001, 31))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.consoleOutputTitle.setFont(font)
        self.consoleOutputTitle.setFrameShape(QFrame.Shape.NoFrame)
        self.consoleOutputTitle.setFrameShadow(QFrame.Shadow.Raised)
        self.consoleOutputTitle.setTextFormat(Qt.TextFormat.RichText)
        self.consoleOutputTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.npmPlot = PlotWidget(self.frame_4)
        self.npmPlot.setObjectName(u"npmPlot")
        self.npmPlot.setGeometry(QRect(10, 50, 1001, 711))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.consoleOutputTitle.setText(QCoreApplication.translate("Dialog", u"Mean Velocity vs. Voltage", None))
    # retranslateUi

