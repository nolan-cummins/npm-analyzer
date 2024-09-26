# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_embossAdjustments_v2iGysme.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFrame,
    QHBoxLayout, QLabel, QSizePolicy, QSlider,
    QSpinBox, QVBoxLayout, QWidget)

class Ui_Adjust(object):
    def setupUi(self, Adjust):
        if not Adjust.objectName():
            Adjust.setObjectName(u"Adjust")
        Adjust.resize(370, 154)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Adjust.sizePolicy().hasHeightForWidth())
        Adjust.setSizePolicy(sizePolicy)
        Adjust.setMinimumSize(QSize(370, 154))
        Adjust.setMaximumSize(QSize(370, 154))
        self.verticalLayoutWidget_3 = QWidget(Adjust)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(10, 3, 351, 141))
        self.verticalLayout_5 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(-1, 0, -1, -1)
        self.blurTitle = QLabel(self.verticalLayoutWidget_3)
        self.blurTitle.setObjectName(u"blurTitle")
        font = QFont()
        font.setPointSize(12)
        self.blurTitle.setFont(font)
        self.blurTitle.setFrameShape(QFrame.Shape.NoFrame)
        self.blurTitle.setFrameShadow(QFrame.Shadow.Raised)
        self.blurTitle.setTextFormat(Qt.TextFormat.RichText)
        self.blurTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_6.addWidget(self.blurTitle)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.blurToggle = QCheckBox(self.verticalLayoutWidget_3)
        self.blurToggle.setObjectName(u"blurToggle")
        self.blurToggle.setFont(font)

        self.horizontalLayout_4.addWidget(self.blurToggle)

        self.blurSlider = QSlider(self.verticalLayoutWidget_3)
        self.blurSlider.setObjectName(u"blurSlider")
        self.blurSlider.setEnabled(False)
        self.blurSlider.setMaximum(100)
        self.blurSlider.setOrientation(Qt.Orientation.Horizontal)
        self.blurSlider.setInvertedAppearance(False)
        self.blurSlider.setInvertedControls(False)
        self.blurSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.blurSlider.setTickInterval(10)

        self.horizontalLayout_4.addWidget(self.blurSlider)

        self.blurValue = QSpinBox(self.verticalLayoutWidget_3)
        self.blurValue.setObjectName(u"blurValue")
        self.blurValue.setEnabled(False)
        self.blurValue.setMinimumSize(QSize(100, 0))
        self.blurValue.setFont(font)
        self.blurValue.setMaximum(100)
        self.blurValue.setDisplayIntegerBase(10)

        self.horizontalLayout_4.addWidget(self.blurValue)


        self.verticalLayout_6.addLayout(self.horizontalLayout_4)


        self.verticalLayout_5.addLayout(self.verticalLayout_6)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.dilationTitle = QLabel(self.verticalLayoutWidget_3)
        self.dilationTitle.setObjectName(u"dilationTitle")
        self.dilationTitle.setFont(font)
        self.dilationTitle.setFrameShape(QFrame.Shape.NoFrame)
        self.dilationTitle.setFrameShadow(QFrame.Shadow.Raised)
        self.dilationTitle.setTextFormat(Qt.TextFormat.RichText)
        self.dilationTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_7.addWidget(self.dilationTitle)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.dilationToggle = QCheckBox(self.verticalLayoutWidget_3)
        self.dilationToggle.setObjectName(u"dilationToggle")
        self.dilationToggle.setFont(font)

        self.horizontalLayout_5.addWidget(self.dilationToggle)

        self.dilationSlider = QSlider(self.verticalLayoutWidget_3)
        self.dilationSlider.setObjectName(u"dilationSlider")
        self.dilationSlider.setEnabled(False)
        self.dilationSlider.setMinimum(0)
        self.dilationSlider.setMaximum(5)
        self.dilationSlider.setPageStep(1)
        self.dilationSlider.setOrientation(Qt.Orientation.Horizontal)
        self.dilationSlider.setInvertedAppearance(False)
        self.dilationSlider.setInvertedControls(False)
        self.dilationSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.dilationSlider.setTickInterval(1)

        self.horizontalLayout_5.addWidget(self.dilationSlider)

        self.dilationValue = QSpinBox(self.verticalLayoutWidget_3)
        self.dilationValue.setObjectName(u"dilationValue")
        self.dilationValue.setEnabled(False)
        self.dilationValue.setMinimumSize(QSize(100, 0))
        self.dilationValue.setFont(font)
        self.dilationValue.setMinimum(0)
        self.dilationValue.setMaximum(5)
        self.dilationValue.setDisplayIntegerBase(10)

        self.horizontalLayout_5.addWidget(self.dilationValue)


        self.verticalLayout_7.addLayout(self.horizontalLayout_5)


        self.verticalLayout_5.addLayout(self.verticalLayout_7)


        self.retranslateUi(Adjust)

        QMetaObject.connectSlotsByName(Adjust)
    # setupUi

    def retranslateUi(self, Adjust):
        Adjust.setWindowTitle(QCoreApplication.translate("Adjust", u"Dialog", None))
        self.blurTitle.setText(QCoreApplication.translate("Adjust", u"Gaussian Blur", None))
        self.blurToggle.setText("")
        self.dilationTitle.setText(QCoreApplication.translate("Adjust", u"Dilation", None))
        self.dilationToggle.setText("")
    # retranslateUi

