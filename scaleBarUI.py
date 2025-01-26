# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'scaleBarUIQEtZUW.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QGridLayout, QLabel, QSizePolicy,
    QSlider, QSpinBox, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(378, 271)
        self.scaleBarButton = QDialogButtonBox(Dialog)
        self.scaleBarButton.setObjectName(u"scaleBarButton")
        self.scaleBarButton.setGeometry(QRect(30, 240, 341, 32))
        self.scaleBarButton.setOrientation(Qt.Orientation.Horizontal)
        self.scaleBarButton.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)
        self.formLayoutWidget = QWidget(Dialog)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(9, 13, 361, 221))
        self.gridLayout = QGridLayout(self.formLayoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.scaleFactor_slider = QSlider(self.formLayoutWidget)
        self.scaleFactor_slider.setObjectName(u"scaleFactor_slider")
        self.scaleFactor_slider.setMaximumSize(QSize(100, 16777215))
        self.scaleFactor_slider.setOrientation(Qt.Orientation.Horizontal)
        self.scaleFactor_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout.addWidget(self.scaleFactor_slider, 1, 2, 1, 1)

        self.label_2 = QLabel(self.formLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.label_6 = QLabel(self.formLayoutWidget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 7, 0, 1, 1)

        self.label_3 = QLabel(self.formLayoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.x_slider = QSlider(self.formLayoutWidget)
        self.x_slider.setObjectName(u"x_slider")
        self.x_slider.setMaximumSize(QSize(100, 16777215))
        self.x_slider.setOrientation(Qt.Orientation.Horizontal)
        self.x_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout.addWidget(self.x_slider, 5, 2, 1, 1)

        self.label_7 = QLabel(self.formLayoutWidget)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)

        self.label_4 = QLabel(self.formLayoutWidget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)

        self.width_slider = QSlider(self.formLayoutWidget)
        self.width_slider.setObjectName(u"width_slider")
        self.width_slider.setMaximumSize(QSize(100, 16777215))
        self.width_slider.setOrientation(Qt.Orientation.Horizontal)
        self.width_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout.addWidget(self.width_slider, 3, 2, 1, 1)

        self.label_5 = QLabel(self.formLayoutWidget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 6, 0, 1, 1)

        self.label = QLabel(self.formLayoutWidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.length_input = QDoubleSpinBox(self.formLayoutWidget)
        self.length_input.setObjectName(u"length_input")
        self.length_input.setMaximum(10000000000000.000000000000000)

        self.gridLayout.addWidget(self.length_input, 2, 1, 1, 1)

        self.scaleFactor_input = QDoubleSpinBox(self.formLayoutWidget)
        self.scaleFactor_input.setObjectName(u"scaleFactor_input")
        self.scaleFactor_input.setMaximum(10000000000000.000000000000000)

        self.gridLayout.addWidget(self.scaleFactor_input, 1, 1, 1, 1)

        self.x_input = QDoubleSpinBox(self.formLayoutWidget)
        self.x_input.setObjectName(u"x_input")
        self.x_input.setMaximum(10000000000000.000000000000000)

        self.gridLayout.addWidget(self.x_input, 5, 1, 1, 1)

        self.y_input = QDoubleSpinBox(self.formLayoutWidget)
        self.y_input.setObjectName(u"y_input")
        self.y_input.setMaximum(10000000000000.000000000000000)

        self.gridLayout.addWidget(self.y_input, 6, 1, 1, 1)

        self.divisions_input = QSpinBox(self.formLayoutWidget)
        self.divisions_input.setObjectName(u"divisions_input")
        self.divisions_input.setMaximum(1000000000)

        self.gridLayout.addWidget(self.divisions_input, 4, 1, 1, 1)

        self.width_input = QDoubleSpinBox(self.formLayoutWidget)
        self.width_input.setObjectName(u"width_input")
        self.width_input.setMaximum(10000000000000.000000000000000)

        self.gridLayout.addWidget(self.width_input, 3, 1, 1, 1)

        self.fontScale_input = QDoubleSpinBox(self.formLayoutWidget)
        self.fontScale_input.setObjectName(u"fontScale_input")
        self.fontScale_input.setMaximum(10000000000000.000000000000000)

        self.gridLayout.addWidget(self.fontScale_input, 7, 1, 1, 1)

        self.fontScale_slider = QSlider(self.formLayoutWidget)
        self.fontScale_slider.setObjectName(u"fontScale_slider")
        self.fontScale_slider.setMaximumSize(QSize(100, 16777215))
        self.fontScale_slider.setOrientation(Qt.Orientation.Horizontal)
        self.fontScale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout.addWidget(self.fontScale_slider, 7, 2, 1, 1)

        self.y_slider = QSlider(self.formLayoutWidget)
        self.y_slider.setObjectName(u"y_slider")
        self.y_slider.setMaximumSize(QSize(100, 16777215))
        self.y_slider.setOrientation(Qt.Orientation.Horizontal)
        self.y_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout.addWidget(self.y_slider, 6, 2, 1, 1)

        self.divisions_slider = QSlider(self.formLayoutWidget)
        self.divisions_slider.setObjectName(u"divisions_slider")
        self.divisions_slider.setMaximumSize(QSize(100, 16777215))
        self.divisions_slider.setOrientation(Qt.Orientation.Horizontal)
        self.divisions_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout.addWidget(self.divisions_slider, 4, 2, 1, 1)

        self.length_slider = QSlider(self.formLayoutWidget)
        self.length_slider.setObjectName(u"length_slider")
        self.length_slider.setMaximumSize(QSize(100, 16777215))
        self.length_slider.setOrientation(Qt.Orientation.Horizontal)
        self.length_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout.addWidget(self.length_slider, 2, 2, 1, 1)


        self.retranslateUi(Dialog)
        self.scaleBarButton.accepted.connect(Dialog.accept)
        self.scaleBarButton.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"Length (um)", None))
        self.label_6.setText(QCoreApplication.translate("Dialog", u"Font Scale", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"Divisions", None))
        self.label_7.setText(QCoreApplication.translate("Dialog", u"Width (pix)", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"X (pix)", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"Y (pix)", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"Scale Factor (pix/um)", None))
    # retranslateUi

