import sys
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
import pyqtgraph as pg
import numpy as np
import pandas as pd
from time import *
import concurrent.futures
import os
import cv2
import traceback
import ctypes
import warnings
import traceback
from collections import defaultdict
import inspect

myappid = 'nil.npm.pyqt.2' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

from importlib import reload
import json

import ui_main
reload(ui_main)
from ui_main import *

import npmgraph
reload(npmgraph)
from npmgraph import *

import scaleBarUI
reload(scaleBarUI)
from scaleBarUI import Ui_Dialog as ScaleBar_Dialog

from scaleBar import scaleBar

videoFormats = [
    "avi",
    "mp4",
    "mov",
    "mkv",
    "wmv",
    "flv",
    "mpeg",
    "mpg",
]

data=None
videoData=None

def findAngleDif(center, corner1, box2):
    dists=[]
    for corner in box2:
        dists.append(np.linalg.norm(corner1-corner))
    nearest_index=np.argmin(dists)
    nearest_point=box2[nearest_index]
    dx_n = nearest_point[0] - center[0]
    dy_n = nearest_point[1] - center[1]
    angle_n = np.degrees(np.arctan2(dy_n, dx_n))

    dx_o = corner1[0] - center[0]
    dy_o = corner1[1] - center[1]
    angle_o = np.degrees(np.arctan2(dy_o, dx_o))

    angle_diff = (angle_n - angle_o + 180) % 360 - 180
    
    return nearest_point, angle_diff    

def calculateAngle(box2D, new_corner, pixToUm, reference_angle=0):
    """
    Calculate a stable orientation angle for a box2D object based on a constant corner.

    Args:
        box2D: The box2D object (center, size, angle).
        new_corner: The consistent corner of the box2D, regardless of rotation.
        reference_angle: The reference axis angle in degrees (0 = x-axis, 90 = y-axis).

    Returns:
        orientation_angle: The stable orientation angle of the box with respect to the reference axis.
    """
    center, _, _ = box2D
    width, height = box2D[1]

    corners = np.array(cv2.boxPoints(box2D))

    distances = np.linalg.norm(corners - new_corner, axis=1)
    closest_corner = corners[np.argsort(distances)[1]]
    
    midpoint = (new_corner + closest_corner) / 2

    points = np.array([midpoint, center])
    dx = midpoint[0] - center[0]
    dy = midpoint[1] - center[1]
    angle = np.abs(np.abs((np.degrees(np.arctan2(dy, dx)) + 180) % 360 - 180) - 180)
    if angle > 90:
        angle = np.abs(angle - 180)

    return angle, midpoint, center

class Arrow:
    def __init__(self, start_point, end_point, color=(0, 255, 0), thickness=2, tipLength=0.3):
        self.start_point = start_point
        self.end_point = end_point
        self.color = color
        self.thickness = thickness
        self.tipLength = tipLength

    def draw(self, image):
        # Draw the arrow on the provided image
        cv2.arrowedLine(image, self.start_point, self.end_point, self.color, self.thickness, tipLength=self.tipLength)

def random_neon_color():

    primary = np.random.choice([0, 1, 2])
    
    color = [0, 0, 0]
    color[primary] = np.random.randint(200, 256)
    
    secondary_channels = [i for i in range(3) if i != primary]
    for channel in secondary_channels:
        color[channel] = np.random.randint(0, 100)
        

    return color

def getMaxThreads():
    # Get the number of available CPUs
    numCPUs = os.cpu_count()
    
    # Default maximum threads in ThreadPoolExecutor
    maxThreads = numCPUs * 5 if numCPUs else 1
    
    return maxThreads

def resizeVideos(rows, cols, width, height, layout): # resize all videos to fit
    videos=[]
    for row in range(rows): 
        for col in range(cols):
            widget = layout.itemAtPosition(row, col)
            if widget is not None:
                video = widget.widget()
                video.setMaximumSize(width, height)
                video.setIconSize(QtCore.QSize(width, height))
                video.setStyleSheet("border: none; background: transparent;")
                videos.append(video)
    return videos

def clearVideos(layout):
    for i in reversed(range(layout.count())):
        widget = layout.itemAt(i).widget()
        if widget is not None:
            widget.setParent(None)

def cv2FrameToPixmap(frame):
    # Convert BGR (OpenCV format) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a QImage
    height, width, channel = frame.shape
    bytesPerLine = 3 * width
    qImage = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(qImage).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap

class WorkerSignals(QObject): # Source: https://www.pythonguis.com/tutorials/multithreading-pyside6-applications-qthreadpool/
    '''
    Defines the signals available from a running worker thread.
    
    Supported signals are:
    
    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything
    '''

    error = Signal(tuple)
    result = Signal(object)
    
class VideoWorker(QRunnable): # Source: https://www.pythonguis.com/tutorials/multithreading-pyside6-applications-qthreadpool/
    def __init__(self, mutex, fn, *args, **kwargs):
        super(VideoWorker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.mutex = mutex

        # Add the callback to our kwargs
        self.kwargs['resultCallback'] = self.signals.result

        # run flag
        self._running = True

    def stop(self):
        print('STOP')
        self.running = False # stop flag
    
    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        if self._running:
            # Retrieve args/kwargs here; and fire processing using them
            try:
                result = self.fn(*self.args, **self.kwargs)
            except:
                traceback.print_exc()
                exctype, value = sys.exc_info()[:2]
                self.signals.error.emit((exctype, value, traceback.format_exc()))
            else:
                self.mutex.lock()
                try:
                    self.signals.result.emit(result)  # Return the result of the processing
                finally:
                    self.mutex.unlock()       

class ScaleBar(QDialog, ScaleBar_Dialog): # save position dialog box
    valuesUpdated = Signal(float, float, int, int, int, int, float)
    
    def __init__(self, frameHeight, frameWidth, pixToUm, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle("Scale Bar")
        self.pixToUm = pixToUm

        # Set slider ranges
        self.scaleFactor_slider.setRange(0, 1000)
        self.length_slider.setRange(0, int(frameHeight*self.pixToUm))
        self.width_slider.setRange(0, frameWidth)
        self.x_slider.setRange(0, frameWidth-5)
        self.y_slider.setRange(0, frameHeight-5)
        self.divisions_slider.setRange(0, 100)
        self.fontScale_slider.setRange(0, 10)

        # Connect sliders and spinboxes
        self.scaleFactor_slider.valueChanged.connect(self.scaleFactor_input.setValue)
        self.scaleFactor_input.valueChanged.connect(self.scaleFactor_slider.setValue)

        self.length_slider.valueChanged.connect(self.length_input.setValue)
        self.length_input.valueChanged.connect(self.length_slider.setValue)

        self.width_slider.valueChanged.connect(self.width_input.setValue)
        self.width_input.valueChanged.connect(self.width_slider.setValue)

        self.x_slider.valueChanged.connect(self.x_input.setValue)
        self.x_input.valueChanged.connect(self.x_slider.setValue)

        self.y_slider.valueChanged.connect(self.y_input.setValue)
        self.y_input.valueChanged.connect(self.y_slider.setValue)

        self.divisions_slider.valueChanged.connect(self.divisions_input.setValue)
        self.divisions_input.valueChanged.connect(self.divisions_slider.setValue)

        self.fontScale_slider.valueChanged.connect(self.fontScale_input.setValue)
        self.fontScale_input.valueChanged.connect(self.fontScale_slider.setValue)    

        self.scaleFactor_input.setValue(self.pixToUm)
        self.length_input.setValue(50)
        self.width_input.setValue(30)
        self.x_input.setValue(5)
        self.y_input.setValue(5)
        self.divisions_input.setValue(5)
        self.fontScale_input.setValue(1.5)

        def set_fixed_ticks(slider, num_ticks):
            range_min = slider.minimum()
            range_max = slider.maximum()
            tick_interval = (range_max - range_min) / (num_ticks - 1)
            slider.setTickInterval(int(tick_interval))
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        # Apply the fixed tick count
        set_fixed_ticks(self.scaleFactor_slider, 10)
        set_fixed_ticks(self.length_slider, 10)
        set_fixed_ticks(self.width_slider, 10)
        set_fixed_ticks(self.x_slider, 10)
        set_fixed_ticks(self.y_slider, 10)
        set_fixed_ticks(self.divisions_slider, 10)
        set_fixed_ticks(self.fontScale_slider, 10)

        
        self.scaleBarButton.accepted.connect(self.send_values)

    def setFrameSize(self, frameSize):
        height, width = frameSize
        self.length_slider.setRange(0, int(height*self.pixToUm))
        self.width_slider.setRange(0, width)
        self.x_slider.setRange(0, width-5)
        self.y_slider.setRange(0, height-5)
    
    def send_values(self):
        self.valuesUpdated.emit(
            self.scaleFactor_input.value(),
            self.length_input.value(),
            int(self.width_input.value()),
            int(self.x_input.value()),
            int(self.y_input.value()),
            int(self.divisions_input.value()),
            self.fontScale_input.value()
        )

class Dialog(QDialog, Ui_Dialog): # save position dialog box
    def __init__(self, data, parent=None):
        QDialog.__init__(self, parent)
        self.data = data
        self.setupUi(self)
        self.setWindowTitle("NPM Graph")
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.npmPlot.showGrid(x=True, y=True)
        self.npmPlot.setLabel("left", "Velocity (um/s)")
        self.npmPlot.setLabel("bottom", "Voltage (V)")
        self.npmPlot.setLimits(xMin=-5, xMax=5, yMin=-10, yMax=10)
        self.npmPlot.setXRange(-5, 5)
        self.npmPlot.setYRange(-10, 10)
        self.curve = self.npmPlot.plot([0], [0], pen='r', symbol=None, name='Mean Velocity')
        self.errorBars = pg.ErrorBarItem()
        self.npmPlot.addItem(self.errorBars)

    def updatePlot(self, data):
        if data:
            try:
                voltage, velocity, error = zip(*data) # unpack data
                self.curve.setData(voltage, velocity) # update data points
                
                self.errorBars.setData(x=np.array(voltage), # update the error bars
                                        y=np.array(velocity), 
                                        top=np.array(error), 
                                        bottom=np.array(error),
                                        beam=0.5)
            except Exception as e:
                err = traceback.format_exc()
                print(err)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow): # main window
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initialized = False
        self.setupUi(self)
        self.mutex = QMutex() # locking threads
        self.rectBufferMutex = QMutex() # locking threads
        
        # setup threads
        self.threadPool = QtCore.QThreadPool()
        self.threadPool.setMaxThreadCount(getMaxThreads())
        self.videoConnections = []
        
        # set icon and title
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.setWindowTitle("NPM Video Analyzer")

        self.videoFiles = [] # store file paths
        
        self.actionOpen_Video.triggered.connect(self.openFile)  # Menubar "Open"
        self.actionSave.triggered.connect(self.exportVideo) # Menubar "Export"
        self.actionClear.triggered.connect(self.onClear) # Menubar "Clear"
        self.actionEdit.triggered.connect(self.onEditScaleBar)
        self.actionCollate_Data_NPM.triggered.connect(self.onCollateData) # Menubar "Collate by Voltage (NPM Only)"
        self.NPMPlot = Dialog(None)

        # connect sliders
        self.contourValue.valueChanged.connect(self.embossFunction)
        self.contrastValue.valueChanged.connect(self.contrastFunction)
        self.thresholdValue.valueChanged.connect(self.thresholdFunction)
        
        self.frameDiffValue.valueChanged.connect(self.onFrameDifferencing)
        self.frameDiffToggle.toggled.connect(self.onFrameDifferencing)
        
        self.subBackValue.valueChanged.connect(self.subtractBackgroundFunction)
        self.subBackToggle.toggled.connect(self.subtractBackgroundFunction)
        self.subBackMethod.currentIndexChanged.connect(self.subtractBackgroundFunction)
        
        self.consecutiveFramesValue.valueChanged.connect(self.onConsecutiveFramesValue)
        self.record.toggled.connect(self.onRecord)
        self.contDetectSlider.valueChanged.connect(self.contDetectValue.setValue)
        self.contDetectValue.valueChanged.connect(self.contDetectSlider.setValue)
        self.contDetectValue.valueChanged.connect(self.onPersistence)
        
        self.blurSlider.valueChanged.connect(self.blurValue.setValue)
        self.blurValue.valueChanged.connect(self.blurSlider.setValue)
        self.blurToggle.toggled.connect(self.blurValue.setEnabled)
        self.blurToggle.toggled.connect(self.blurSlider.setEnabled)
        self.blurToggle.toggled.connect(self.onBlur)
        
        self.dilationToggle.toggled.connect(self.dilationSlider.setEnabled)
        self.dilationToggle.toggled.connect(self.onDilate)
        self.dilationValue.valueChanged.connect(self.dilationSlider.setValue)
        self.dilationSlider.valueChanged.connect(self.dilationValue.setValue)
        self.dilationToggle.toggled.connect(self.dilationValue.setEnabled)
        
        self.medianToggle.toggled.connect(self.medianSlider.setEnabled)
        self.medianToggle.toggled.connect(self.onMedian)
        self.medianValue.valueChanged.connect(self.medianSlider.setValue)
        self.medianSlider.valueChanged.connect(self.medianValue.setValue)
        self.medianToggle.toggled.connect(self.medianValue.setEnabled)

        self.contourSlider.valueChanged.connect(self.contourValue.setValue)
        self.contourValue.valueChanged.connect(self.contourSlider.setValue)
        self.contourToggle.toggled.connect(self.contourValue.setEnabled)
        self.contourToggle.toggled.connect(self.contourSlider.setEnabled)
        self.contourToggle.toggled.connect(self.embossFunction) 
        
        self.contrastSlider.valueChanged.connect(self.contrastValue.setValue)
        self.contrastValue.valueChanged.connect(self.contrastSlider.setValue)
        self.contrastToggle.toggled.connect(self.contrastValue.setEnabled)
        self.contrastToggle.toggled.connect(self.contrastSlider.setEnabled)
        self.contrastToggle.toggled.connect(self.contrastFunction)    

        self.thresholdSlider.valueChanged.connect(self.thresholdValue.setValue)
        self.thresholdValue.valueChanged.connect(self.thresholdSlider.setValue)
        self.thresholdToggle.toggled.connect(self.thresholdValue.setEnabled)
        self.thresholdToggle.toggled.connect(self.thresholdSlider.setEnabled)     
        self.thresholdToggle.toggled.connect(self.thresholdFunction)    
        
        self.medianValue.valueChanged.connect(self.onMedian)
        self.blurValue.valueChanged.connect(self.onBlur)
        self.dilationValue.valueChanged.connect(self.onDilate)

        self.record.toggled.connect(self.showA.setEnabled)
        self.record.toggled.connect(self.aToggle.setEnabled)
        self.record.toggled.connect(self.showV.setEnabled)
        self.record.toggled.connect(self.vToggle.setEnabled)
        self.record.toggled.connect(self.showR.setEnabled)
        self.record.toggled.connect(self.rToggle.setEnabled)

        self.aToggle.toggled.connect(lambda checked: self.update_toggles(self.aToggle, checked))
        self.vToggle.toggled.connect(lambda checked: self.update_toggles(self.vToggle, checked))
        self.rToggle.toggled.connect(lambda checked: self.update_toggles(self.rToggle, checked))
        
        # initialize values
        self.thresholdVal = 0
        self.contrastVal = 1
        self.embossVal = 1

        # initialize video display
        self.videos = []
        self.testButton.clicked.connect(self.addVideos)
        self.testButton.hide() # test button to add blank videos
        self.emptyVideo = QPixmap(u"background.jpg") # blank video background
        self.emptyTruth = True
        self.numColumns = 2
        self.dim = [640, 480]
        self.numVideos = 1
        self.countPlaceholder=0
        self.COUNTADDVIDEO = 0 # testing  count variables
        self.COUNTCAPTURE = 0
        self.COUNTPARSE = 0
        self._running = True
        self.backSubsMOG2={} # background subtraction
        self.backSubsKNN={}
        self.subBackVal = 0
        self.blurVal=0
        self.dilateVal=0
        self.medianVal=0
        self.consecutiveFrames=12 # for frame diff
        self.consecutiveFramesValue.setValue(12)
        self.frameDifVal=30
        self.frameDiffValue.setValue(30)
        self.resetFrames=False
        self.persistence = 30
        self.carriageReturnLines=[]

        # set empty placeholder video
        self.video_1.setIcon(QtGui.QIcon(u"background.jpg"))
        self.video_1.setMaximumSize(self.dim[0], self.dim[1])
        self.video_1.setIconSize(QtCore.QSize(self.dim[0]*.99, self.dim[1]*.99))
        self.video_1.setCheckable(True)
        self.video_1.setStyleSheet("border: none; background: transparent;")

        # update plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePlots)
        self.timer.start(33)
        self.clearPlotsTruth=False

        # carriage return
        self.Rtimer = QTimer()
        self.Rtimer.timeout.connect(self.updateCarriageReturn)
        self.Rtimer.start(200)
        self.startCarriageReturn=False
        self.currentText=self.printOutput.toPlainText()

        # data collection
        self.data={}
        self.dataTemp={}
        self.velocityAreaData={}
        #self.pixToum = 4.2 # 20x
        self.pixToum = 1/0.083 # 60x
        self.scaleLength = 50
        self.barHeight = 30
        self.posX = 5
        self.posY = 5
        self.divisions = 5
        self.fontScale = 1.5
        self.scaleBarDialog = ScaleBar(self.dim[1], self.dim[0], self.pixToum)
        self.scaleBarDialog.valuesUpdated.connect(self.update_values)
        self.frameSize=(640, 480)

        # data view
        self.summaryPlot.showGrid(x=True, y=True)
        self.summaryPlot.setLabel("left", "Velocity (um/s)")
        self.summaryPlot.setLabel("bottom", "Time (s)")
        self.summaryPlot.setLimits(xMin=0, xMax=360, yMin=-10000, yMax=10000)
        self.summaryPlot.setXRange(0, 30)
        self.summaryPlot.setYRange(-10, 10)
        self.summaryData=defaultdict(list)
        self.tempData=defaultdict(list)
        self.rectBuffer={}
        self.velocities=[] 
        self.videoNames=[]
        self.curves={}
        self.colors={}
        self.previousCRFunction=''
        self.npmData={}
        self.show

        try:
            self.output_dir = 'results'
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(e)
            
    def update_toggles(self, current_toggle, checked):
        if checked:
            for toggle in [self.aToggle, self.vToggle, self.rToggle]:
                if toggle != current_toggle:
                    toggle.setChecked(False)
    
    def update_values(self, scaleFactor, length, width, x, y, divisions, fontScale):
        self.pixToum = scaleFactor
        self.scaleLength = length
        self.barHeight = width
        self.posX = x
        self.posY = y
        self.divisions = divisions
        self.fontScale = fontScale
    
    def onCollateData(self):
        self.NPMPlot.show()

    def onEditScaleBar(self):
        self.scaleBarDialog.setFrameSize(self.frameSize)
        self.scaleBarDialog.show()

    def kalmanFilterFunc(self, data):
        # Kalman filter parameters
        n_iter = len(data)
        z = data

        # Allocate space for arrays
        x = np.zeros(n_iter)
        P = np.zeros(n_iter)
        x[0] = z[0]
        P[0] = 1

        Q = 0.000000469  # Process noise covariance
        R = 0.000043299  # Measurement noise covariance

        # Kalman filter algorithm
        for k in range(1, n_iter):
            x_pred = x[k-1]
            P_pred = P[k-1] + Q

            K = P_pred / (P_pred + R)
            x[k] = x_pred + K * (z[k] - x_pred)
            P[k] = (1 - K) * P_pred

        return x, z
    
    def updatePlots(self):
        for item in self.summaryPlot.items(): # return all labels
            if isinstance(item, pg.TextItem):
                self.summaryPlot.removeItem(item) # clear each label
        for video in self.videoNames:
            try:
                self.summaryData[video] # check if video exists
            except Exception as e:
                print(f'Error reading data for {e}', end='\r')
                return
            if not self.summaryData[video] == self.tempData[video] and self.record.isChecked(): # check if record button is pressed and new data is different from previous frame
                duplicate = set()
                unique = []
                try:
                    for coord in self.summaryData[video]:
                        if coord[0] not in duplicate: # remove duplicate; can't have 2 different velocities for a given time
                            unique.append(coord)
                            duplicate.add(coord[0])
                    t, v, rot, ali = zip(*unique)
                    if self.kalmanToggle.isChecked():
                        v, _ = self.kalmanFilterFunc(v)
                        rot, _ = self.kalmanFilterFunc(rot)
                        ali, _ = self.kalmanFilterFunc(ali)
                    if self.rToggle.isChecked():
                        y = rot
                    elif self.vToggle.isChecked():
                        y = v
                    elif self.aToggle.isChecked():
                        y = ali
                    else:
                        y = v
                    if video not in self.curves: # initialize curve for each video
                        pen = pg.mkPen(color=self.colors[video])
                        self.curves[video] = self.summaryPlot.plot(t, y, pen=pen, symbol=None, name='Median Velocity')
                    else: # update instead of clearing and replotting each curve
                        self.curves[video].setData(t, y)
                    label = pg.TextItem(text=f'({round(y[-1],2):.2f}, {round(t[-1],2):.2f})', anchor=(0.5, 1))
                    label.setPos(t[-1], y[-1])
                    self.summaryPlot.addItem(label) # add velocity label
                    self.tempData[video]=self.summaryData[video].copy() # maintain current frame data to compare to next frame

                    '''
                    NPM Data Extraction
                    '''
                    if 'V' in video:
                        try:
                            voltage = float(video.split('_')[1].split('V')[0])
                            filteredVelocity, error = self.dataStatistics('IQR', v)
                            meanVelocity = np.mean(filteredVelocity)
                            self.npmData[voltage]=(meanVelocity, error)
                        except:
                            pass
                except Exception as e:
                    err = traceback.format_exc()
                    print(err)
                    pass
            elif self.record.isChecked() and self.summaryData[video]:
                t, v, rot, ali = zip(*self.summaryData[video])
                dct = {"Timestamp (s)":t,"Velocity (um/s)":v,"Rotational Velocity (deg/s)":rot,"Alignment Angle (deg)":ali}
                summaryData = pd.DataFrame(dct)
                summaryData.to_csv(f'{self.output_dir}/{video.split(".")[0]}.csv', index=False, header=True)

                rawData = pd.DataFrame(self.data[video])
                rawData.rename_axis("Timestamp (s)")
                rawData.to_csv(f'{self.output_dir}/{video.split(".")[0]}_raw.csv', header=True)
                
                VAData = pd.DataFrame(self.velocityAreaData[video])
                VAData.rename_axis("Timestamp (s)")
                VAData.to_csv(f'{self.output_dir}/{video.split(".")[0]}_VA.csv', header=True)
                
        self.summaryPlot.update() # update GUI
        if self.NPMPlot.isVisible() and self.npmData:
            data=[]
            for voltage in self.npmData:
                velocity = self.npmData[voltage][0]
                error = self.npmData[voltage][1]
                data.append((voltage, velocity, error))
            data=sorted(data)
            self.NPMPlot.updatePlot(data)

    def dataStatistics(self, method, data):
        try:
            if method == 'IQR':
                data=np.array(data)
                # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                
                # Calculate IQR
                IQR = Q3 - Q1
                
                # Define lower and upper bounds for outliers
                lowerBound = Q1 - 1.5 * IQR
                upperBound = Q3 + 1.5 * IQR
                
                # Filter out outliers
                filteredData = data[(data >= lowerBound) & (data <= upperBound)]
                error = IQR / 2
        
                return filteredData, error
            else:
                print(f'{method} not a statistical method!')
                pass
        except Exception as e:
            err = traceback.format_exc()
            print(err)

    def onPersistence(self):
        self.persistence = self.contDetectValue.value()
    
    def onRecord(self, checked):
        global data
        global videoData

        QTimer.singleShot(100, self.clearPlots)

    def clearPlots(self):
        if self.record.isChecked():
            self.summaryPlot.clear()
            self.tempData.clear()
            self.curves.clear()
            for video in self.videoNames:
                try:
                    self.summaryData[video].clear()
                    self.printNewLine(f'Cleared plot for: {video}')
                    self.clearPlotsTruth = True
                except:
                    pass
            self.summaryPlot.update()
            
    def checkCounts(self):
        self.printCarriageReturn(f'{(self.COUNTPARSE, self.COUNTADDVIDEO, self.COUNTCAPTURE)}')
    
    def onClear(self):
        clearVideos(self.videoLayout) # clear videos
        
        # Perform cleanup tasks
        self._running=False
        for connection in self.videoConnections:
            connection.signals.result.disconnect(self.onFrameReady)
            connection.stop()

        # Clean up video connections
        self.videoConnections.clear()
        self.threadPool.waitForDone()

        # create blank object
        tempVideo = QPushButton('')
        tempVideo.setObjectName('1_blank')
        tempVideo.setCheckable(True)
        tempVideo.setMaximumSize(self.dim[0], self.dim[1])
        tempVideo.setStyleSheet("QPushButton { border: none; background-color: transparent; }")
        tempVideo.setIcon(QtGui.QIcon(u"background.jpg"))
        tempVideo.setIconSize(QtCore.QSize(self.dim[0]*.99, self.dim[1]*.99))
        tempVideo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.numColumns = 2
        self.countPlaceholder=0
        self.numVideos=1
        self.videos = []
        self.emptyTruth = True
        self._running=True
        self.videoNames.clear()
        self.videoFiles.clear()
        self.summaryData.clear()
        self.videoLayout.addWidget(tempVideo, 0, 0)    

    def cropFrame(self, frame, targetSize):
        # check if frame needs to be cropped
        oldH, oldW = frame.shape[:2]
        targetH, targetW = targetSize
        oldAR = oldW/oldH
        newAR = targetW/targetH

        # stretch image to fit while maintaining aspect ratio
        if oldAR > newAR: # original is wider, scale by height
            newH = targetH
            newW = int(targetH*oldAR)
        else: # original is taller, scale by width
            newW = targetW
            newH = int(targetW/oldAR)

        # resize while maintaining aspect ratio
        resizedFrame = cv2.resize(frame, (newW, newH), interpolation=cv2.INTER_AREA) # resize to PyQt frame

        # calculate cropping coordinates
        startX = (newH-targetW) // 2
        startY = (newH - targetH) // 2

        # crop image
        croppedFrame = resizedFrame[startX:startX + targetW, startY:startY + targetH]

        return croppedFrame
    
    def captureVideos(self, videoPath, videoName, resultCallback):
        cap = cv2.VideoCapture(videoPath)
        frame_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        truth = True
        checked = False
        frames=[]
        evenFrame=None
        oddFrame=None
        temp=0
        detectedContours=None
        n=0
        medianFrames=[]
        timer=None
        runOnce=True
        stopData=False
        collectData={}
        color = self.colors[videoName][::-1]
        delay = 1/cap.get(cv2.CAP_PROP_FPS)
        t1=0
        t2=0
        frame2=0
        frame1=0
        adjName = os.path.splitext(videoName)[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{self.output_dir}/{adjName}.mp4', fourcc, frame_fps, (frame_width,frame_height))
        originalOut = cv2.VideoWriter(f'{self.output_dir}/{adjName}_original.mp4', fourcc, frame_fps, (frame_width,frame_height))
        #out = None
        #originalOut = None
        median = 0
        frameCount = 0
        
        if not cap.isOpened():
            msg = f"Error: Could not open video {videoName}.\n"
            self.printNewLine(msg)
            return   
        
        while self._running:
            ret, frame = cap.read()
            timer = cv2.getTickCount()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
                if self.record.isChecked() and not runOnce and not stopData:
                    stopData=True
                    self.printNewLine(f'Switching to collected data for {videoName}. . .')
                    if out and originalOut:
                        out.release()
                        originalOut.release()
                continue  # Continue the loop to read the video from the start

            if self.record.isChecked() and runOnce:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                out = cv2.VideoWriter(f'{self.output_dir}/{adjName}.mp4', fourcc, frame_fps, (frame_width,frame_height))
                originalOut = cv2.VideoWriter(f'{self.output_dir}/{adjName}_original.mp4', fourcc, frame_fps, (frame_width,frame_height))
                self.summaryData[videoName].clear()
                self.data[videoName].clear()
                self.velocityAreaData[videoName].clear()
                collectData.clear()
                frames.clear()
                detectedContours=[]
                n=0
                runOnce=False
                stopData=False
                height, width = frame.shape[:2]

                self.rectBufferMutex.lock()
                self.frameSize = (height, width)
                self.rectBufferMutex.unlock()
                
            elif not self.record.isChecked():
                runOnce=True
                if out and originalOut:
                    out.release()
                    originalOut.release()

            originalFrame = frame.copy()
            
            for video in self.videos:
                if video.objectName() == videoName:
                    checked = video.isChecked() # check if any video is checked/clicked on
            dark = False

            try:
                if checked or len(self.videos) == 0:
                    if any([self.contrastToggle.isChecked(),
                                                          self.blurToggle.isChecked(),
                                                          self.dilationToggle.isChecked(),
                                                          self.subBackToggle.isChecked(),
                                                          self.contourToggle.isChecked(),
                                                          self.thresholdToggle.isChecked(),
                                                          self.frameDiffToggle.isChecked()]): # don't darken if only one video is shown or if any process is toggled
                        dark = False
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # remove colors, most processes, such as frame differencing require this
                    if self.contrastToggle.isChecked(): # contrast
                        frame = cv2.convertScaleAbs(frame, alpha=self.contrastVal, beta=0)
                        
                    if self.blurToggle.isChecked(): # gaussian blur
                        frame = cv2.GaussianBlur(frame, (self.blurVal, self.blurVal), 0)
                        
                    if self.subBackToggle.isChecked(): # subtract background
                        if self.subBackMethod.currentText() == "MOG2": # Mixture of Gaussians 2 background subtraction method
                            frame = self.backSubsMOG2[videoName].apply(frame, learningRate=0.001)
                        if self.subBackMethod.currentText() == "KNN": # K Nearest Neighbors background subtraction method
                            frame = self.backSubsKNN[videoName].apply(frame, learningRate=0.01)
                            
                    if self.contourToggle.isChecked(): # emboss, misnamed as "contour" when originally writing, too lazy to replace every reference
                        kernel = np.array([[2, 1, 0],[1, 0, -1],[0, -1, -2]])
                        frame = cv2.convertScaleAbs(cv2.filter2D(frame, -1, kernel)*self.embossVal)
                        
                    if self.thresholdToggle.isChecked(): # threshold
                        _, frame = cv2.threshold(frame, self.thresholdVal, 255, cv2.THRESH_BINARY)

                    if self.invertToggle.isChecked():
                        frame = cv2.bitwise_not(frame)
                    
                    if self.dilationToggle.isChecked(): # dilation for emboss
                        frame = cv2.dilate(frame, None, iterations=self.dilateVal)
                        kernel = np.ones((3, 3), np.uint8)
                        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                    
                    medianFrames.append(frame)
                    if len(medianFrames) >= self.medianVal:
                        medianFrames = medianFrames[-self.medianVal:]
                    if self.medianToggle.isChecked() and self.medianVal > 0: # median frame filtering, helps remove flicker after applying all filters
                        # Stack frames and apply median filter
                        stack = np.stack(medianFrames, axis=2)
                        frame = np.average(stack, axis=2).astype(np.uint8)

                    if self.thresholdToggle.isChecked(): # threshold
                        _, frame = cv2.threshold(frame, self.thresholdVal, 255, cv2.THRESH_BINARY)
                        
                    elif len(self.videos) != 0:
                        dark = True

                    frameCount=cap.get(cv2.CAP_PROP_POS_FRAMES)

                    if self.consecutiveFrames > 1:
                        if frameCount % 2 == 0:
                            evenFrame=frame.copy()
                            temp=cap.get(cv2.CAP_PROP_POS_FRAMES)
                        elif frameCount > temp or frameCount == 1: # make sure that it takes the difference from a frame and a previous frame, not sure if necessary
                            oddFrame=frame.copy()
    
                        if evenFrame is not None and oddFrame is not None:
                            frames.append(cv2.absdiff(oddFrame, evenFrame)) # find difference between current 2 most recent frames

                        if len(frames) == self.consecutiveFrames + 1:
                            frames.pop(0)
                        
                        if len(frames) > self.consecutiveFrames + 1:
                            self.printNewLine(f'Cached frames "{len(frames)}" exceeds CF "{self.consecutiveFrames}" for {videoName}! Clearing. . .')
                            frames=frames[:-self.consecutiveFrames]
                    else:
                        frames=[frame]

                    if not stopData:
                        self.rectBufferMutex.lock() # thread locking
                    try:
                        if len(frames) == self.consecutiveFrames:
                            if self.frameDiffToggle.isChecked():  
                                try:
                                    if not stopData:
                                        frameDifference = self.frameDifference(frames, videoName, frameCount, color, frame_fps)
                                        if frameDifference is not None:
                                            detectedContours, trackedContours, trackedObjects = frameDifference
                                            collectData[frameCount]=(detectedContours, trackedContours, trackedObjects)
                                    else:
                                        try:
                                            detectedContours, trackedContours, trackedObjects = collectData[frameCount]
                                        except Exception as e:
                                            err = traceback.format_exc()
                                            self.printNewLine(f'Error loading contours for frame {frameCount}, {videoName}: {err}')
                                    if len(detectedContours) == 3:
                                        median = detectedContours[2]
                                except Exception as e:
                                    err = traceback.format_exc()
                                    self.printNewLine(f'Error detecting contours for {videoName}: {err}')
                        try:
                            if detectedContours is not None and self.frameDiffToggle.isChecked(): # have bounding boxes persist
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                if self.actionOriginal_Frame.isChecked():
                                    frame = originalFrame
                                if self.record.isChecked() and trackedContours and trackedObjects:
                                    if self.actionBounding_Boxes_2.isChecked():
                                        for box in trackedContours:
                                            try:
                                                cv2.drawContours(frame,[box],0,color,2)
                                                cv2.drawContours(originalFrame,[box],0,color,2)
                                            except Exception as e:
                                                self.printNewLine(f'Error drawing tracking rects for {videoName}: {e}')
                                    for trackedObject in trackedObjects:
                                        try:
                                            if trackedObject is not None:
                                                msg=str(trackedObject[0])
                                                cv2.line(frame, trackedObject[1][4], trackedObject[1][5], color=self.colors[videoName], thickness=2)
                                                cv2.line(originalFrame, trackedObject[1][4], trackedObject[1][5], color=self.colors[videoName], thickness=2)
                                                if self.actionSpeed.isChecked():
                                                    msg+=f', {trackedObject[2]}'
                                                if self.actionDirection.isChecked():
                                                    trackedObject[1][2].draw(frame)
                                                    trackedObject[1][2].draw(originalFrame)
                                                if self.actionAngular_Velocity.isChecked():
                                                    msg+=f', {trackedObject[3]}'
                                                    cv2.circle(frame, trackedObject[1][3], radius=4, color=self.colors[videoName], thickness=-1)
                                                    cv2.circle(originalFrame, trackedObject[1][3], radius=4, color=self.colors[videoName], thickness=-1)
                                                if self.actionAlignment_Angle.isChecked():
                                                    msg+=f', {trackedObject[4]}'
                                                if self.actionPoints_2.isChecked():
                                                    cv2.circle(frame, trackedObject[1][1], radius=4, color=color, thickness=-1)
                                                    center=trackedObject[1][1]
                                                    cv2.circle(originalFrame, trackedObject[1][1], radius=4, color=color, thickness=-1)
                                                    center=trackedObject[1][1]
                                                else:
                                                    center=trackedObject[1][0]
                                                cv2.putText(frame, msg, center,
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)  
                                                cv2.putText(originalFrame, msg, center,
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)  
                                                
                                        except Exception as e:
                                            self.printNewLine(f'Error drawing text for {videoName}: {e}')
                                else:
                                    try:
                                        if self.actionBounding_Boxes_2.isChecked():
                                            for box in detectedContours[0]:
                                                cv2.drawContours(frame,[box],0,color,2)
                                        if self.actionPoints_2.isChecked():
                                            for point in detectedContours[1]:
                                                cv2.circle(frame, point, radius=4, color=color, thickness=-1)    
                                    except:
                                        pass
                        except Exception as e:
                            err = traceback.format_exc()
                            self.printNewLine(f'Error drawing contours for {videoName}: {err}')
                    finally:
                        if not stopData:
                            self.rectBufferMutex.unlock()
            except Exception as e:
                self.printNewLine(e)

            if self.actionOriginal_Frame.isChecked() and not self.frameDiffToggle.isChecked():
                frame = originalFrame
            
            if self.actionShow.isChecked():
                scaleBar(frame, scaleFactor=self.pixToum, scaleLength=self.scaleLength, 
                         divisions = self.divisions, posX = self.posX, posY = self.posY, 
                         fontScale = self.fontScale, thickness = 2, border = 1, barHeight = self.barHeight)
                scaleBar(originalFrame, scaleFactor=self.pixToum, scaleLength=self.scaleLength, 
                         divisions = self.divisions, posX = self.posX, posY = self.posY, 
                         fontScale = self.fontScale, thickness = 2, border = 1, barHeight = self.barHeight)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            pos=(5, 5)
            height, width = frame.shape[:2]
            fontScale = 1.5
            thickness = 2
            if self.actionFPS.isChecked():
                msgFPS = f'FPS : {str(int(fps))}'
                rectFPS, posFPS = self.textBackground(msgFPS, fontScale, thickness, pos)
                cv2.rectangle(frame, rectFPS[0], rectFPS[1], color=(255, 255, 255), thickness=-1)
                cv2.putText(frame, msgFPS, posFPS, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)
                cv2.rectangle(originalFrame, rectFPS[0], rectFPS[1], color=(255, 255, 255), thickness=-1)
                cv2.putText(originalFrame, msgFPS, posFPS, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)
                pos=(5, 5+rectFPS[1][1])

            if self.actionDetails.isChecked():
                msgDetails=f'{videoName.split("_")[1]}, {width}x{height}'
                rectDetails, posDetails = self.textBackground(msgDetails, fontScale, thickness, pos)
                cv2.rectangle(frame, rectDetails[0], rectDetails[1], color=(255, 255, 255), thickness=-1)
                cv2.putText(frame, msgDetails, posDetails, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)
                cv2.rectangle(originalFrame, rectDetails[0], rectDetails[1], color=(255, 255, 255), thickness=-1)
                cv2.putText(originalFrame, msgDetails, posDetails, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)                
                pos=(pos[0], 5+rectDetails[1][1])
            
            if self.actionMedian_Velocity_2.isChecked():
                if median:
                    msgMedian=f'Median: {median:.2f} um/s'
                    rectMedian, posMedian = self.textBackground(msgMedian, fontScale, thickness, pos)
                    cv2.rectangle(frame, rectMedian[0], rectMedian[1], color=(255, 255, 255), thickness=-1)
                    cv2.putText(frame, msgMedian, posMedian, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)
                    cv2.rectangle(originalFrame, rectMedian[0], rectMedian[1], color=(255, 255, 255), thickness=-1)
                    cv2.putText(originalFrame, msgMedian, posMedian, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)                   
            
                
            if not stopData and self.record.isChecked():
                out.write(frame)
                originalOut.write(originalFrame)
            elif out and originalOut:
                out.release()
                originalOut.release()
            
            if dark:
                frame = np.clip(frame*0.5, 0, 255).astype(np.uint8) # darken videos when selected
            scaleFactor=1/self.numColumns
            frame = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_AREA) # resize video for performance

            percentage = 0.1
            height, width = frame.shape[:2]
            bw = int(percentage*width)
            bh = int(percentage*height)
            brect = [(bw, bh), (width - bw, height - bh)]
            #cv2.rectangle(frame, brect[0], brect[1], (0,0,255), thickness=2)
            
            pixmap = cv2FrameToPixmap(frame)  # convert cv2 frame to PyQt pixmap
            frameRatio = int((frame_fps/60))

            if frameRatio == 0 or frame_fps <= 60:
                resultCallback.emit((pixmap, videoName))
            elif frameCount % frameRatio == 0:
                resultCallback.emit((pixmap, videoName))

            sleep(1/frame_fps)
            
            if truth:
                self.COUNTCAPTURE+=1
                # return upon successful first reading video, basically just a check that video was indeed read initially
                msg = f"Captured frame from {videoName} with dimensions: {frame.shape if frame is not None else 'None'}\n"
                self.printNewLine(msg)
                truth = False

        if out and originalOut:
            out.release()
        cap.release() # close video

    def textBackground(self, text, fontScale, thickness, pos):
        (textWidth, textHeight), baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        rect = [(pos[0], pos[1]), (pos[0] + textWidth, pos[1] + textHeight+baseline)]
        textPos = (pos[0], pos[1] + textHeight + baseline // 2)
        
        return (rect, textPos)

    def rectBorder(self, start, end, borderL, borderR, borderU, borderD): # return border positions for any given rect
        newStart = (start[0]-borderL, start[1]-borderU)
        newEnd = (end[0] + borderR, end[1]+borderD)
        return [newStart, newEnd]

    def returnVideos(self, layout):
        rows = layout.rowCount()
        cols = layout.columnCount()
        videos=[]
        for row in range(rows): 
            for col in range(cols):
                widget = layout.itemAtPosition(row, col)
                if widget is not None:
                    videos.append(widget.widget())
        return videos

    def onFrameReady(self, result):
        if result is not None:
            pixmap = result[0]
            name = result[1]
            videos = self.returnVideos(self.videoLayout)
            for video in videos:
                try:
                    if video.objectName() == name:
                        video.setIcon(QtGui.QIcon(pixmap))
                except Exception as e:
                    msg = f'No match for {video.objectName()}: {e}'
                    self.printNewLine(msg)
        else:
            pass
            
    def addVideos(self, name='BLANK', path=None): 
        if self.emptyTruth:
            self.videoLayout.itemAtPosition(0, 0).widget().setObjectName(f"{self.numVideos}_{name}")
            self.emptyTruth = False
        elif not self.numVideos == self.numColumns**2: # stop at max
            self.countPlaceholder+=1  
            self.numVideos+=1
            
            # Calculate the row and column for the widget (I have no idea why I need to add 1 but it works)
            row = self.countPlaceholder // (self.numColumns + 1) # Integer division for row index
            column = self.countPlaceholder % (self.numColumns + 1) # Modulo operation for column index
            if column == 1: # no clue why this works
                column = 2
                self.countPlaceholder+=1                
            
            tempVideo = QPushButton('')
            tempVideo.setObjectName(f"{self.numVideos}_{name}")
            tempVideo.setCheckable(True)
            tempVideo.setIcon(QtGui.QIcon(u"background.jpg"))
            tempVideo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.videoLayout.addWidget(tempVideo, row, column)  # Add to the layout

            rowCount = 0
            colCount = 0
            for i in range(self.videoLayout.count()):
                widget = self.videoLayout.itemAt(i)
                row, col, rowSpan, colSpan = self.videoLayout.getItemPosition(i)
                rowCount = max(rowCount, row + rowSpan)
                colCount = max(colCount, col + colSpan)
            
            rows = rowCount #self.videoLayout.rowCount()
            cols = colCount #self.videoLayout.columnCount()
            width = self.dim[0]/rows
            height = self.dim[1]/(cols-1) # no clue
            self.videos = resizeVideos(rows, cols, width, height, self.videoLayout) # resize videos to fit
        else:
            self.countPlaceholder=0
            clearVideos(self.videoLayout)
            self.numColumns+=1
            self.numVideos+=1
            for video in self.videos:
                # Calculate the row and column for the widget
                row = self.countPlaceholder // (self.numColumns + 1) # Integer division for row index
                column = self.countPlaceholder % (self.numColumns + 1) # Modulo operation for column index
                if column == 1:
                    column = 2
                    self.countPlaceholder+=1   
                self.videoLayout.addWidget(video, row, column)  # Add to the layout
                self.countPlaceholder+=1

                        
            row = self.countPlaceholder // (self.numColumns + 1) # Integer division for row index
            column = self.countPlaceholder % (self.numColumns + 1) # Modulo operation for column index
            if column == 1:
                column = 2
                self.countPlaceholder+=1   
                
            tempVideo = QPushButton('')
            tempVideo.setObjectName(f"{self.numVideos}_{name}")
            tempVideo.setCheckable(True)
            tempVideo.setIcon(QtGui.QIcon(u"background.jpg"))
            tempVideo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.videoLayout.addWidget(tempVideo, row, column)  # Add to the layout
            
            rows = self.videoLayout.rowCount()
            cols = self.videoLayout.columnCount()
            width = self.dim[0]/rows
            height = self.dim[1]/(cols-1) # no clue
            self.videos = resizeVideos(rows, cols, width, height, self.videoLayout) # resize videos to fit
        if path:
            videoName = f"{self.numVideos}_{name}"
            self.colors[videoName]=random_neon_color()
            self.videoNames.append(videoName)
            self.data[videoName]={}
            self.velocityAreaData[videoName]={}
            self.dataTemp[videoName]={}
            self.COUNTADDVIDEO+=1
            self.printNewLine(f'Added video: {videoName}')
            playVideo = VideoWorker(self.mutex, self.captureVideos, path, videoName)
            playVideo.signals.result.connect(self.onFrameReady)
            self.videoConnections.append(playVideo)
            self.threadPool.start(playVideo)  # Start the worker
            self.backSubsMOG2[videoName] = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=16+48*self.subBackVal)
            self.backSubsKNN[videoName] = cv2.createBackgroundSubtractorKNN(dist2Threshold=100+500*self.subBackVal, history=90, detectShadows=False)
    
    def printNewLine(self, msg):
        # scroll to the bottom
        msg = str(msg)
        self.printOutput.append(msg)
        print(msg)

    def updateCarriageReturn(self):
        self.currentText = self.printOutput.toPlainText() # get current text
        if self.startCarriageReturn:
            # join lines back and update widget
            self.printOutput.setPlainText('\n'.join(self.carriageReturnLines))
    
            # scroll to the bottom
            self.printOutput.verticalScrollBar().setValue(self.printOutput.verticalScrollBar().maximum())
            self.startCarriageReturn=False
    
    def printCarriageReturn(self, msg):
        msg = str(msg)
        lines = self.currentText.splitlines() # separate into lines at \n
        callerFunctionName = inspect.stack()[1].function

        # overwite last message
        if callerFunctionName == self.previousCRFunction:
            lines[-1] = msg
        else:
            lines.append(msg)

        self.carriageReturnLines = lines
        self.startCarriageReturn = True

        print(msg.ljust(200), end='\r')
        self.previousCRFunction=callerFunctionName
    
    def thresholdFunction(self, *args):
        self.thresholdVal = (self.thresholdValue.value()/100)*255
        msg = f'Setting threshold to: {int(100*self.thresholdVal/255)}%'
        if self.initialized:
            if type(args[0]) == bool: # check for "toggled" state 
                if args[0]:
                    self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on toggled
            else:
                self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on value changed

    def contrastFunction(self, *args):
        self.contrastVal = (self.contrastValue.value()/100)+1 # values greater than one increase intensity, less than decrease
        msg = f'Setting contrast to: {int(100*self.contrastVal/2)}%'
        if self.initialized:
            if type(args[0]) == bool: # check for "toggled" state 
                if args[0]:
                    self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on toggled
            else:
                self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on value changed

    def embossFunction(self, *args):
        self.embossVal = (self.contourValue.value()/100)+1 # values greater than one increase intensity, less than decrease
        msg = f'Setting emboss to: {int(self.contourValue.value())}%'
        if self.initialized:
            if type(args[0]) == bool: # check for "toggled" state 
                if args[0]:
                    self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on toggled
            else:
                self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on value changed

    def onBlur(self, *args):
        def nearestOdd(n):
            # If n is odd, return it, otherwise add 1 to make it odd
            return n if n % 2 == 1 else n + 1
        self.blurVal=nearestOdd(int(31*self.blurValue.value()/100)) # 31 is the reccomended max given by ChatGPT
        msg=f'Setting blur to {int(100*self.blurVal/31)}%'
        if self.initialized:
            if type(args[0]) == bool: # check for "toggled" state 
                if args[0]:
                    self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on toggled
            else:
                self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on value changed

    def onDilate(self, *args):
        self.dilateVal=int(self.dilationValue.value())
        msg=f'Setting dilation to {int(100*self.dilateVal/5)}%'
        if self.initialized:
            if type(args[0]) == bool: # check for "toggled" state 
                if args[0]:
                    self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on toggled
            else:
                self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on value changed

    def onMedian(self, *args):
        self.medianVal=int(self.medianValue.value())
        msg=f'Averaging over {self.medianVal} frames'
        if self.initialized:
            if type(args[0]) == bool: # check for "toggled" state 
                if args[0]:
                    self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on toggled
            else:
                self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on value changed
    
    def onFrameDifferencing(self, *args): # allow for multiple types of parameters
        self.frameDifVal = self.frameDiffValue.value()
        msg = f'Frame Differencing with Sensitivity: {self.frameDifVal}%'
        if self.initialized:
            if type(args[0]) == bool: # check for "toggled" state 
                if args[0]:
                    self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on toggled
            else:
                self.printCarriageReturn(msg) # print frame differencing sensitivity with \r on value changed
        if not self.initialized:
            self.initialized=True

    def onConsecutiveFramesValue(self):
        self.consecutiveFrames = self.consecutiveFramesValue.value()

    def frameDifference(self, frame, name, frameCount, color, fps):
        if self.resetFrames:
            self.printCarriageReturn(f'Waiting on frames for {name}. . .')
            return None
        else:
            n=int(frameCount)
            boxes=[]
            centers=[]
            frames=[]
            boxes2D=[]
            centersExact=[]
            medianVel=0
            medianRot=0
            if self.consecutiveFrames > 1:
                if isinstance(frame, list):
                    for f in frame:
                        if len(f.shape) == 3:
                            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                        else:
                            frames.append(f)
                else:
                    self.printNewLine(f'Invalid frames for {name}!')
                    return None
                if len(frames) > 1:
                    print(type(frames))
                    processFrame = cv2.convertScaleAbs(np.sum(frames, axis=0).astype(np.uint8))
            elif isinstance(frame, list):
                processFrame = frame[0]
            contours, hierarchy = cv2.findContours(processFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.frameDifVal:
                    continue
                box2D = cv2.minAreaRect(contour)
                box = cv2.boxPoints(box2D)
                box = np.intp(box)
                boxes.append(box)
                boxes2D.append(box2D)
                centers.append(np.intp(box2D[0]))
                centersExact.append(box2D[0])

            centersExact=np.array(centersExact)
            frameVelocities = []
            frameRotVelocities=[]
            frameAlignment=[]
            trackedRectangles= []
            trackedObjects= []
            timeStamp = frameCount/fps
            if not self.record.isChecked(): # do nothing if record not pressed
                self.data[name]={}
                self.dataTemp[name]={}
                self.velocityAreaData[name]={}
                self.rectBuffer[name]={}
                return (boxes, centers), False, False
            elif boxes2D and self.clearPlotsTruth:
                if n not in self.dataTemp[name]:
                    self.dataTemp[name][n]={}
                if not self.data[name] or not self.dataTemp[name][n-1]: # assign initial points
                    self.printNewLine(f'Initializing tracking for {name}. . .')
                    i=0
                    for box2D in boxes2D: # save all detected rects in first frame to compare to for 2nd frame
                        self.data[name][i]={timeStamp : box2D}
                        self.dataTemp[name][n][i]={timeStamp : box2D, 'corner': cv2.boxPoints(box2D)[0]} # used to return specifically previous frame rectangle information
                        self.rectBuffer[name][i]={'velocities':[],'angularVelocities':[]}
                        self.velocityAreaData[name][i]={}
                        i+=1
                    self.printNewLine(f'Found {i} objects in frame {n} for {name}')
                    return (boxes, centers), False, False
                else:
                    if not self.dataTemp[name][n-1]:
                        return (boxes, centers), False, False
                    else:
                        oldBoxDict = self.dataTemp[name][n-1]
                    oldBoxObjects = list(oldBoxDict.keys())
                    if not oldBoxObjects:
                        return (boxes, centers), False, False
                    j=max(oldBoxObjects) # set j equal to last detected object
                    oldTimeStamp = list(oldBoxDict[oldBoxObjects[0]].keys())[0]
                    oldBoxCenters = []
                    for oldBox in oldBoxDict:
                        oldBoxCenters.append(next(iter(oldBoxDict[oldBox].values()))[0]) 

                    for newBox2D in boxes2D:
                        result = 0
                        detected_box = cv2.boxPoints(newBox2D)
                        try:
                            distances = np.linalg.norm(np.array(oldBoxCenters) - newBox2D[0], axis=1)
                        except:
                            break
                        if len(distances) == 1:
                            sortedOldBoxObjects = np.array(oldBoxObjects)[:1]
                        else:
                            closestIndices = np.argsort(distances)[:2]
                            sortedOldBoxObjects = np.array(oldBoxObjects)[closestIndices]
                        for oldBoxObject in sortedOldBoxObjects:
                            oldBox2D = oldBoxDict[oldBoxObject][oldTimeStamp] # return each rect from the previous frame
                            old_box = cv2.boxPoints(oldBox2D)
                            old_corner = oldBoxDict[oldBoxObject]['corner']
                            result, _ = cv2.rotatedRectangleIntersection(newBox2D, oldBox2D) # check if overlapping
                            if result > 0:
                                if self.actionLock_Size.isChecked():
                                    newBox2D = (newBox2D[0], oldBox2D[1], newBox2D[2])
                                oldBoxCenters.remove(oldBox2D[0])
                                oldBoxObjects.remove(oldBoxObject)
                                timeDif = timeStamp - oldTimeStamp
                                centers = np.array([newBox2D[0],oldBox2D[0]])      

                                new_corner, angle_dif = findAngleDif(newBox2D[0], old_corner, detected_box)
                                    
                                angularVelocity=angle_dif/timeDif
                                alignment, midpoint, centerpoint=calculateAngle(newBox2D, new_corner, self.pixToum)
                                #alignment = alignment*4.5
                                velocities = np.diff(centers, axis=0)[0]/(timeDif*self.pixToum)
                                area = np.prod(newBox2D[1])
                                dimensions = np.array(newBox2D[1])
                                center = np.array(newBox2D[0])
                                
                                labelPos = self.getLabelPos(newBox2D)
                                if angle_dif:
                                    labelPos= new_corner.astype(int)
                                    
                                self.velocityAreaData[name][oldBoxObject].update({timeStamp : [tuple(velocities*self.pixToum), 
                                                                                               tuple(dimensions),
                                                                                               tuple(center)]})
                                self.data[name][oldBoxObject].update({timeStamp : newBox2D})
                                self.dataTemp[name][n][oldBoxObject]={timeStamp : newBox2D, 'corner' : new_corner}

                                if oldBoxObject not in self.rectBuffer[name]:
                                    print(self.rectBuffer[name][oldBoxObject])
                                    self.rectBuffer[name][oldBoxObject] = {'velocities':[],'angularVelocities':[]}
                                self.rectBuffer[name][oldBoxObject]['velocities'].append(np.array(velocities))
                                self.rectBuffer[name][oldBoxObject]['angularVelocities'].append(angularVelocity)
                                    
                                buffer = self.rectBuffer[name][oldBoxObject]
                                appendix=[oldBoxObject,[labelPos, newBox2D[0]], 'nan']
                                if buffer is not None:
                                    if len(buffer['velocities']) > int(self.persistence) + 1:
                                        buffer['velocities'] = buffer['velocities'][:-self.persistence] # remove any overflow from buffer
                                        buffer['angularVelocities'] = buffer['angularVelocities'][:-self.persistence]
                                    if len(buffer['velocities']) < int(self.persistence): # return no velocity if buffer is not filled
                                        appendix = None
                                    if len(buffer['velocities']) == int(self.persistence) + 1: # remove first element if buffer is overfilled
                                        buffer['velocities'].pop(0)
                                        buffer['angularVelocities'].pop(0)
                                    if len(buffer['velocities']) == int(self.persistence): # calculate average velocity if buffer is filled
                                        meanVelocities = np.median(buffer['velocities'], axis=0)
                                        meanRotational = np.mean(buffer['angularVelocities'])
                                        if meanVelocities[np.argmax(np.abs(meanVelocities))] < 0:
                                            sign = -1
                                        else:
                                            sign = 1
                                        speed = np.linalg.norm(meanVelocities)*sign
                                        
                                        start_point = (int(oldBox2D[0][0]), int(oldBox2D[0][1]))
                                        end_point = (
                                            int(start_point[0] - meanVelocities[0] * 10),
                                            int(start_point[1] - meanVelocities[1] * 10)
                                        )
                                        arrow = Arrow(start_point, end_point, color=color, thickness=2, tipLength=0.3)
                                        
                                        frameVelocities.append(speed)
                                        frameRotVelocities.append(meanRotational)
                                        frameAlignment.append(alignment)
                                        trackedRectangles.append(detected_box.astype(int))
                                        appendix = [oldBoxObject,[labelPos, 
                                                                  np.intp(newBox2D[0]), 
                                                                  arrow, 
                                                                  new_corner.astype(int), 
                                                                  np.array(midpoint).astype(int), 
                                                                  np.array(centerpoint).astype(int)],
                                                    f'{speed:.2f} um/s',
                                                    f'{meanRotational:.2f} deg/s',
                                                    f'{alignment:.2f} deg']
                                trackedObjects.append(appendix)
                                break
                        if result == 0:
                            j+=1 
                            self.data[name][j]={timeStamp : newBox2D}
                            self.dataTemp[name][n][j]={timeStamp : newBox2D, 'corner': detected_box[0]}
                            self.rectBuffer[name][j]={'velocities':[],'angularVelocities':[]}
                            self.velocityAreaData[name][j]={}

                if frameVelocities and frameRotVelocities and frameAlignment:
                    if np.array(frameVelocities).size == 0:
                        self.printNewLine(f'Array "frameVelocities" is empty for {name}! {frameVelocities}')
                    else:
                        medianVel = np.median(frameVelocities, axis=0)
                        medianRot = np.median(frameRotVelocities, axis=0)
                        medianAli = np.median(frameAlignment, axis=0)
                        self.summaryData[name].append((timeStamp, medianVel, medianRot, medianAli))
                        #print(f'Frame: {median} um/s'.ljust(200), end='\r') 
                if not boxes2D:
                    self.printNewLine(f'Error: boxes2D empty for {name}! {boxes2D}')
                    return (boxes, centers), False, trackedObjects
                if not trackedObjects:
                    self.printNewLine(f'Error: trackedObjects empty for {name}! {trackedObjects}')
                    return (boxes, centers), trackedRectangles, False
                if not trackedRectangles and not trackedObjects:
                    self.printNewLine(f'Error: trackedObjects empty for {name}! {trackedObjects} trackedRectangles empty for {name}! {trackedRectangles}')
                    return (boxes, centers), False, False
                return (boxes, centers, medianVel), trackedRectangles, trackedObjects

    def getLabelPos(self, box):
        # Convert Box2D to vertices (four points)
        vertices = cv2.boxPoints(box)
        
        # Sort the points: first by y-coordinate (upper-most), then by x-coordinate (right-most)
        vertices = sorted(vertices, key=lambda point: (point[1], -point[0]))
    
        # The first point in the sorted list is the upper-most right vertex
        return tuple(np.intp(vertices[0]))
    
    def subtractBackgroundFunction(self, *args):
        self.subBackVal = self.subBackValue.value()/100
        if self.subBackValue.value() != 0:
            QTimer.singleShot(250,self.onSubBack)

    def onSubBack(self):
        history=450
        for video in self.returnVideos(self.videoLayout):
            try:
                if self.subBackMethod.currentText() == "MOG2":
                    self.backSubsMOG2[video.objectName()] = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=16+48*self.subBackVal, detectShadows=False)
                elif self.subBackMethod.currentText() == "KNN":
                    self.backSubsKNN[video.objectName()] = cv2.createBackgroundSubtractorKNN(dist2Threshold=100+500*self.subBackVal, history=history, detectShadows=False)
                self.printCarriageReturn(f'Subtracting background with intensity {self.subBackVal} using {self.subBackMethod.currentText()}')
            except Exception as e:
                msg=None
                self.printCarriageReturn(f'Failed to subtract background for {video.objectName()}: {e}')

    def openFile(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Video Files (*.avi;*.mp4;*.mov;*.mkv;*.wmv;*.flv;*.mpeg;*.mpg)", options=options)
        
        if files:
            for file in files:
                self.parseFiles(file)

    def exportVideo(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,
                                                  "Save CSV/Excel File",
                                                  "",
                                                  "CSV Files (*.csv);;Excel Files (*.xlsx)",
                                                  options=options)
        
        if fileName:
            print(f"File saved as: {fileName}")
    
    def parseFiles(self, file):
        accepted = False
        fileType = file.split(".")[-1] # return file extension
        for videoType in videoFormats:
            if fileType == videoType:
                accepted = True # return true if file is a video
                break
        if not accepted:
            msg = f'File type: ".{fileType}" not accepted. \nAllowed file types:  ".avi," ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"\n'
            self.printNewLine(msg)
            return
        else:
            for videoFile in self.videoFiles:
                if file == videoFile:
                    msg = f'File "{file}" already added!'
                    self.printNewLine(msg)
                    accepted = False
                    return
        if accepted:
            msg = f'File opened: {file}'
            fileName = file.split(r'/')[-1]
            self.printNewLine(msg)
            try:
                self.addVideos(fileName, file)
                self.videoFiles.append(file)
                self.COUNTPARSE+=1
            except Exception as e:
                msg = f'Failed to add file: {file} {e}'
                self.printNewLine(msg)        
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): # check if dragged item has file location/url
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            Qfiles = event.mimeData().urls()
            files = [Qfile.toLocalFile() for Qfile in Qfiles] # convert qt-type to file path string
            for file in files:
                self.parseFiles(file)
            event.accept()
        else:
            event.ignore()
    
    def closeEvent(self, event):
        try:
            self._running=False
            # Perform cleanup tasks
            for connection in self.videoConnections:
                connection.signals.result.disconnect(self.onFrameReady)
                connection.stop()

            # Clean up video connections
            self.videoConnections.clear()
            self.threadPool.waitForDone()

            # Perform any other necessary cleanup tasks
            event.accept()  # Accept the event to close the window
            
            super().closeEvent(event)
        except Exception as e:
            msg = f"Error during close event: {e}"
            self.printNewLine(msg)
            event.ignore()  # Optionally ignore the event if cleanup fails
        print('\nExited')

if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication(sys.argv)
else:
    app = QtWidgets.QApplication.instance()

if __name__ == '__main__':
    window = MainWindow()
    app.setStyle('Windows')
    window.show()
    print('Running\n')
    app.exec()