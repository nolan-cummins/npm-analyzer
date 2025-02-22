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
from collections import deque

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
from videotools import *
from qtools import *

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

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow): # main window
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initialized = False
        self.setupUi(self)
        self.mutex, self.rectBufferMutex = QMutex(), QMutex() # locking threads
        
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
        self.subBackMethod.currentIndexChanged.connect(self.subtractBackgroundFunction)
        self.NPMPlot = Dialog(None)

        value_changed_connections = {
            self.blurValue: [self.blurSlider.setValue, self.onBlur],
            self.blurSlider: [self.blurValue.setValue],
            self.consecutiveFramesValue: [self.onConsecutiveFramesValue],
            self.contDetectValue: [self.contDetectSlider.setValue, self.onPersistence],
            self.contDetectSlider: [self.contDetectValue.setValue],
            self.embossValue: [self.embossSlider.setValue, self.embossFunction],
            self.embossSlider: [self.embossValue.setValue],
            self.adaptSliderArea: [self.adaptValueArea.setValue, self.adaptFunction],
            self.adaptValueArea: [self.adaptSliderArea.setValue, self.adaptFunction],
            self.adaptSliderC: [self.adaptValueC.setValue, self.adaptFunction],
            self.adaptValueC: [self.adaptSliderC.setValue, self.adaptFunction],
            self.dilationValue: [self.dilationSlider.setValue, self.onDilate],
            self.dilationSlider: [self.dilationValue.setValue],
            self.frameDiffValue: [self.onFrameDifferencing],
            self.frameDiffSliderMax: [self.frameDiffValueMax.setValue],
            self.frameDiffValueMax: [self.frameDiffSliderMax.setValue],
            self.medianValue: [self.medianSlider.setValue, self.onMedian],
            self.medianSlider: [self.medianValue.setValue],
            self.subBackValue: [self.subtractBackgroundFunction],
            self.thresholdValue: [self.thresholdSlider.setValue, self.thresholdFunction],
            self.thresholdSlider: [self.thresholdValue.setValue],
        }
        
        for obj, handlers in value_changed_connections.items():
            for handler in handlers:
                obj.valueChanged.connect(handler)

        toggled_connections = {
            self.frameDiffToggle: [self.onFrameDifferencing],
            self.subBackToggle: [self.subtractBackgroundFunction],
            self.record: [self.onRecord, self.showA.setEnabled, self.aToggle.setEnabled, 
                          self.showV.setEnabled, self.vToggle.setEnabled, 
                          self.showR.setEnabled, self.rToggle.setEnabled],
            
            self.blurToggle: [self.blurValue.setEnabled, self.blurSlider.setEnabled, self.onBlur],
            self.dilationToggle: [self.dilationSlider.setEnabled, self.onDilate, self.dilationValue.setEnabled],
            self.medianToggle: [self.medianSlider.setEnabled, self.onMedian, self.medianValue.setEnabled],
            
            self.embossToggle: [self.embossValue.setEnabled, self.embossSlider.setEnabled, self.embossFunction],
            self.adaptToggle: [self.adaptValueArea.setEnabled, self.adaptSliderArea.setEnabled, self.adaptValueC.setEnabled,
                               self.adaptSliderC.setEnabled, self.adaptMethod.setEnabled, self.adaptFunction],
            self.thresholdToggle: [self.thresholdValue.setEnabled, self.thresholdSlider.setEnabled, self.thresholdFunction],
            
            self.aToggle: [lambda checked: self.update_toggles(self.aToggle, checked)],
            self.vToggle: [lambda checked: self.update_toggles(self.vToggle, checked)],
            self.rToggle: [lambda checked: self.update_toggles(self.rToggle, checked)]
        }
        
        for obj, handlers in toggled_connections.items():
            for handler in handlers:
                obj.toggled.connect(handler)

        
        # initialize values
        self.thresholdVal = 0
        self.adaptVal = self.embossVal = 1
        self.videos = []
        self.testButton.clicked.connect(self.addVideos)
        self.testButton.hide()  # Test button to add blank videos
        self.emptyVideo = QPixmap("background.jpg")  # Blank video background
        self.emptyTruth = True
        self.numColumns, self.numVideos = 2, 1
        self.dim = [640, 480]
        self.countPlaceholder = self.COUNTADDVIDEO = self.COUNTCAPTURE = self.COUNTPARSE = 0
        self._running = True
        self.resetFrames = False
        self.backSubsMOG2, self.backSubsKNN = {}, {}
        self.subBackVal = self.blurVal = self.dilateVal = self.medianVal = 0
        self.consecutiveFrames = 12
        self.consecutiveFramesValue.setValue(12)
        self.frameDifVal = 30
        self.frameDiffValue.setValue(30)
        self.persistence = 30
        self.carriageReturnLines = []
        self.area_value = 21

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
        self.posX, self.posY = 5, 5
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
        self.curves={}
        self.colors={}
        self.npmData={}
        self.velocities=self.videoNames=[] 
        self.previousCRFunction=''
        self.show
        self.output_dir = 'results'
            
    def update_toggles(self, current_toggle, checked):
        if checked:
            for toggle in [self.aToggle, self.vToggle, self.rToggle]:
                if toggle != current_toggle:
                    toggle.setChecked(False)
    
    def update_values(self, scaleFactor, length, width, x, y, divisions, fontScale):
        self.pixToum = scaleFactor
        self.scaleLength = length
        self.barHeight = width
        self.posX, self.posY = x, y
        
        self.divisions = divisions
        self.fontScale = fontScale
    
    def onCollateData(self):
        self.NPMPlot.show()

    def onEditScaleBar(self):
        self.scaleBarDialog.setFrameSize(self.frameSize)
        self.scaleBarDialog.show()

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
                        v, _ = kalmanFilterFunc(v)
                        rot, _ = kalmanFilterFunc(rot)
                        ali, _ = kalmanFilterFunc(ali)
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
                            filteredVelocity, error = dataStatistics('IQR', v)
                            meanVelocity = np.mean(filteredVelocity)
                            self.npmData[voltage]=(meanVelocity, error)
                        except:
                            pass
                except Exception as e:
                    err = traceback.format_exc()
                    print(err)
                    pass
            elif self.record.isChecked() and self.summaryData[video]:
                try:
                    if not os.path.isdir(self.output_dir):
                        os.makedirs(self.output_dir, exist_ok=True)
                except Exception as e:
                    print(e)
                
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

    def onPersistence(self):
        self.persistence = self.contDetectValue.value()
    
    def onRecord(self, checked):
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
        detectedContours=None
        medianFrames=[]
        timer=None
        runOnce=True
        stopData=False
        collectData={}
        color = self.colors[videoName][::-1]
        delay = 1/cap.get(cv2.CAP_PROP_FPS)
        t1 = t2 = n = temp = frame2 = frame1 = median = frameCount = 0

        adjName = os.path.splitext(videoName)[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{self.output_dir}/{adjName}.mp4', fourcc, frame_fps, (frame_width,frame_height))
        originalOut = cv2.VideoWriter(f'{self.output_dir}/{adjName}_original.mp4', fourcc, frame_fps, (frame_width,frame_height))
        
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
                for dataset in [self.summaryData[videoName], self.data[videoName], self.velocityAreaData[videoName], collectData, frames]:
                    dataset.clear()

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
                    if any([self.adaptToggle.isChecked(),
                            self.blurToggle.isChecked(),
                            self.dilationToggle.isChecked(),
                            self.subBackToggle.isChecked(),
                            self.embossToggle.isChecked(),
                            self.thresholdToggle.isChecked(),
                            self.frameDiffToggle.isChecked()]): # don't darken if only one video is shown or if any process is toggled
                        dark = False
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # remove colors, most processes, such as frame differencing require this  
                    if self.blurToggle.isChecked(): # gaussian blur
                        frame = cv2.GaussianBlur(frame, (self.blurVal, self.blurVal), 0)

                    if self.adaptToggle.isChecked(): # adapt
                        method = self.adaptMethod.currentText()
                        if method == "Mean":
                            frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY,self.area_value,self.adaptValueC.value())
                        elif method == "Gaussian":
                            frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,self.area_value,self.adaptValueC.value())

                    if self.thresholdToggle.isChecked(): # threshold
                        if self.autoToggle.isChecked():
                            otsu, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            otsu_normal = int(otsu/255*100)
                            self.thresholdValue.setValue(otsu_normal)
                            self.thresholdSlider.setValue(otsu_normal)
                        else:
                            _, frame = cv2.threshold(frame, self.thresholdVal, 255, cv2.THRESH_BINARY)
                        
                    if self.subBackToggle.isChecked(): # subtract background
                        if self.subBackMethod.currentText() == "MOG2": # Mixture of Gaussians 2 background subtraction method
                            frame = self.backSubsMOG2[videoName].apply(frame, learningRate=0.001)
                        if self.subBackMethod.currentText() == "KNN": # K Nearest Neighbors background subtraction method
                            frame = self.backSubsKNN[videoName].apply(frame, learningRate=0.01)
                        _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            
                    if self.embossToggle.isChecked():
                        kernel = np.array([[2, 1, 0],[1, 0, -1],[0, -1, -2]])
                        frame = cv2.convertScaleAbs(cv2.filter2D(frame, -1, kernel)*self.embossVal)

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
                for f in [frame, originalFrame]:
                    scaleBar(f, scaleFactor=self.pixToum, scaleLength=self.scaleLength, 
                             divisions = self.divisions, posX = self.posX, posY = self.posY, 
                             fontScale = self.fontScale, thickness = 2, border = 1, barHeight = self.barHeight)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            pos=(5, 5)
            height, width = frame.shape[:2]
            fontScale = 1.5
            thickness = 2
            if self.actionFPS.isChecked():
                msgFPS = f'FPS : {str(int(fps))}'
                rectFPS = self.placeLabel(frame, originalFrame, msgFPS, fontScale, thickness, pos)       
                pos=(5, 5+rectFPS[1][1])

            if self.actionDetails.isChecked():
                msgDetails=f'{videoName.split("_")[1]}, {width}x{height}'
                rectDetails = self.placeLabel(frame, originalFrame, msgDetails, fontScale, thickness, pos)                 
                pos=(pos[0], 5+rectDetails[1][1])
            
            if self.actionMedian_Velocity_2.isChecked():
                if median:
                    msgMedian=f'Median: {median:.2f} um/s'
                    self.placeLabel(frame, originalFrame, msgMedian, fontScale, thickness, pos)                  
                
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

            if frameRatio == 0 or frame_fps <= 60 or frameCount % frameRatio == 0:
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

    def placeLabel(self, frame, originalFrame, msg, fontScale, thickness, position):
        rect, pos = textBackground(msg, fontScale, thickness, position)
        cv2.rectangle(frame, rect[0], rect[1], color=(255, 255, 255), thickness=-1)
        cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)
        cv2.rectangle(originalFrame, rect[0], rect[1], color=(255, 255, 255), thickness=-1)
        cv2.putText(originalFrame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,0), thickness, lineType=cv2.LINE_AA)    
        return rect

    def onFrameReady(self, result):
        if result is not None:
            pixmap = result[0]
            name = result[1]
            videos = returnVideos(self.videoLayout)
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

    def checkInitialized(self, args, msg):
        if self.initialized:
            if type(args[0]) == bool:
                if args[0]:
                    self.printCarriageReturn(msg)
            else:
                self.printCarriageReturn(msg)
        
    def thresholdFunction(self, *args):
        if self.adaptToggle.isChecked() and self.thresholdToggle.isChecked():
            self.adaptToggle.setChecked(False)
        self.thresholdVal = (self.thresholdValue.value()/100)*255
        msg = f'Setting threshold to: {int(100*self.thresholdVal/255)}%'
        self.checkInitialized(args, msg)

    def adaptFunction(self, *args):
        if self.thresholdToggle.isChecked() and self.adaptToggle.isChecked():
            self.thresholdToggle.setChecked(False)
        sender = self.sender()
        name = sender.objectName()
        if "Area" in name:
            area_value = self.adaptValueArea.value()
            if area_value % 2 == 0:
                area_value+=1
            self.area_value = area_value
            msg = f'Setting adapt_area to: {area_value}'
        else:
            msg = f'Setting adapt_c to: {self.adaptValueC.value()}'
        self.checkInitialized(args, msg)

    def embossFunction(self, *args):
        self.embossVal = (self.embossValue.value()/100)+1 # values greater than one increase intensity, less than decrease
        msg = f'Setting emboss to: {int(self.embossValue.value())}%'
        self.checkInitialized(args, msg)

    def onBlur(self, *args):
        def nearestOdd(n):
            # If n is odd, return it, otherwise add 1 to make it odd
            return n if n % 2 == 1 else n + 1
        self.blurVal=nearestOdd(int(31*self.blurValue.value()/100)) # 31 is the reccomended max given by ChatGPT
        msg=f'Setting blur to {int(100*self.blurVal/31)}%'
        self.checkInitialized(args, msg)

    def onDilate(self, *args):
        self.dilateVal=int(self.dilationValue.value())
        msg=f'Setting dilation to {int(100*self.dilateVal/5)}%'
        self.checkInitialized(args, msg)

    def onMedian(self, *args):
        self.medianVal=int(self.medianValue.value())
        msg=f'Averaging over {self.medianVal} frames'
        self.checkInitialized(args, msg)
    
    def onFrameDifferencing(self, *args): # allow for multiple types of parameters
        self.frameDifVal = self.frameDiffValue.value()
        msg = f'Frame Differencing with Sensitivity: {self.frameDifVal}%'
        self.checkInitialized(args, msg)
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
            persistence = int(self.persistence)
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
                    processFrame = cv2.convertScaleAbs(np.sum(frames, axis=0).astype(np.uint8))
            elif isinstance(frame, list):
                processFrame = frame[0]
            contours, hierarchy = cv2.findContours(processFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.frameDifVal or area > self.frameDiffValueMax.value()*3:
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
                        self.rectBuffer[name][i] = {'velocities': deque(maxlen=persistence),
                                                    'angularVelocities': deque(maxlen=persistence)}
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

                    oldBoxCenters = np.array(oldBoxCenters)
                    oldBoxObjects = np.array(oldBoxObjects)
                    for newBox2D in boxes2D:
                        result = 0
                        detected_box = cv2.boxPoints(newBox2D)
                        try:
                            distances = np.sum((np.array(oldBoxCenters) - newBox2D[0]) ** 2, axis=1)
                        except Exception as e:
                            print(e)
                            break
                        oldBoxObject = oldBoxObjects[np.argmin(distances)]

                        oldBox2D = oldBoxDict[oldBoxObject][oldTimeStamp] # return each rect from the previous frame
                        old_box = cv2.boxPoints(oldBox2D)
                        old_corner = oldBoxDict[oldBoxObject]['corner']
                        
                        threshold = np.max(np.concatenate((newBox2D[1], oldBox2D[1])))**2
                        distance = np.linalg.norm(np.asarray(newBox2D[0]) - np.asarray(oldBox2D[0]))
                        if distance < threshold:
                            result, _ = cv2.rotatedRectangleIntersection(newBox2D, oldBox2D) # check if overlapping
                        else:
                            result = 0
                            
                        if result > 0:
                            if self.actionLock_Size.isChecked():
                                newBox2D = (newBox2D[0], oldBox2D[1], newBox2D[2])
                            oldBoxCenters[oldBoxCenters != oldBox2D[0]]
                            oldBoxObjects[oldBoxObjects != oldBoxObject]
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
                            
                            labelPos = getLabelPos(newBox2D)
                            if angle_dif:
                                labelPos= new_corner.astype(int)
                                
                            self.velocityAreaData[name][oldBoxObject].update({timeStamp : [tuple(velocities*self.pixToum), 
                                                                                           tuple(dimensions),
                                                                                           tuple(center)]})
                            self.data[name][oldBoxObject].update({timeStamp : newBox2D})
                            self.dataTemp[name][n][oldBoxObject]={timeStamp : newBox2D, 'corner' : new_corner}

                            buffer = self.rectBuffer[name].setdefault(
                                oldBoxObject, {'velocities': deque(maxlen=persistence), 'angularVelocities': deque(maxlen=persistence)}
                            )

                            appendix=[oldBoxObject,[labelPos, newBox2D[0]], 'nan']
                            buffer['velocities'].append(np.array(velocities))
                            buffer['angularVelocities'].append(angularVelocity)
                            
                            if buffer is not None:
                                if len(buffer['velocities']) < persistence:
                                    appendix = None  
                            
                                if len(buffer['velocities']) == persistence:
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
                                    trackedRectangles.append(detected_box.astype(np.intp))
                                    appendix = [oldBoxObject,[labelPos, 
                                                              np.intp(newBox2D[0]), 
                                                              arrow, 
                                                              new_corner.astype(np.intp), 
                                                              np.array(midpoint, dtype=np.intp), 
                                                              np.array(centerpoint, dtype=np.intp)],
                                                f'{speed:.2f} um/s',
                                                f'{meanRotational:.2f} deg/s',
                                                f'{alignment:.2f} deg']
                            trackedObjects.append(appendix)
                        elif result == 0:
                            j+=1 
                            self.data[name][j]={timeStamp : newBox2D}
                            self.dataTemp[name][n][j]={timeStamp : newBox2D, 'corner': detected_box[0]}
                            self.rectBuffer[name][j] = {'velocities': deque(maxlen=persistence),
                                                        'angularVelocities': deque(maxlen=persistence)}
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

    def subtractBackgroundFunction(self, *args):
        self.subBackVal = self.subBackValue.value()/100
        if self.subBackValue.value() != 0:
            QTimer.singleShot(250,self.onSubBack)

    def onSubBack(self):
        history=450
        for video in returnVideos(self.videoLayout):
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
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory", "", options=options)
        
        if directory:
            print(f"Directory selected: {directory}")
            try:
                self.output_dir = directory
                if not os.path.isdir(directory):
                    os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                print(e)
    
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