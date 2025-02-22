from npm_analyzer import *
import threading
"""
All Qt tools/classes for main
"""

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
        self.thread_id=None

        # Add the callback to our kwargs
        self.kwargs['resultCallback'] = self.signals.result

        # run flag
        self._running = True

    def stop(self):
        print('STOP')
        print(f"Stopping thread: {self.thread_id}")
        self.running = False # stop flag
    
    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        if self._running:
            self.thread_id = threading.get_ident()
            print(f"Running in thread: {self.thread_id}")
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