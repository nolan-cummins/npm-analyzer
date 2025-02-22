from npm_analyzer import *
"""
All functions for video analysis
"""

def dataStatistics(method, data):
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

def kalmanFilterFunc(data):
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

def textBackground(text, fontScale, thickness, pos):
    (textWidth, textHeight), baseline = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
    rect = [(pos[0], pos[1]), (pos[0] + textWidth, pos[1] + textHeight+baseline)]
    textPos = (pos[0], pos[1] + textHeight + baseline // 2)
    
    return (rect, textPos)
    
def getLabelPos(box):
    # Convert Box2D to vertices (four points)
    vertices = cv2.boxPoints(box)
    
    # Sort the points: first by y-coordinate (upper-most), then by x-coordinate (right-most)
    vertices = sorted(vertices, key=lambda point: (point[1], -point[0]))

    # The first point in the sorted list is the upper-most right vertex
    return tuple(np.intp(vertices[0]))

def returnVideos(layout):
    rows = layout.rowCount()
    cols = layout.columnCount()
    videos=[]
    for row in range(rows): 
        for col in range(cols):
            widget = layout.itemAtPosition(row, col)
            if widget is not None:
                videos.append(widget.widget())
    return videos

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