import cv2
import numpy as np

def scaleBar(image, scaleFactor=6.9, scaleLength = 30, scaleUnit = 'um', barHeight = 10, border = 1, divisions = 30, fontScale = 1, thickness = 1, posX = 20, posY = 30, font=cv2.FONT_HERSHEY_DUPLEX):
    scaleLengthPixels = int(scaleLength*scaleFactor)
    shape = image.shape
    if len(shape) == 2:  # Grayscale image
        height, width = shape
        channels = 1  # Grayscale has one channel
    elif len(shape) == 3:  # Color image
        height, width, channels = shape
    else:
        raise ValueError("Unexpected image shape.")
    msg = f'{scaleLength} {scaleUnit}'
    scaleBarStart = (posX+border, height - posY)
    scaleBarEnd = (scaleBarStart[0] + scaleLengthPixels, scaleBarStart[1]-barHeight)
    
    def rectBorder(start, end, borderL, borderR, borderU, borderD): # return border positions for any given rect
        newStart = (start[0]-borderL, start[1]-borderU)
        newEnd = (end[0] + borderR, end[1]+borderD)
        return [newStart, newEnd]

    (w, h), b = cv2.getTextSize(str(msg), font, fontScale, thickness)
    textPosition = (scaleBarStart[0], scaleBarStart[1] - h - b - barHeight - 5)  # Adjust position as needed
    
    def textBackground(text, fontScale, thickness, pos):
        (textWidth, textHeight), baseline = cv2.getTextSize(str(text), font, fontScale, thickness)
        rect = [(pos[0], pos[1]), (pos[0] + textWidth, pos[1] + textHeight+baseline)]
        textPos = (pos[0], pos[1] + textHeight + baseline // 2)
        
        return (rect, textPos)

    rect, textPos = textBackground(msg, fontScale, thickness, textPosition)
    cv2.rectangle(image, rect[0], rect[1], color=(255, 255, 255), thickness=-1) # background

    def evenDivision(n, f):
        # Step 1: Calculate the step size
        f=f+1
        if f > 1:
            stepSize = n / (f - 1)
        else:
            return np.array([0])  # If f is 1, return a single value
        
        # Step 2: Create evenly spaced values using rounding
        distributedArray = np.round(np.linspace(0, n, f)).astype(int)
        
        # Ensure that the last element is exactly n
        distributedArray[-1] = n
        
        divisions=np.insert(np.diff(distributedArray), 0, 0)
    
        return divisions
    
    dist = evenDivision(scaleLengthPixels, divisions)
    for i in range(0, len(dist)): # black/white divisions
        if i == len(dist)-1:
            break
        start = (scaleBarStart[0]+np.sum(dist[0:i+1]),scaleBarStart[1])
        end = (start[0]+dist[i+1],scaleBarEnd[1])
        if i % 2 == 0:
            cv2.rectangle(image, start, end, color=(0, 0, 0), thickness=-1)
        else:
            cv2.rectangle(image, start, end, color=(255, 255, 255), thickness=-1)
    
    cv2.rectangle(image, scaleBarStart, scaleBarEnd, color=(100, 100, 100), thickness=border) # scale bar border
    
    cv2.putText(image, msg, textPos, font, # text
                fontScale=fontScale, color=(0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)