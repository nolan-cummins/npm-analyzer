import cv2

# imagePath = 'C:/Users/Nolan/Documents/Python Scripts/npm/analysis/data/calibration 20x.png'
# imageName = imagePath.split(r'/')[-1].split('.')[0]
# image = cv2.imread(imagePath)

# if image is None:
#     raise ValueError("Could not load image. Check the file path.")

def scaleBar(image, height=480, scaleFactor=6.9, scaleLength = 30, scaleUnit = 'um', barHeight = 10, border = 1, divisions = 30, fontScale = 1, thickness = 1, posX = 20, posY = 30, textSpacing = 10):
    scaleLengthPixels = int(scaleLength*scaleFactor)
    
    scaleBarStart = (posX+border, height - posY)
    scaleBarEnd = (scaleBarStart[0] + scaleLengthPixels, scaleBarStart[1]-barHeight)
    
    def rectBorder(start, end, borderL, borderR, borderU, borderD): # return border positions for any given rect
        newStart = (start[0]-borderL, start[1]-borderU)
        newEnd = (end[0] + borderR, end[1]+borderD)
        return [newStart, newEnd]
    
    textPosition = (scaleBarStart[0], scaleBarStart[1] - textSpacing - barHeight)  # Adjust position as needed
    
    (textWidth, textHeight), _ = cv2.getTextSize(f'{scaleLength} {scaleUnit}', cv2.FONT_HERSHEY_DUPLEX, fontScale, thickness)
    textStart = (textPosition[0], textPosition[1] - textHeight)
    textEnd = (textPosition[0] + textWidth, textPosition[1])
    
    scaleBarBackground = rectBorder(textStart, textEnd, 0, 0, 5, 5)
    
    cv2.rectangle(image, scaleBarBackground[0], scaleBarBackground[1], color=(255, 255, 255), thickness=-1) # background
    
    i=0
    while True: # black/white divisions
        dist = int((scaleBarEnd[0] - scaleBarStart[0])/divisions)
        start = (scaleBarStart[0]+i*dist,
                 scaleBarStart[1])
        end = (scaleBarStart[0]+(i+1)*dist,
               scaleBarEnd[1])
        #print(start[0], end[0], dist, i)
        if end[0]+(i+1)*dist == scaleBarEnd[0] or end[0] >= scaleBarEnd[0]: # stop if end of bar reached
            end = scaleBarEnd
            if i % 2 == 0:
                cv2.rectangle(image, start, end, color=(0, 0, 0), thickness=-1)
            else:
                cv2.rectangle(image, start, end, color=(255, 255, 255), thickness=-1)
            break
        elif i % 2 == 0:
            cv2.rectangle(image, start, end, color=(0, 0, 0), thickness=-1)
        else:
            cv2.rectangle(image, start, end, color=(255, 255, 255), thickness=-1)
        i+=1
    
    cv2.rectangle(image, scaleBarStart, scaleBarEnd, color=(0, 0, 0), thickness=border) # scale bar border
    
    cv2.putText(image, f'{scaleLength} {scaleUnit}', textPosition, cv2.FONT_HERSHEY_DUPLEX, # text
                fontScale=fontScale, color=(0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)