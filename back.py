import cv2
from matplotlib import pyplot as plt
image = cv2.imread("nuts_1.jpeg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
im,thresh = cv2.threshold(gray,50,255,cv2.THRESH_TOZERO) 
#threshold
plt.imshow(thresh)
plt.show()
 
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
img,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
# get contours
# for each contour found, draw a rectangle around it on original 
image
for contour in contours:
 
    # get rectangle bounding contour
 
    [x,y,w,h] = cv2.boundingRect(contour)
    # discard areas that are too large
 
#    if h>300 and w>300:
 
 #       continue
    # discard areas that are too small
 
    if h<5 or w<5:
 
        continue
    # draw rectangle around contour on original image
 
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
# write original image with added contours to disk  
 
cv2.imwrite("contoured.jpg", image) 
