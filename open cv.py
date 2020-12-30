
import cv2 as cv

img=cv.imread('photos/panda.jpg')
cv.imshow('panda',img)
cv.waitKey(0)

#converting to grayscale

img =cv.imread('photos/panda.jpg')
Gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',Gray)
cv.waitKey(0)

#blur

img=cv.imread('photos/panda.jpg')
blur=cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)
cv.waitKey(0)

#canny

canny=cv.Canny(blur,125,175)
cv.imshow('canny edges',canny)
cv.waitKey(0)

#diliting the image

dilated=cv.dilate(canny, (7,7),iterations=5)
cv.imshow('Dilited',dilated)
cv.waitKey(0)

#erode
import numpy as np
erode=cv.erode(dilated,(7,7), iterations=5)
cv.imshow('Erode',erode)
cv.waitKey(0)

#############################################################################

#enlarge =inter_cubic / inter_linear
#shrink=inter_area

img=cv.imread('photos/panda.jpg')
resize=cv.resize(img, (1500,1000), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resize)
cv.waitKey(0)

#shrink
img=cv.imread('photos/panda.jpg')
resize=cv.resize(img,(500,500),interpolation=cv.INTER_AREA)
cv.imshow('Resize',resize)
cv.waitKey(0)

#cropping

img=cv.imread('photos/panda.jpg')
crop=img[0:200,0:400]
cv.imshow('cropped',crop)
cv.waitKey(0)

####################################################

capture=cv.VideoCapture('photos/vtest.avi')

while True:
    isTrue, frame =capture.read()
    cv.imshow('Video',frame)
    
    if cv.waitKey(20)==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

capture=cv.VideoCapture('photos/vtest.avi')

while True:
    isTrue, frame =capture.read()
    cv.imshow('Video',frame)
    k=cv.waitKey(20)
    if k==27:
        break
    
       
capture.release()
cv.destroyAllWindows()

#3#####################333############################3

import cv2 as cv
import numpy as np

blank=np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank',blank)
cv.waitKey(0)

#paint the image a certain color
blank[200:300,300:400]=100,100,255#bgr
cv.imshow('color',blank)
cv.waitKey(0)

#draw a rectangle

cv.rectangle(blank,(0,0), (blank.shape[1]//2,blank.shape[0]//2),
             (0,255,0),thickness=-1)
cv.imshow('Rectangle',blank)
cv.waitKey(0)

#Draw a cicle

cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2),
          40,(0,0,255),thickness=2)
cv.imshow('circle',blank)
cv.waitKey(0)

#Draw a line\
    
cv.line(blank,(100,250),(300,400),(255,255,255), thickness=3)
cv.imshow('line',blank)
cv.waitKey(0)

#write text

cv.putText(blank, 'Hello selmon boi',(0,225),
           cv.FONT_HERSHEY_TRIPLEX, 1.0,(0,255,0),2)
cv.imshow('Text', blank)
cv.waitKey(0)


#################################################################
##############################################################

import cv2 as cv
import numpy as np

full=np.zeros((800,800,4))

cv.imshow('Full',full)
cv.waitKey(0)

cv.rectangle(full,(250,250), (full.shape[1]//2, full.shape[0]//2),
             (0,255,255), thickness=1)
cv.imshow('rect',full)
cv.waitKey(0)

cv.line(full,(800,10),(400,250),(0,255,255),thickness=3)
cv.imshow('rect',full)
cv.waitKey(0)

cv.line(full,(300,5),(200,125),(0,255,255),thickness=3)
cv.imshow('rect',full)
cv.waitKey(0)

cv.line(full,(50,15),(200,125),(0,255,255),thickness=3)
cv.imshow('rect',full)
cv.waitKey(0)

###############################################################################

import cv2 as cv
import numpy as np
img= cv.imread('photos/panda.jpg')
cv.imshow('panda',img)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)


threshold, thresh=cv.threshold(gray, 150, 255,cv.THRESH_BINARY)
cv.imshow('Simple threshold',thresh)

threshold, thresh_inv=cv.threshold(gray, 150,255,cv.THRESH_BINARY_INV)
cv.imshow('Simple threshold',thresh_inv)

adaptive_thresh=cv.adaptiveThreshold(gray,255,
                                    cv.ADAPTIVE_THRESH_GAUSSIAN_C
                                    ,cv.THRESH_BINARY_INV,11,9)
cv.imshow('Simple threshhold',adaptive_thresh)
cv.waitKey(0)

#############################################################################3

import cv2 as cv
import numpy as np

img=cv.imread('photos/panda.jpg')
cv.imshow('cat',img)

def translate(img, x ,y):
    transMat =np.float32([[1,0,x],[0,1,y]])
    print(transMat)
    dimensions=(img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x  left
# -y up
# x right
# y down

translated=translate(img, -100,-100)
cv.imshow('Translate',translated)

cv.waitKey(0)

##############################################################################

import cv2 as cv
import numpy as np

img=cv.imread('photos/panda.jpg')
cv.imshow('cat',img)

def rotate(img,angle, rotPoint=None):
    (height,width)=img.shape[:2]
    
    if rotPoint is None:
        rotPoint=(width//2,height//2)
        rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
        dimensions=(width,height)
        
        return cv.warpAffine(img, rotMat, dimensions)

rotated=rotate(img, -45)
cv.imshow('rotated', rotated)
cv.waitKey(0)

rotated_rotated=rotate(img, -90)
cv.imshow('rotated',rotated_rotated)
cv.waitKey(0)

rotated_rotated=rotate(rotated, -90)
cv.imshow('rotated',rotated_rotated)
cv.waitKey(0)

###############################################################################

#Flipping

flip=cv.flip(img,1)
cv.imshow('flip',flip)
cv.waitKey(0)

#############################################################################

import cv2 as cv
import numpy as np

img =cv.imread('photos/panda.jpg')
cv.imshow('panda',img)

blank=np.zeros(img.shape, dtype='uint8')
cv.imshow('blank',blank)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

blur=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)

canny=cv.Canny(blur,125,175)
cv.imshow('canny edges', canny)

contours, hierarchies =cv.findContours(canny,cv.RETR_LIST,
                                       cv.CHAIN_APPROX_SIMPLE)

print(len(contours),'contour(s) found')

cv.drawContours(blank, contours, -1,(0,0,255),2)
cv.imshow('contours',blank)

cv.waitKey(0)


##################################################################

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img =cv.imread('photos/panda.jpg')
cv.imshow('panda',img)
cv.waitKey(0)

plt.imshow(img)
plt.show()

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv',hsv)

lab=cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('lab',lab)

cv.waitKey(0)

rgb=cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('gray',rgb)
cv.waitKey(0)

hsv_bgr=cv.cvtColor(img, cv.COLOR_HSV2BGR)
cv.imshow('hsv',hsv_bgr)
cv.waitKey(0)

lab_bgr=cv.cvtColor(img, cv.COLOR_LAB2BGR)
cv.imshow('lab',lab_bgr)
cv.waitKey(0)

##################################################################################


import os
print(os.name)


import os
print(os.getcwd())

os.path.abspath('.')#to find the absolute path

os.listdir('.')#to print files and directories in the current directory

###############################################################################
#saving of the image 

import cv2 as cv
import os

img_path =r'D:\python class\Open cv\photos\panda.jpg'

directory=r'D:\python class\Open cv\photos\ff'

img=cv.imread(img_path)
os.chdir(directory)

print("before saving the image")
print(os.listdir(directory))

filename='savedImage.jpg'

cv.imwrite(filename, img)

print('After saving image')
print(os.listdir(directory))

print("successfullly saved")

##############################################################################

import cv2 as cv
vid=cv.VideoCapture(0)

while(True):
    ret, frame=vid.read()
    cv.imshow('frame',frame)
    
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
    
vid.release()
cv.destroyAllWindows()

################################################################################
video =cv.VideoCapture(0)
if(video.isOpened()==False):    
    print("Error reading file")
    
frame_width=int(video.get(3))
frame_height=int(video.get(4))

size=(frame_width, frame_height)


result=cv.VideoWriter('ff/files.avi',cv.VideoWriter_fourcc(*'MJPG'),10, size)

while(True):
    ret,frame=video.read()
    
    if ret==True:
        
        result.write(frame)
        cv.imshow('Frame',frame)
        
        if cv.waitKey(1) & 0xFF ==ord('s'):
            break
    else:
        break
        
video.release()
result.release()

cv.destroyAllWindows()

print("saved successfully")

###############################################################################

import cv2 as cv
vid=cv.VideoCapture(0)

while(True):
    ret, frame=vid.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    cv.imshow('frame',frame)
    cv.imshow('gray',gray)
    cv.imshow('frame',rgb)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
    
vid.release()
cv.destroyAllWindows()




import cv2 as cv
vid=cv.VideoCapture(0)

while(True):
    ret, frame=vid.read()
    rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    cv.imshow('frame',frame)
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
    
vid.release()
cv.destroyAllWindows()

###############################################################################

cap=cv.VideoCapture(0)

check, vid=cap.read()
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))

size=(frame_width, frame_height)

counter=0

check=True

frame_list=[]

while(check==True):
    cv.imwrite("frame%d.jpg"%counter, vid)
    check, vid=cap.read()
    
    frame_list.append(vid)
    
    counter +=1
    
frame_list.pop()
for frame in frame_list:
    cv.imshow('frmae',frame)
    
    if cv.waitKey(25) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()

frame_list.reverse()

for frame in frame_list:
    cv.imshow('frame',frame)
    if cv.waitKey(25) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()


