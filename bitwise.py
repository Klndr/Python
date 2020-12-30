import cv2 as cv
import numpy as np

blank=np.zeros((400,400),dtype='uint8')
rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
cv.imshow('rectangle',rectangle)


circle=cv.circle(blank.copy(),(200,200),200,255,-1)
cv.imshow('circle',circle)

#-> bitwise and intersecting regions

bitwise_and = cv.bitwise_and(rectangle,circle)

cv.imshow('bitwise_and',bitwise_and)

bitwise_or=cv.bitwise_or(rectangle,circle)
cv.imshow('bitwise_or',bitwise_or)

bitwise_xor=cv.bitwise_xor(rectangle,circle)
cv.imshow('bitwise_xor',bitwise_xor)

bitwise_not=cv.bitwise_not(circle)
cv.imshow('bitwise_not',bitwise_not)
cv.waitKey(0)


##############################################################################

img=cv.imread('photos/panda.jpg')
cv.imshow('panda',img)
cv.waitKey(0)
blank=np.zeros(img.shape[:2],dtype='uint8')

circle=cv.circle(blank.copy(),(img.shape[1]//2,img.shape[0]//2),100,255,-1)

rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)

weird_shape=cv.bitwise_and(circle,rectangle)
cv.imshow('wierd shape',weird_shape)
cv.waitKey(0)

shape_or=cv.bitwise_or(circle,rectangle)
cv.imshow('shape xor',shape_or)
cv.waitKey(0)

masked=cv.bitwise_and(img,img,mask=shape_or)
cv.imshow('weird shaped masked image',masked)
cv.waitKey(0)



cv.waitKey(0)

import cv2 as cv
import numpy as np
img1=cv.imread('photos/cute1.jpg',1)
#cv.imshow('panda3',img1)

img2=cv.imread('photos/cute2.jpg',1)
#cv.imshow('panda1',img2)


result=cv.addWeighted(img1,0.9,img2,0.4,0)
cv.imshow('result',result)
cv.waitKey(0)


import cv2 as cv
import numpy as np
def rescaleFrame(frame,scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
def changeres(width,height):
    capture.set(3,width)
    capture.set(4,height)
    
capture=cv.VideoCapture('photos/vtest.avi')

while True:
    isTrue, frame=capture.read()
    frame_resized =rescaleFrame(frame)

    
    cv.imshow('video',frame)
    cv.imshow('video resized',frame_resized)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

##########################################################################

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('photos/cute1.jpg')
cv.imshow('cats',img)

blank=np.zeros(img.shape[:2], dtype='uint8')

mask=cv.circle(blank, (img.shape[1]//2,img.shape[0]//2),
               100,255,-1)

masked=cv.bitwise_and(img,img,mask=mask)
cv.imshow('mask',masked)
cv.waitKey(0)

plt.figure()
plt.title('color Histogram')
plt.xlabel('bins')
plt.ylabel('#of pixels')
colors=('b','g','r')

for i,col in enumerate(colors):
    hist=cv.calcHist([img],[i],mask,[256],[0,256])
    print(hist)
    plt.plot(hist,color=col)
    plt.xlim([0,256])

plt.show()

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
cv.waitKey(0)

gray_hist=cv.calcHist([gray],[0],mask,[256],[0,256])
plt.figure()
plt.title('grayscale Histogram')
plt.xlabel('bins')
plt.ylabel('#of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()


def fibonacci(n):
    a,b=0,1
    while a<n:
        print(a)
        a,b=b,a+b
    print()
fibonacci(10)
        
        
a,b=0,1
while a<10:
    print (a)
    a,b=b,a+b
            
    
