import numpy as np
import argparse
#import imutils
import glob
import cv2
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb
from scipy import signal

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
green = [120,255,255]
red = [0,255,255]
blue = [240,255,255]
yellow = [60,255,255]

lightyellow = [73,255,255]
darkyellow = [47,255,255]

lightred = [13,255,255]
darkred = [0,201,201]

lightblue = [110,50,50]
darkblue = [130,255,255]

lightgreen= np.asarray([110,50,50])
darkgreen = np.asarray([130,255,255])

def prepareImage(image,lowerColorRange,upperColorRange):
    img = cv2.imread(image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    ##img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(img,lowerColorRange,upperColorRange)
    
    #cv2.imshow("CASKINGO",mask)
    #img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.Canny(img,100,200)

    img = np.asarray(img,dtype='float32')/255.0
    return img 


#img = prepareImage('imagen.png',lightWhite,darkWhite)
#template = prepareImage('imagen1.png',lightWhite,darkWhite)
img = prepareImage('blue chainshaw pic.png',lightgreen,darkgreen)

template = prepareImage('blue chainshaw.png',lightgreen,darkgreen)
print("ya entre")

imgR = signal.correlate2d(img,template,boundary='symm',mode='same')
print("ya sali")
y,x=np.unravel_index(np.argmax(imgR),imgR.shape)
print(x,y)

bw = 200
bh = 50

img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.rectangle(img,(x-bw,y-bh),(x+bw,y+bh),(0,0,255),3)
cv2.rectangle(img,(x-bw,y-bh),(x+bw,x+bh),(0,0,255),3)
plt.imshow(img,cmap='gray')
plt.show()
cv2.imshow('Detection', img)
