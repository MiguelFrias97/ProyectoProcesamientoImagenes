import numpy as np
import argparse
import imutils
import glob
import cv2
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb
from scipy import signal

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

lightWhite = (0,0,255)
darkWhite=(255,255,255)

def prepareImage(image,lowerColorRange,upperColorRange):
    img = cv2.imread(image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ##img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img,lowerColorRange,upperColorRange)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.Canny(img,50,200)
    img = np.asarray(img,dtype='float32')/255.0

    return img 


#img = prepareImage('imagen.png',lightWhite,darkWhite)
#template = prepareImage('imagen1.png',lightWhite,darkWhite)
img = prepareImage('imagen.png',lightWhite,darkWhite)
template = prepareImage('imagen1.png',lightWhite,darkWhite)

cv2.imshow("Template", template)
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgR = signal.correlate2d(img,template,boundary='symm',mode='same')

y,x=np.unravel_index(np.argmax(imgR),imgR.shape)
print(x,y)

bw = 50
bh = 50

img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.rectangle(img,(x-bw,y-bh),(x+bw,y+bh),(0,0,255),3)
##cv2.rectangle(img,(x-bw,y-bh),(x+bw,x+bh),(0,0,255),3)
#plt.imshow(img,cmap='gray')
#plt.show()
cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

