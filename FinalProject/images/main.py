import numpy as np
import argparse
#import imutils
import glob
import cv2
import matplotlib.pyplot as plt
import csv

from matplotlib.colors import hsv_to_rgb
from scipy import signal
from generalFunctions import *

if __name__="__main__":
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    green = [120,255,255]
    red = [0,255,255]
    blue = [240,255,255]
    yellow = [60,255,255]

    lightyellow = np.asarray([25,50,50])
    darkyellow = np.asarray([45,255,255])

    lightred = np.asarray([0,50,50])
    darkred = np.asarray([20,255,255])

    lightgreen = np.asarray([50,50,50])
    darkgreen = np.asarray([70,255,255])

    lightblue= np.asarray([110,50,50])
    darkblue = np.asarray([130,255,255])

    colors = [[lightblue,darkblue],[lightyellow,darkyellow],[lightgreen,darkgreen],[lightred,darkred]]
    tools = [['blue hammer.png','blue chainsaw.png'],['yellow chainsaw.png'],['green shovel.png'],['red shovel.png','red hammer.png']]


    #img = prepareImage('imagen.png',lightWhite,darkWhite)
    #template = prepareImage('imagen1.png',lightWhite,darkWhite)
    img_name = captureImage()
    img, color = prepareImage(img_name,colors)

    tool, x, y = detectTool(img,tools,colors,color)
    print(tool,x,y)
    try:
        tools = openCSV('tools.csv')
        if tool in tools:
            tools[tool] += 1
        else:
            tools[tool] = 1
    except:
        tools = {tool:1}
        writeCSV('tools.csv',tools)

    bw = 10
    bh = 10

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img,(x-bw,y-bh),(x+bw,y+bh),(0,0,255),3)
    cv2.rectangle(img,(x-bw,y-bh),(x+bw,x+bh),(0,0,255),3)
    plt.imshow(img,cmap='gray')
    plt.show()
    img = cv2.resize(img,(col*3,row*3))
    cv2.imshow('Detection', img)
    


