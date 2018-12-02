import numpy as np
import argparse
#import imutils
import glob
import cv2
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb
from scipy import signal

def main():
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
    img, color = prepareImage('blue hammer pic.png',colors)
    print('here')
    row = img.shape[0]
    col = img.shape[1]
    M = cv2.getRotationMatrix2D((col/2,row/2),0,1)
    img = cv2.warpAffine(img,M,(col,row))


    tool, x, y = detectTool(img,tools,colors,color)
    print(tool,x,y)

    bw = 10
    bh = 10

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img,(x-bw,y-bh),(x+bw,y+bh),(0,0,255),3)
    cv2.rectangle(img,(x-bw,y-bh),(x+bw,x+bh),(0,0,255),3)
    plt.imshow(img,cmap='gray')
    plt.show()
    img = cv2.resize(img,(col*3,row*3))
    cv2.imshow('Detection', img)
def detectTool(img,tools,colors, color):
    tool = ''
    value_Count = []
    point = []
    for i in range(len(tools[color])):
        value_Count.append(0)
    for i in range(len(tools[color])):
        print(tools[color][i])
        
        template, irrel = prepareImage(tools[color][i],colors)
        imgR = signal.correlate2d(img,template,boundary='symm',mode='full')
        
        y,x=np.unravel_index(np.argmax(imgR),imgR.shape)
        point.append([x,y])
        row = imgR.shape[0]
        col = imgR.shape[1]
        for p1 in range(row):
            for p2 in range(col):
                if imgR[p1,p2] != 0:
                    value_Count[i]+=1
    tool_value = np.argmin(value_Count)
    tool = tools[color][tool_value]
    x, y = point[tool_value][0],point[tool_value][1]
    return tool, x, y

def prepareImage(image,colors):
    a = cv2.imread(image)
    row = a.shape[0]//3
    col = a.shape[1]//3
    img = cv2.resize(a, (col,row))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    ##img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    color = detectColor(img, colors)
    mask = cv2.inRange(img,colors[color][0],colors[color][1])
    #cv2.imshow("CASKINGO",mask)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.Canny(img,100,200)

    img = np.asarray(img,dtype='float32')/255.0
    return img, color

def detectColor(img,colors):
    for i in range(len(colors)):
        attempt = cv2.inRange(img,colors[i][0],colors[i][1])
        row = attempt.shape[0]
        col = attempt.shape[1]
        color = True
        for x in range(row):
            if color:
                for y in range(col):
                    if attempt[x,y] != 0:
                        color = False
            else:
                value = i
                break
       
        if color == False:
            break
    return value
main()
