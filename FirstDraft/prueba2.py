import numpy as np
import argparse
#import imutils
import glob
import cv2
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb
from scipy import signal
import csv
import pandas as pd

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
    img_name = captureImage()
    img, color = prepareImage(img_name,colors)

    tool, x, y = detectTool(img,tools,colors,color)
    #print(tool,x,y)
    
    tool_type = tool.split(".")[0]
    print(tool_type)

    file_string = "test_file.csv"

    try:
        df = pd.read_csv(file_string)

        if tool_type in df.values:
            df.loc[df["Tool"] == tool_type, "Qty"] += 1
            df.to_csv(file_string, index=False)

        else:
            new_tool = {'Tool':tool_type, 'Qty':1}
            add2CSV(file_string, new_tool)
        
    except:
        new_tool = {'Tool':tool_type, 'Qty':1}
        writeCSV(file_string, new_tool)

    #bw = 10
    #bh = 10

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #cv2.rectangle(img,(x-bw,y-bh),(x+bw,y+bh),(0,0,255),3)
    #cv2.rectangle(img,(x-bw,y-bh),(x+bw,x+bh),(0,0,255),3)
    plt.imshow(img,cmap='gray')
    plt.show()
    #img = cv2.resize(img,(col*3,row*3))
    #cv2.imshow('Detection', img)
    
def detectTool(img,tools,colors, color):
    tool = ''
    value_Count = []
    point = []
    for i in range(len(tools[color])):
        value_Count.append(0)
    for i in range(len(tools[color])):
        print("Comparing to " + str(tools[color][i]))
        
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

##def openCSV(nombre):
##    dic = {}
##    with open(nombre, newline='') as csvfile:
##         spamreader = csv.reader(csvfile, delimiter=' ',)
##         for row in spamreader:
##             dic[row[0]] = int(row[1])
##    return dic

def add2CSV(file, tool):
    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['Tool', 'Qty']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(tool)

def writeCSV(file, tool):
    with open(file, 'w', newline='') as csvfile:
        fieldnames = ['Tool', 'Qty']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow(tool)
    
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

def captureImage():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Image Capture")
    while True:
        ret, frame = cam.read()
        cv2.imshow("Image Capture", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            img_name = "captured tool.png"
            cv2.imwrite(img_name, frame)
            break
    cam.release()
    cv2.destroyAllWindows()
    return img_name
main()
