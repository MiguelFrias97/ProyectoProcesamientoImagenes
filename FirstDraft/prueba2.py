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

    colors = determineRanges()
    tools = [['blue chainsaw.png','blue hammer.png'],['yellow chainsaw.png','yellow spring clamp.png'],['green shovel.png','green spring clamp.png'],['red shovel.png','red hammer.png']]
    img_name = captureImage()
    #img_name = 'blue chainsaw pic.png'
    img, color = prepareImage(img_name,colors)
    row = img.shape[0]
    col = img.shape[1]
    tool = detectTool(img,tools,colors,color)
    
    tool_type = tool.split(".")[0]
    
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
    print(tool_type)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img,(col*2,row*2))
    plt.imshow(img,cmap='gray')
    plt.show()
    
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
        row = img.shape[0]
        col = img.shape[1]
        for p1 in range(row):
            for p2 in range(col):
                if imgR[p1,p2] != 0:
                    value_Count[i]+=1
    tool_value = np.argmin(value_Count)
    tool = tools[color][tool_value]
    return tool

def prepareImage(image,colors):
    a = cv2.imread(image)
    row = a.shape[0]//2
    col = a.shape[1]//2
    img = cv2.resize(a, (col,row))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask, color = detectColor(img, colors)
    #mask = cv2.inRange(img,colors[color][0],colors[color][1])
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
    value_Count = []
    trys = []
    for i in range(len(colors)):
        value_Count.append(0)
        attempt = cv2.inRange(img,colors[i][0],colors[i][1])
        trys.append(attempt)
        row = attempt.shape[0]
        col = attempt.shape[1]
        for x in range(row):
                for y in range(col):
                    if attempt[x,y] != 0:
                        value_Count[i]+=1
    color_value = np.argmax(value_Count)
    attempt = trys[color_value]
    return attempt, color_value

def determineRanges():
    lightyellow = np.asarray([25,50,50])
    darkyellow = np.asarray([40,255,255])

    lightred = np.asarray([0,50,50])
    darkred = np.asarray([20,255,255])

    lightgreen = np.asarray([50,50,50])
    darkgreen = np.asarray([70,255,255])

    lightblue= np.asarray([110,50,50])
    darkblue = np.asarray([130,255,255])

    colors = [[lightblue,darkblue],[lightyellow,darkyellow],[lightgreen,darkgreen],[lightred,darkred]]
    return colors
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
