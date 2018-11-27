from generalFunctions import *
from scipy import signal

import cv2
import subprocess as sbp

def executeBashCommand(command):
    process = sbp.Popen(command.split(), stdout=sbp.PIPE)
    out,error = process.communicate()
    return out

if __name__ == "__main__":
    tools = executeBashCommand('ls').decode('utf-8').split('\n')
    tools.pop(tools.index('main.py'))
    tools.pop(tools.index('generalFunctions.py'))
    tools.pop(tools.index('__pycache__'))
    
    blueHSV = (np.array([240,3,100]),np.array([219,100,40]))
    objSearch = cv2.imread('blue hammer.png')
    w, h,c = objSearch.shape
    objHSV = detectColor(objSearch,blueHSV,0)

    mask = np.zeros_like(objSearch)

    gray = cv2.cvtColor(objSearch,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0.1*gray.max(),255,0)
    _,contour,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    objSearchFinal = cv2.drawContours(mask, contour, -1, (137,109,46), 2)

    cv2.imshow('test',objSearchFinal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    for tool in tools:
        imgIn = cv2.imread('blue hammer.png')
        blueHSV = (np.array([240,3,100]),np.array([219,100,40]))
##        imgInHSV = detectColor(imgIn,blueHSV,0)

        mask = np.zeros_like(imgIn)

        gray = cv2.cvtColor(imgIn,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,0.1*gray.max(),255,0)
        _,contour,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        imgInComp = cv2.drawContours(mask, contour, -1, (137,109,46), 2)

        res = cv2.matchTemplate(objSearchFinal,imgInComp,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            print(pt)
            cv2.rectangle(objSearch, pt, (pt[0] + 50, pt[1] + 50), (0,0,255), 2)

        cv2.imshow('Result',objSearch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        

        

    
