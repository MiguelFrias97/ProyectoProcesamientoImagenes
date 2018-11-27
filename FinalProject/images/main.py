from generalFunctions import *

import cv2
import subprocess as sbp

def executeBashCommand(command):
    process = sbp.Popen(command.split(), stdout=sbp.PIPE)
    out,error = process.communicate()
    return out

if __name__ == "__main__":
    tools = executeBashCommand('ls').decode('utf-8').split()
    tools.pop(tools.index('main.py'))
    tools.pop(tools.index('generalFunctions.py'))
    
    blueHSV = (np.array([240,3,100]),np.array([219,100,40]))
    objSearch = cv2.imread('blue hammer.png')
    objHSV = detectColor(objSearch,blueHSV,0)

    objHSVContour = cv2.findContours()

    for tool in tools:
        imgIn = cv2.imread(tool)

        

    
