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
    
    objetoBuscar = cv2.imread('blue hammer.png')
    

    for tool in tools:
        imgEntrada = cv2.imread(tool)

        

    
