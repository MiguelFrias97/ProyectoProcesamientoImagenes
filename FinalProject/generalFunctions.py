import numpy as np
import cv2
import matplotlib.pyplot as plt

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

def detectColor(img,color):
    """
    From an image detects the given color. Return the mask.
    :param img: numpy Array
    :param color: tuple
    :return: numpy Array
    """
    img = cv2.cvtcolor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtcolor(img,cv2.COLOR_RGB2HSL)
    mask = cv2.inRange(img,color[0],color[1])
    img = cv2.bitwise_and(img,50,200)
    img = np.asarray(img,dtype='float32')/255.0

    return img

def detectColors(image,colors):
    """
    From an image detects the given colors. Return the mask.
    :param img: numpy Array
    :param colors: list
    :return: numpy Array
    """
    result = np.zeros_like(image,'float32')
    for color in colors:
        img = image.copy()
        img = cv2.cvtcolor(img,cv2.COLOR_BGR2RGB)
        img = cv2.cvtcolor(img,cv2.COLOR_RGB2HSL)
        mask = cv2.inRange(img,color[0],color[1])
        img = cv2.bitwise_and(img,50,200)
        img = np.asarray(img,dtype='float32')/255.0
        result += img
        
    return result
