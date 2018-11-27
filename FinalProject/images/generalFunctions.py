import numpy as np
import cv2
import matplotlib.pyplot as plt

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

def detectColor(img,color,colorModel):
    """
    Given an image detects the given color. Return the mask.
    :param img: numpy Array
    :param color: tuple
    :param colorModel: int 0:HSV, 1:HSL
    :return: numpy Array
    """
    if colorModel == 0:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img,color[0],color[1])
        img = cv2.bitwise_and(img,50,200)
        img = np.asarray(img,dtype='float32')/255.0
    else:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSL)
        mask = cv2.inRange(img,color[0],color[1])
        img = cv2.bitwise_and(img,50,200)
        img = np.asarray(img,dtype='float32')/255.0

    return img

def detectColors(image,colors,colorModel):
    """
    Given an image detects the given colors. Return the mask.
    :param image: numpy Array
    :param colors: list
    :param colorModel: int 0:HSV, 1:HSL
    :return: numpy Array
    """
    result = np.zeros_like(image,'float32')
    if colorModel == 0:
        for color in colors:
            img = image.copy()
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(img,color[0],color[1])
            img = cv2.bitwise_and(img,50,200)
            img = np.asarray(img,dtype='float32')/255.0
            result += img
        else: 
            for color in colors:
                img = image.copy()
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(img,color[0],color[1])
                img = cv2.bitwise_and(img,50,200)
                img = np.asarray(img,dtype='float32')/255.0
                result += img
            
    return result

def resize(img,scale):
    """
    Given an image returns a resized image. 
    :param img: numpy Array
    :param scale: float or int
    :return: numpy Array
    """
    row = img.shape[0]
    col = img.shape[1]

    return cv2.resize(img,(row*scale,col*scale))

def rotate(img,angle):
    """
    Given an image returns a rotated image. 
    :param img: numpy Array
    :param angle: float or int
    :return: numpy Array
    """
    row = img.shape[0]
    col = img.shape[1]

    return cv2.warpAffine(img,cv2.getRotationMatrix2D((col,row),angle,1),(col*2,row*2))

def resizeAndRotate(img,scale,angle):
    """
    Given an image returns a resized and rotated image. 
    :param img: numpy Array
    :param scale: float or int
    :param angle: float or int
    :return: numpy Array
    """
    row = img.shape[0]
    col = img.shape[1]

    res = cv2.resize(img,(row*scale,col*scale))
    return cv2.warpAffine(res,cv2.getRotationMatrix2D((col,row),angle,1),(col*scale,row*scale))

