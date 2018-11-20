import cv2
import matplotlib.pyplot as plt
img = cv2.imread("red shovel.png",1)
a = img
row = a.shape[0]
col = a.shape[1]


c = cv2.resize(a, (row*2,col*2))
M = cv2.getRotationMatrix2D((col,row),180,1)
b = cv2.warpAffine(c,M,(col*2,row*2))

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
b = cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
c = cv2.cvtColor(c,cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()
