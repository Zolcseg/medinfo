import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread(r'C:\Users\sepsi\Desktop\Python\CompVision\medinfo\images\lena.jpg')
color = ('b', 'g', 'r')

# inverting colors by deducting color code from 255
img2 = 255 - img

# inverting colors by bitwise_not
img3 = cv2.bitwise_not(img)

# inverting colors by scanning through pixels
img4 = np.ndarray(img.shape,dtype=img.dtype)
for x in xrange(img.shape[0]):
    for y in xrange(img.shape[1]):
        blue = img[x,y][0]
        green = img[x,y][1]
        red = img[x,y][2]
        img4[x,y][0] = 255 - blue
        img4[x,y][1] = 255 - green
        img4[x,y][2] = 255 - red


# making image grayscale
img5 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# equalizing histogram for picture
img6 = cv2.equalizeHist(img5)

# showing images
cv2.imshow('original img',img)
cv2.imshow('inverting2',img2)
cv2.imshow('inverting3',img3)
cv2.imshow('inverting4',img4)

cv2.imshow('grayscale',img5)

cv2.imshow('equalized',img6)

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

for i, col in enumerate(color):
    histr2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
    plt.plot(histr2, color=col)
    plt.xlim([0, 256])

plt.show()

histr3 = cv2.calcHist([img5], [0], None, [256], [0, 256])
histr4 = cv2.calcHist([img6], [0], None, [256], [0, 256])
plt.plot(histr3, color='r')
plt.plot(histr4, color='b')
plt.xlim([0, 256])

plt.show()

