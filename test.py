
import numpy as np
import cv2
import utils as u
import kernels

img_path = 'images/lena2.jpg'
cv_img = cv2.imread(img_path, 1)
img = np.ndarray((200,200,3),dtype='uint8')

immm = u.convolve(cv_img,kernels.get_kernel("sharpen"))

img[0:50,:] = (45,50,55)
img[50:100,:] = (110,100,90)
img[100:150,:] = (170,150,190)
img[150:,:] = (255,255,255)

histo, img = u.linear_stretch_hist_img_and_img(img)

#histo = u.create_histogram_image(img)

cv2.imshow('grey img', img)
cv2.imshow('histo', histo)

cv2.waitKey(0)