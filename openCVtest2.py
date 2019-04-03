import numpy as np
import cv2
from histogram import create_image_from_histogram

import os
dirname = os.path.dirname(__file__)


img_path = os.path.join(dirname, 'images/lena.jpg')

cv_img = cv2.imread(img_path,1)

hist_img = create_image_from_histogram(cv_img)


cv2.imshow('colorhist',hist_img)

cv2.imshow('original image',cv_img)
cv2.waitKey(0)