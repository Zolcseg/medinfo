import numpy as np
import cv2
import utils as u

import os
dirname = os.path.dirname(__file__)


img_path = os.path.join(dirname, 'images/dark3.jpg')

cv_img = cv2.imread(img_path,1)


hist_img = u.create_histogram_image(cv_img)

cv2.imshow('original image',cv_img)
cv2.imshow('orig histo',hist_img)

cv2.imshow('grey image', u.convert2grey(cv_img))
cv2.imshow('grey histo',u.create_grey_histogram_image(u.convert2grey(cv_img)))

cv2.imshow('grey eq image', u.equalize_hist_img(cv_img))
cv2.imshow('grey eq histo',u.create_grey_histogram_image(u.equalize_hist_img(cv_img)))

cust_histo , cust_img = u.extended_hist_img_and_img(cv_img)

cv2.imshow('grey custom image', cust_img)
cv2.imshow('grey custom histo',cust_histo)

cv2.waitKey(0)