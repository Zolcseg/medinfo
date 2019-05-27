import numpy as np
import cv2
import utils as u

import os
dirname = os.path.dirname(__file__)

img_name = 'lena'


#img_path = os.path.join(dirname, 'images/' + img_name + '.jpg')
img_path = 'images/' + img_name + '.jpg'
cv_img = cv2.imread(img_path,1)

hist_img = u.create_histogram_image(cv_img)

cv2.imshow('original image',cv_img)
cv2.imshow('orig histo',hist_img)
cv2.imwrite('images/' + img_name + '_orig_histo.jpg',hist_img)

grey_img = u.convert2grey(cv_img)
grey_histo = u.create_grey_histogram_image(grey_img)
cv2.imshow('grey image', grey_img)
cv2.imshow('grey histo',grey_histo)
cv2.imwrite('images/' + img_name + '_grey_img.jpg',grey_img)
cv2.imwrite('images/' + img_name + '_grey_histo.jpg',grey_histo)

eqHist_img = u.equalize_hist_img(cv_img)
eqHist_histo = u.create_grey_histogram_image(u.equalize_hist_img(cv_img))
cv2.imshow('grey eq image (cv.equalizeHist method)', eqHist_img)
cv2.imshow('grey eq histo (cv.equalizeHist method)',eqHist_histo)
cv2.imwrite('images/' + img_name + '_cvEqualize_img.jpg',eqHist_img)
cv2.imwrite('images/' + img_name + '_cvEqualize_histo.jpg',eqHist_histo)

cust_histo , cust_img = u.linear_stretch_hist_img_and_img(cv_img)
cv2.imshow('grey custom image', cust_img)
cv2.imshow('grey custom histo',cust_histo)
cv2.imwrite('images/' + img_name + '_linearStretch_img.jpg',cust_img)
cv2.imwrite('images/' + img_name + '_linearStretch_histo.jpg',cust_histo)

cv2.waitKey(0)