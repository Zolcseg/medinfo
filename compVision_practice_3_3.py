import numpy as np
import cv2
import utils as u
import sys
import os
from datetime import datetime

dirname = os.path.dirname(__file__)

if __name__ == "__main__":

    # setting default picture name to use
    if len(sys.argv) == 1: sys.argv[1:] = ["lena"]
    img_name = sys.argv[1]

    # loading the image
    img_path = 'images/' + img_name + '.jpg'
    cv_img = cv2.imread(img_path,1)

    if cv_img is None:
        print("---> File " + img_name + " can't be loaded...quitting...")
        exit(0)

    # creating simple histogram and show/save the histogram image
    hist_img = u.create_histogram_image(cv_img)
    cv2.imshow('original image',cv_img)
    cv2.imshow('orig histo',hist_img)
    cv2.imwrite('images/' + img_name + '_orig_histo.jpg',hist_img)

    # converting to greyscale and create histogram and show/save the new image+histo
    grey_img = u.convert2grey(cv_img)
    grey_histo = u.create_grey_histogram_image(grey_img)
    cv2.imshow('grey image', grey_img)
    cv2.imshow('grey histo',grey_histo)
    cv2.imwrite('images/' + img_name + '_grey_img.jpg',grey_img)
    cv2.imwrite('images/' + img_name + '_grey_histo.jpg',grey_histo)

    # applying openCV equalizeHist method and create histogram and show/save the new image+histo
    eqHist_img = u.equalize_hist_img(cv_img)
    eqHist_histo = u.create_grey_histogram_image(u.equalize_hist_img(cv_img))
    cv2.imshow('grey eq image (cv.equalizeHist method)', eqHist_img)
    cv2.imshow('grey eq histo (cv.equalizeHist method)',eqHist_histo)
    cv2.imwrite('images/' + img_name + '_cvEqualize_img.jpg',eqHist_img)
    cv2.imwrite('images/' + img_name + '_cvEqualize_histo.jpg',eqHist_histo)

    # applying our own built linear stretching method for greyscale histogram normalization
    # and create histogram and show/save the new image+histo
    cust_histo , cust_img = u.linear_stretch_hist_img_and_img(cv_img)
    cv2.imshow('grey custom image', cust_img)
    cv2.imshow('grey custom histo',cust_histo)
    cv2.imwrite('images/' + img_name + '_linearStretch_img.jpg',cust_img)
    cv2.imwrite('images/' + img_name + '_linearStretch_histo.jpg',cust_histo)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # ------------------------------------------ #
    # ---- Bitwise operations for inverting ---- #
    # ------------------------------------------ #

    color = ('b', 'g', 'r')

    # inverting colors by deducting color code from 255 using numpy array calcs
    starttime = datetime.now()
    img2 = 255 - cv_img
    endtime = datetime.now()
    t = endtime - starttime
    pr_time2 = str(float(t.seconds + t.microseconds/1000000.0)) + 'sec '

    # inverting colors by bitwise_not
    starttime = datetime.now()
    img3 = cv2.bitwise_not(cv_img)
    endtime = datetime.now()
    t = endtime - starttime
    pr_time3 = str(float(t.seconds + t.microseconds / 1000000.0)) + 'sec '

    # inverting colors by scanning through pixels
    starttime = datetime.now()
    img4 = np.ndarray(cv_img.shape, dtype=cv_img.dtype)
    for x in xrange(cv_img.shape[0]):
        for y in xrange(cv_img.shape[1]):
            blue = cv_img[x, y][0]
            green = cv_img[x, y][1]
            red = cv_img[x, y][2]
            img4[x, y][0] = 255 - blue
            img4[x, y][1] = 255 - green
            img4[x, y][2] = 255 - red
    endtime = datetime.now()
    t = endtime - starttime
    pr_time4 = str(float(t.seconds + t.microseconds / 1000000.0)) + 'sec '

    # showing images
    cv2.imshow('original img', cv_img)
    cv2.imshow(pr_time2 + ' - onestep - inverting by one step matrix calc', img2)
    cv2.imshow(pr_time3 + ' - cv.bitwise_not - inverting by openCV bitwise method', img3)
    cv2.imshow(pr_time4 + ' - scan through - inverting by scanning through each pixel', img4)

    cv2.waitKey(0)

