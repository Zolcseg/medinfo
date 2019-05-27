import numpy as np
import cv2
import utils as u
import kernels
import sys
import os
from datetime import datetime

dirname = os.path.dirname(__file__)

if __name__ == "__main__":

    # setting default picture name to use
    if len(sys.argv) == 1:
        print("---> File can't be loaded...quitting...")
        exit(0)
    elif len(sys.argv) == 2:
        img_name = sys.argv[1]
        print("---> Kernel name is not provided...choose next time from this list: " + str(kernels.get_kernel_list()))
        exit(0)
    elif len(sys.argv) > 2:
        img_name = sys.argv[1]
        kernel_name = sys.argv[2]
        if len(sys.argv) == 4:
            kernel_iterations = int(sys.argv[3])
        else:
            kernel_iterations = 1

    # loading the image
    img_path = 'images/' + img_name + '.jpg'
    cv_img = cv2.imread(img_path,1)


    # -------------------------------------------- #
    # ---- Bitwise operations for convolution ---- #
    # -------------------------------------------- #

    #color = ('b', 'g', 'r')

    cv2.imshow('original img', cv_img)

    grey_img = u.convert2grey(cv_img)
    output = grey_img
    kernel = kernels.get_kernel(kernel_name)

    cv2.imshow('grey img', grey_img)

    for x in xrange(kernel_iterations):
        starttime = datetime.now()
        output = u.convolve(output,kernel)
        #output1 = cv2.filter2D(output,-1,kernel)
        endtime = datetime.now()
        t = endtime - starttime
        pr_time1 = str(float(t.seconds + t.microseconds / 1000000.0)) + 'sec '
        cv2.imshow('#' + str(x) + ' ' + pr_time1 + ' - convoluted img', output)
        #cv2.imshow('#' + str(x) + ' ' + pr_time1 + ' - convoluted img 2', output1)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print(str(cv2.Laplacian(output,cv2.CV_64F).var()))
    print(str(cv2.Laplacian(grey_img,cv2.CV_64F).var()))
