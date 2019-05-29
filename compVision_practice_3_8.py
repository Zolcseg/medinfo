import numpy as np
import cv2
import utils as u
import kernels
import sys
import os
from datetime import datetime

dirname = os.path.dirname(__file__)

backSub1 = cv2.createBackgroundSubtractorMOG2()

backSub2 = cv2.createBackgroundSubtractorKNN()

backSub = backSub2

kernel = kernels.get_kernel("gaussian")


def components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    ret = cap.set(3, 640)
    ret = cap.set(4, 480)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        histo = u.create_histogram_image(frame)

        img = cv2.threshold(u.convert2grey(frame), 127, 255, cv2.THRESH_BINARY)[1]
        ret, labels = cv2.connectedComponents(img)

        labeled_img = components(labels)

        #grey_img = u.convert2grey(frame)
        #output = grey_img
        #output = u.convolve(output,kernel)
        #output = cv2.filter2D(output, -1, kernel)

        # Our operations on the frame come here
        #gray = u.convert2grey(frame)
        #histo = u.create_grey_histogram_image(gray)

        #histo1, cust_gray = u.linear_stretch_hist_img_and_img(frame)

        #cust_gray2 = u.equalize_hist_img(frame)
        #histo2 = u.create_grey_histogram_image(cust_gray2)

        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Display the resulting frame
        cv2.imshow('frame - press q to quit', frame)
        cv2.imshow('histo - press q to quit', histo)
        cv2.imshow('Connected Component', labeled_img)
        cv2.imshow('FG Mask', fgMask)
        #cv2.imshow('convolved grey', output)
        #cv2.imshow('eq1 gray - press q to quit', cust_gray)
        #cv2.imshow('eq1 histo - press q to quit', histo1)
        #cv2.imshow('eq2 gray - press q to quit', cust_gray2)
        #cv2.imshow('eq2 histo - press q to quit', histo2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()