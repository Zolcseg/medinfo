import cv2
import numpy as np

def create_image_from_histogram(cvimg):

    width = 256#cvimg.shape[1]
    h = np.zeros((cvimg.shape[0],width,3),dtype=cvimg.dtype )#np.zeros(cvimg.shape,dtype=cvimg.dtype)

    bins = np.arange(width).reshape(width,1)
    color = [(255,0,0),(0,255,0),(0,0,255)]

    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([cvimg], [ch], None, [width], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    h_image=np.flipud(h)

    return h_image