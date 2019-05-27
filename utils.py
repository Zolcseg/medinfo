import cv2
import numpy as np

def create_histogram_image(cvimg):

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

def create_grey_histogram_image(cvimg):

    width = 256#cvimg.shape[1]
    h = np.zeros((cvimg.shape[0],width,1),dtype=cvimg.dtype )#np.zeros(cvimg.shape,dtype=cvimg.dtype)

    bins = np.arange(width).reshape(width,1)
    color = [(255,0,0)]

    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([cvimg], [ch], None, [width], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    h_image=np.flipud(h)

    return h_image

def negate_img(cvimg):
    return 255 - cvimg

def bitwise_negate_img(cvimg):
    return cv2.bitwise_not(cvimg)

def convert2grey(cvimg):
    return cv2.cvtColor(cvimg,cv2.COLOR_BGR2GRAY)

def equalize_hist_img(cvimg):
    return cv2.equalizeHist(convert2grey(cvimg))

def linear_stretch_hist_img_and_img(cvimg):

    g_img = convert2grey(cvimg)
    width = 256
    grey_histo = cv2.calcHist([g_img], [0], None, [width], [0, 255])

    histmin = 0
    for i in range(0,255):
        if grey_histo[i] != 0:
            histmin = i
            break

    histmax = 255
    for i in range(255,0,-1):
        if grey_histo[i] != 0:
            histmax = i
            break

    tmp_grey = (g_img - histmin) * 255.0 / (histmax - histmin)
    cust_grey = np.uint8(np.around(tmp_grey))

    bins = np.arange(width).reshape(width,1)
    color = [(255,0,0)]
    h = np.zeros((cust_grey.shape[0], width, 1), dtype=cust_grey.dtype)
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([cust_grey], [ch], None, [width], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    custom_h_image=np.flipud(h)

    return custom_h_image, cust_grey