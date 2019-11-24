import cv2
import os
import copy
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage.feature import canny
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line, rotate
import math
from scipy import ndimage
from matplotlib.path import Path


#skew correction for scanned
def skew_correction_scanned(img):
    neg_img = 255 - img
    
    img_edges = canny(neg_img, sigma=7)
    #plt.imshow(img_edges,cmap='gray')
    #plt.show()
    h, theta, d = hough_line(img_edges)
    h, theta, d = hough_line_peaks(h, theta, d, min_distance=0, min_angle=0, num_peaks=4)
    theta = [int(np.round(math.degrees(i))) for i in theta]    
    #print(theta)
    # get dominant orientati0on
    dom_orient = max(theta)
    deskewing_angle = dom_orient - 90

    while abs(deskewing_angle) > 45: 
        deskewing_angle = deskewing_angle - (abs(deskewing_angle)/deskewing_angle) * 90

    #print(deskewing_angle)

    #rotate image by deskewing angle
    deskew_img = skimage.transform.rotate(neg_img, deskewing_angle, mode='constant', cval=0, preserve_range=True, clip=True)
    
    deskew_img = 255-deskew_img
    # do morphological opening with square 3X3 structuring element
    str_ele = skimage.morphology.selem.square(3)
    deskew_img = skimage.morphology.binary_opening(deskew_img,selem=str_ele)
    return deskew_img.astype(np.uint8)


#skew correction for photos
def skew_correction(img):
    neg_img = 255 - img
    
    img_edges = canny(neg_img, sigma=7)
    #plt.imshow(img_edges,cmap='gray')
    #plt.show()
    h, theta, d = hough_line(img_edges)
    h, theta, d = hough_line_peaks(h, theta, d, min_distance=0, min_angle=0, num_peaks=4)
    theta = [int(np.round(math.degrees(i))) for i in theta]    
    #print(theta)

    dom_orient = max(theta)
    deskewing_angle = dom_orient - 90

    while abs(deskewing_angle) > 45: 
        deskewing_angle = deskewing_angle - (abs(deskewing_angle)/deskewing_angle) * 90

    #print(deskewing_angle)
    #rotate image by deskewing angle

    deskew_img = skimage.transform.rotate(neg_img, deskewing_angle, mode='constant', cval=0, preserve_range=True, clip=True)
    
    deskew_img = 255-deskew_img
    # close image by using square 3x3 structuring element and then dilate with 5x5 structuring element 
    str_ele = skimage.morphology.selem.square(3)
    str_ele1 = skimage.morphology.selem.square(5)
    deskew_img = skimage.morphology.binary_closing(deskew_img,selem=str_ele)
    deskew_img = skimage.morphology.binary_dilation(deskew_img, selem = str_ele1)
    return deskew_img.astype(np.uint8)