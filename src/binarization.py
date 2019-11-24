
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

# find if the image is a screen shot
def find_scan_screenshot(img):
    # return 0 for scan/screenshot and 1 for photograph
    total_pixels = img.shape[0]*img.shape[1]
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    prop = np.sum(hist[15:241])
    if prop<0.1*total_pixels:
        return 0
    else:
        return 1

# Threshold screenshots and binarize 
def binarization_scans(img):
    # using inbuilt otsu's method.
    _,im = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return im

#threshold photographs and binarize
def binarization_photos(img):
    high_res = 0
    if img.size>2000*1000:
        high_res = 1
    
    if high_res==1:
        img = cv2.GaussianBlur(img,(9,9),3) # fails for even
    
    window_size = int(min(img.shape[0],img.shape[1])/60)
    if window_size%2==0:
        window_size+=1 #cv2.adaptiveThreshold accepts only odd window sizes
    
    thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,window_size,10)
    return thresh_img

#use morphological opening and closing to remove noise and small holes 
def morph_proc(img):
    img = 255 - img
    img[img==255] = 1
    kernel = np.ones((3,3))
    kernel1 = np.ones((5,5))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) 
#     filled = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel1) 
    fin_img = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel1) 
    hole_size = int(0.0001 * img.shape[0] * img.shape[1])
    arr = fin_img>0
    fin_img = remove_small_objects(arr, min_size=hole_size)
    fin_img = fin_img.astype(np.uint8)
    fin_img[fin_img==1] = 255
    return 255-fin_img

#run binarization on input based on type
def binarize_input(img):
    img_type = find_scan_screenshot(img)
    if img_type==0:
        print("Image type: Scan/Screenshot")
    else:
        print("Image type: Photograph")
    if img_type==0:
        new_img = binarization_scans(img)
    else:
        new_img = binarization_photos(img)
        new_img = morph_proc(new_img)
    return new_img