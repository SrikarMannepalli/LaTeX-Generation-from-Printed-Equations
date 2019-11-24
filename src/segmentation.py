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

def check_contains(bbox1, bbox2): 
    '''check if bbox2 is contained in bbox1'''
    if (bbox1[0]<bbox2[0]) and (bbox1[2]>bbox2[2]) and (bbox1[1]<bbox2[1]) and (bbox1[3]>bbox2[3]):
        return True
    return False

def segmentation(im):
    neg_im = 255-im
    kernel = np.ones((3,3), np.uint8) 
   # neg_im_erosion = cv2.erode(neg_im, kernel, iterations=1) 
    neg_im_erosion = neg_im
    #get edge map
    edge_map = neg_im_erosion # change to above
#     print('edge map')
#     plt.imshow(edge_map,cmap="gray")
#     plt.show()

    #do connected component analysis and obtain different character's bounding boxes,centroids, convex hulls
    labels = skimage.measure.label(edge_map, connectivity=2) #connectivity-2 => neighbours 8
    reg = skimage.measure.regionprops(labels)

    bboxs = [i.bbox for i in reg] #min_row, min_col, max_row, max_col
    centroids = [i.centroid for i in reg]
    convex_hulls = [i.convex_image for i in reg]
    imgs = [i.image for i in reg]
    ign_ind = []

    # checking if bounding box is contained, need to change to, convex hull being contained 
    for i in range(0,len(bboxs)): #checks if 'j' is contained in 'i'
        for j in range(0,len(bboxs)):
            if bboxs[i]==bboxs[j]:
                continue
            
            #check if bounding box is contained
            contains = check_contains(bboxs[i],bboxs[j])


    centroids_new = []
    bboxs_new = []
    convex_hulls_new = []
    imgs_new = []
    imgs_sizes = []
    for i in range(0,len(imgs)):
        if i in ign_ind:
            continue
        elif imgs[i].size< 100: #threshold of 1000 on segmented part sizes to get rid of extra noisy stuff
            continue
        centroids_new.append(centroids[i])
        bboxs_new.append(bboxs[i])
        convex_hulls_new.append(convex_hulls[i])
        imgs_new.append(imgs[i])
        imgs_sizes.append(imgs[i].size)
#         plt.imshow(imgs[i],cmap="gray")
#         plt.show()

    imgs_sizes = np.sort(imgs_sizes)
    return centroids_new, bboxs_new, convex_hulls_new, imgs_new

def func(x,y,w,h,shape):
    a = np.zeros(shape)
    a[x:w,y-3:y+3] = 255
    a[x:w,h-3:h+3] = 255
    a[x-3:x+3,y:h] = 255
    a[w-3:w+3,y:h] = 255
    return a.astype(np.bool)

#draw bounding box on each character in image and show
def draw_bounding_box(img,bboxs):
    im = skimage.color.gray2rgb(copy.deepcopy(img))
    for i in bboxs:
        x,y,w,h = i[0],i[1],i[2],i[3]
        a = func(x,y,w,h,im.shape[:2])
        im[a,0] = 255
        im[a,1] = 0
        im[a,2] = 0
    plt.imshow(im)
    plt.title("Image with different segmented regions")
    # plt.imsave("misc/boundingbox.png",im)
    plt.show()