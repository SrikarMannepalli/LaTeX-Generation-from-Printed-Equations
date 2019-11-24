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
from matplotlib.path import Path
from binarization import *
from character import *
from skewcorrection import *
from segmentation import *
from assemble import *


with open('../chars_list.txt','r') as f:
    data = f.read().replace('\n',',')
    chars_list = list(data.split(","))[1::2]

id_profs_list = np.load("./id_profs_list.npy")

img = cv2.imread("../input/Equations/Clean/eq8_hr.jpg",0)
plt.imshow(img,cmap='gray')
plt.show()
plt.imsave("misc/input_image.png",img,cmap='gray')
new_img = binarize_input(img)
plt.imshow(new_img,cmap='gray')
plt.show()
plt.imsave("misc/binarized_image.png",new_img,cmap='gray')
if find_scan_screenshot(img)==1:
    img_rotated = skew_correction(new_img)
else :
    #img_rotated = new_img
    img_rotated = skew_correction_scanned(new_img)
# img_rotated = new_img
img_rotated[img_rotated==1] = 255
plt.imshow(img_rotated,cmap='gray')
plt.show()
plt.imsave("misc/skew_corrected_image.png",img_rotated,cmap='gray')

centroids, bboxs, convex_hulls,imgs = segmentation(img_rotated)
draw_bounding_box(img_rotated,bboxs)
#for i in range(0,len(imgs)):
#     str_ele2 = skimage.morphology.selem.disk(2)
#     str_ele = skimage.morphology.selem.square(3)
#     imgs[i] = skimage.morphology.binary_erosion(imgs[i],selem=str_ele2)
#     imgs[i] = skimage.morphology.binary_dilation(imgs[i],selem=str_ele)
    #plt.imshow(imgs[i],cmap='gray')


centroids_new = create_new_centroids(centroids, bboxs, imgs)
detected_chars = []
if find_scan_screenshot(img)==1:
    #for photograph
    for i in range(0,len(imgs)):
    #     str_ele = skimage.morphology.selem.square(2)
    #     after_and = skimage.morphology.binary_dilation(imgs[i],selem=str_ele)
        after_and = copy.deepcopy(imgs[i])
        str_ele = skimage.morphology.selem.disk(2)
        after_and = skimage.morphology.binary_erosion(after_and,selem=str_ele)
        new_and = copy.deepcopy(after_and)
    #     print(np.max(new_and))
    #     plt.show()
        #print(np.sum(new_and),new_and.size)
        if np.sum(new_and)<0.15*new_and.size:
            #print("hi")
            min_ind = find_nn_srikar(imgs[i],centroids_new[i],id_profs_list)
        elif new_and.shape[0]<20:
            min_ind = 100
        else:
            min_ind = find_nn_srikar(after_and,centroids_new[i],id_profs_list)
        detected_chars.append(chars_list[min_ind])
else : 
    for i in range(0,len(imgs)):
        str_ele = skimage.morphology.selem.square(3)
#       after_and = skimage.morphology.binary_dilation(imgs[i],selem=str_ele)
        after_and = imgs[i]
#       str_ele = skimage.morphology.selem.square(3)
#       after_and = skimage.morphology.binary_erosion(after_and,selem=str_ele)
        min_ind = find_nn_srikar(after_and,centroids_new[i],id_profs_list)
        detected_chars.append(chars_list[min_ind])

print(detected_chars)

eq = assemble_eqn(get_boxes(bboxs),get_centroids(centroids), detected_chars)
print(eq)
a = eq
plt.plot()
plt.text(0.5,0.5,'$%s$'%a)
plt.show()