#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os
from scipy.spatial import distance

def showImage(image, window_name):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, image)

def cropImage(image):
    offset = 100
    x, y, w, h = findEdges(image)
    cropped_image = image[y - offset:(y + h) + offset, x - offset:(x + w) + offset]
    print('[Image cropped] '+str(image.shape)+' -> '+str(cropped_image.shape))
    return cropped_image

def findEdges(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    l_b = np.array([000, 100, 100])
    u_b = np.array([255, 255, 255])
    thresh = cv.inRange(hsv, l_b, u_b)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    maxsize = 0
    best = 0
    count = 0

    for cnt in contours:
        if cv.contourArea(cnt) > maxsize:
            maxsize = cv.contourArea(cnt)
            best = count
        count += 1
    x, y, w, h = cv.boundingRect(contours[best])
    print('[Edges found] '+ str(x), str(y), str(w), str(h))
    return x, y, w, h


def addScaleAndLegendToImage(image, image_legend, scale):
    x, y, w, h = findEdges(image)
    w_image, h_image = (image.shape[0]-10,image.shape[1]-100)
    pollen_diameter_in_pixel = distance.euclidean((x, y),(w, h))
    image_scale_in_micra = int((scale/87)*pollen_diameter_in_pixel)
    cv.line(image,(w_image,h_image),(w_image-image_scale_in_micra,h_image),(0,0,0),20)
    cv.putText(image, str(image_legend), (100,h_image), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    print('[Legend added] - '+image_legend)
    return image

while True:
    print('[Pollen Cropper] Starting...')
    image = "base_images/2015_07_07_5723.JPG"
    image_legend = "10"
    scale = 10
    
    print('[Pollen Cropper] Working on image '+image)
    image = cv.imread(image)
    #image = allignImage(image)
    image = cropImage(image)
    image = addScaleAndLegendToImage(image, image_legend, scale)

    showImage(image, image_legend)

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
print('[Pollen Cropper] Finished.')