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
    return x, y, w, h


def rotateImage(image):
    x, y, w, h = findEdges(image)
    # print(x, y, w, h)
    center_edge = (w / 2, h / 2)
    degrees = (findAngle(center_edge, (center_edge[0], h), (w, h))) * 100
    degrees = 0  # 90 - degrees
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, degrees, 1.0)
    rotated_image = cv.warpAffine(image, M, (w, h))
    return rotated_image


def findAngle(center, x, y):
    b = np.array(center)
    a = np.array(x)
    c = np.array(y)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def addScaleLegendToImage(image, image_legend):
    x, y, w, h = findEdges(image)
    w_image, h_image = (image.shape[0]-10,image.shape[1]-100)
    pollen_diameter_in_pixel = distance.euclidean((x, y),(w, h))
    image_scale_in_micra = int((scale/87)*pollen_diameter_in_pixel)
    cv.line(image,(w_image,h_image),(w_image-image_scale_in_micra,h_image),(0,0,0),20)
    cv.putText(image, str(image_legend), (100,h_image), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    return image


while True:

    image = "base_images/2015_07_07_5723.JPG"
    image_legend = "10"
    scale = 10
    
    image = cv.imread(image)
    #image = rotateImage(image)
    image = cropImage(image)
    image = addScaleLegendToImage(image, image_legend)

    showImage(image, "Cropped")

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()