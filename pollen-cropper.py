#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os


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
    # cv.drawContours(image, contours, best, (0, 255, 0), 3)
    # cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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


while True:

    image_original = cv.imread("/home/jean/Documentos/DevOps/pollen-cropper/base_images/2015_07_07_5723.JPG")
    rotated_image = rotateImage(image_original)
    cropped_image = cropImage(rotated_image)

    cv.putText(cropped_image, "Pollen/Spore", (20, 1770), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4, cv.LINE_AA)

    showImage(image_original, "Image")
    showImage(cropped_image, "Cropped")

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
