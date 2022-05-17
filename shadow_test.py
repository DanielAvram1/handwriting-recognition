from cv2 import dilate
import cv2
import numpy as np

img = cv2.imread('shadow_test.jpeg')

cv2.imshow('original', img)
cv2.waitKey()

dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8)) 
cv2.imshow('dilated', dilated_img)
cv2.waitKey()
median_img = cv2.medianBlur(dilated_img, 21)

cv2.imshow('median', median_img)
cv2.waitKey()

gauss_img = cv2.GaussianBlur(dilated_img, (21, 21), cv2.BORDER_DEFAULT)

cv2.imshow('gauss', gauss_img)
cv2.waitKey()