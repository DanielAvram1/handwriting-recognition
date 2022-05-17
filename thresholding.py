import cv2

img = cv2.imread('data/pages/test_shadow.png', cv2.IMREAD_GRAYSCALE)

val = 100

img[img > val] = img[img > val] - val


ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('thresh otsu', th2)
cv2.waitKey()

cv2.imwrite('doc-images/tresh_shadow.png', th2)
