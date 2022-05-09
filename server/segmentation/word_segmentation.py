import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class DetectorRes:
    img: np.ndarray
    bbox: BBox

def segment(image):

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # cv2.imshow('gray',gray)
  # cv2.waitKey(0)

  #binary
  blur = cv2.GaussianBlur(gray,(5,5),0)
  ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  # ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
  # cv2.imshow('second',thresh)
  # cv2.waitKey(0)

  #dilation
  print(image.shape)
  kernel = np.ones((3,image.shape[1]), np.uint8)
  img_dilation = cv2.dilate(thresh, kernel, iterations=1)
  # cv2.imshow('dilated',img_dilation) 
  # cv2.waitKey(0)

  #find contours
  ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #sort contours
  sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

  detections = []
  for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    detections.append(DetectorRes(roi, BBox(x, y, w, h)))

    # rec, prob = predict(roi)

    # # show ROI
    # # cv2.imshow(f'{rec}, {prob}',roi)
    # cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    # cv2.putText(image2, rec, ( x, y + h ), cv2.FONT_HERSHEY_SIMPLEX, 
    #               1, (0, 0, 0), 2, cv2.LINE_AA)
    # # cv2.waitKey(0)
  return detections