import cv2
import numpy as np
from remove_shadow import remove_shadows
from server.detection import predict
from server.segmentation import segment
#import image
image = cv2.imread('./data/pages/test6.jpeg')
image = remove_shadows(image)
cv2.imshow('orig',image)
cv2.waitKey(0)
detections = segment(image)
image2 = image.copy()
for i, det in enumerate(detections):
  roi = det.img

  x, y, w, h = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
  rec, prob = predict(roi)

  # show ROI
  cv2.imshow(f'{rec}, {prob}',roi)
  cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
  cv2.putText(image2, rec, ( x, y + h ), cv2.FONT_HERSHEY_SIMPLEX, 
                  1, (0, 0, 0), 2, cv2.LINE_AA)
  cv2.waitKey(0)

cv2.imshow('marked areas',image)
cv2.waitKey(0)
cv2.imshow('detected', image2)
cv2.waitKey(0)