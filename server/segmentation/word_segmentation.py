import cv2
import numpy as np
from dataclasses import dataclass
import os
from constants import SAMPLE_PATH
from remove_shadows import ShadowRemover

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

class Segmentator:
  def __init__(self, vocal=False, save_samples=False, remove_shadows=True, normalize=True):
    self.vocal = vocal
    self.save_samples = save_samples
    if remove_shadows:
      self.shadow_remover = ShadowRemover(normalize=normalize, vocal=vocal, save_samples=save_samples)

  def __call__(self, image):
    """
    segment text lines from image.
    """
    original = image.copy()
    if self.shadow_remover is not None:
      image = self.shadow_remover(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if self.vocal:
      cv2.imshow('gray', gray)
      cv2.waitKey(0)
    if self.save_samples:
      cv2.imwrite(os.path.join(SAMPLE_PATH, 'gray.png'), gray)

    #binary
    blur = cv2.GaussianBlur(gray,(5, 5),0)
    if self.vocal:
      cv2.imshow('blur', blur)
      cv2.waitKey()
    if self.save_samples:
      cv2.imwrite(os.path.join(SAMPLE_PATH, 'blur.png'), blur)

    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if self.vocal:
      cv2.imshow('thresh', thresh)
      cv2.waitKey()
    if self.save_samples:
      cv2.imwrite(os.path.join(SAMPLE_PATH, 'thresh.png'), thresh)

    #dilation
    print(image.shape)
    kernel = np.ones((3,image.shape[1]), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    if self.vocal:
      cv2.imshow('dilated', img_dilation)
      cv2.waitKey()
    if self.save_samples:
      cv2.imwrite(os.path.join(SAMPLE_PATH, 'dilated.png'), img_dilation)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    detections = []
    for i, ctr in enumerate(sorted_ctrs):
      # Get bounding box
      x, y, w, h = cv2.boundingRect(ctr)

      # Getting ROI
      roi = original[y:y+h, x:x+w]

      detections.append(DetectorRes(roi, BBox(x, y, w, h)))

      if self.vocal or self.save_samples:
        cv2.rectangle(original,(x,y),( x + w, y + h ),(90,0,255),2)
    
    if self.vocal:
      cv2.imshow('segmented', original)
      cv2.waitKey()
    if self.save_samples:
      cv2.imwrite(os.path.join(SAMPLE_PATH, 'segmented.png'), original)

    return detections


if __name__ == '__main__':
  segmentator = Segmentator(vocal=True, save_samples=True, remove_shadows=True, normalize=True)
  image = cv2.imread('./data/pages/test1.png')
  segmentator(image)