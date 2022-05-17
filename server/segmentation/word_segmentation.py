import cv2
import numpy as np
from dataclasses import dataclass
import os
from .constants import SAMPLE_PATH
from .remove_shadows import ShadowRemover
import matplotlib.pyplot as plt

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
  def __init__(self, vocal=False, save_samples=False, remove_shadows=True, normalize=True, hpp_dilation=True):
    self.vocal = vocal
    self.hpp_dilation = hpp_dilation
    self.hpp_divider = 6
    self.save_samples = save_samples
    if remove_shadows:
      self.shadow_remover = ShadowRemover(normalize=normalize, vocal=vocal, save_samples=save_samples)

  def horizontal_projections(self, thresh):
    #threshold the image.
    sum_of_rows = []
    for row in range(thresh.shape[0]-1):
      sum_of_rows.append(np.sum(thresh[row,:]))
    
    return sum_of_rows

  # hpp = horizontal_projections(thresh)
  # print(img.shape, len(hpp))
  # # plt.plot(hpp)
  # # plt.show()

  def find_peak_regions(self, hpp):
    threshold = (np.max(hpp)-np.min(hpp))/self.hpp_divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
      if hppv > threshold:
        peaks.append([i, hppv])
    return peaks

  def __call__(self, image):
    """
    segment text lines from image.
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if self.vocal:
      cv2.imshow('gray', gray)
      cv2.waitKey(0)
    if self.save_samples:
      cv2.imwrite(os.path.join(SAMPLE_PATH, 'gray.png'), gray)

    if self.shadow_remover is not None:
      gray = self.shadow_remover(gray)
    

    #binary
    blur_horizontal = image.shape[0] // 20
    if blur_horizontal % 2 == 0:
      blur_horizontal -= 1
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
    kernel = np.ones((1, 5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    if self.hpp_dilation:

      hpp = self.horizontal_projections(thresh)
      peaks = self.find_peak_regions(hpp)

      for peak in peaks:
        idx = peak[0]
        img_dilation[idx] = 255
      
      if self.vocal:
        cv2.imshow('dilated', img_dilation)
        cv2.waitKey()
        divider_line = np.empty(len(hpp))
        divider_line = []
        hpp_threshold = (np.max(hpp)-np.min(hpp))/self.hpp_divider
        for i in range(len(hpp)):
          divider_line.append(hpp_threshold)

        plt.gca()
        plt.plot(hpp, 'b', divider_line , 'y')
        plt.show()
        plt.savefig(os.path.join(SAMPLE_PATH, 'hpp_plot.png'))
      if self.save_samples:
        cv2.imwrite(os.path.join(SAMPLE_PATH, 'dilated.png'), img_dilation)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    detections = []
    min_w = image.shape[0] // 50
    for i, ctr in enumerate(sorted_ctrs):
      # Get bounding box
      x, y, w, h = cv2.boundingRect(ctr)

      if w  > min_w:
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
  segmentator = Segmentator(vocal=True, remove_shadows=True, normalize=True)
  image = cv2.imread('./data/pages/test1.png')
  segmentator(image)