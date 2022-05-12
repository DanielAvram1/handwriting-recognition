
import cv2
import numpy as np
from constants import SAMPLE_PATH
import os

class ShadowRemover:
  def __init__(self, vocal=False, save_samples=False, normalize=True):
    self.vocal = vocal
    self.save_samples = save_samples
    self.normalize = normalize

  def __call__(self, img):
    rgb_planes = cv2.split(img)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        if self.normalize:
          diff_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        result_planes.append(diff_img)


    result = cv2.merge(result_planes)
    
    if self.vocal:
      cv2.imshow('original', img)
      cv2.waitKey()
      cv2.imshow('withou shadows', result)
      cv2.waitKey()
    
    if self.save_samples:
      cv2.imwrite(os.path.join(SAMPLE_PATH, 'shadowless.png'), result)

    return result

if __name__ == '__main__':
  img = cv2.imread('./data/pages/test2.jpeg', -1)
  shadow_remover = ShadowRemover(vocal=True, save_samples=True)
  shadow_remover(img)
  