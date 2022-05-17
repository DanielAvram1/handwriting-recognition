import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('./data/pages/test5.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('gray', img)
cv2.waitKey()
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# cv2.imshow('thresh', thresh)
# cv2.waitKey()

def horizontal_projections(thresh):
  #threshold the image.
  sum_of_rows = []
  for row in range(thresh.shape[0]-1):
    sum_of_rows.append(np.sum(thresh[row,:]))
  
  return sum_of_rows

hpp = horizontal_projections(thresh)
print(img.shape, len(hpp))
# plt.plot(hpp)
# plt.show()

def find_peak_regions(hpp, divider=2):
  threshold = (np.max(hpp)-np.min(hpp))/divider
  peaks = []
  peaks_index = []
  for i, hppv in enumerate(hpp):
    if hppv < threshold:
      peaks.append([i, hppv])
  return peaks

segmented = img.copy()
peaks = find_peak_regions(hpp)
print(len(peaks))
for peak in peaks:
  idx = peak[0]
  segmented[idx] = 0

cv2.imshow('segmented', segmented)
cv2.waitKey()