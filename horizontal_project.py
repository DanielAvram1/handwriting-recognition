from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import sobel
import numpy as np
import cv2


img = cv2.imread('./data/pages/test2.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(img, cmap="gray")
plt.show()
def horizontal_projections(sobel_image):
  #threshold the image.
  sum_of_rows = []
  for row in range(sobel_image.shape[0]-1):
    sum_of_rows.append(np.sum(sobel_image[row,:]))
  
  return sum_of_rows

def find_peak_regions(hpp, divider=2):
  threshold = (np.max(hpp)-np.min(hpp))/divider
  peaks = []
  peaks_index = []
  for i, hppv in enumerate(hpp):
    if hppv < threshold:
      peaks.append([i, hppv])
  return peaks

sobel_image = sobel(img)

plt.imshow(sobel_image, cmap="gray")
plt.show()

hpp = horizontal_projections(sobel_image)
print(hpp)
plt.plot(hpp)
plt.show()
hpp = find_peak_regions(hpp)
print(hpp)
plt.plot(hpp)
plt.show()