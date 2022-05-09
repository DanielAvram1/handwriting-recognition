import cv2 as cv
import matplotlib.pyplot as plt
from word_segmentation import WordSegmentator
# from path import Path

test_img = cv.imread('./data/pages/test3.jpeg')
print(test_img.shape)

# fig = plt.figure(figsize=(10, 10))
word_segmentator = WordSegmentator(preserve_height=True)

detections = word_segmentator(img=test_img, kernel_size=51, sigma=11, theta=1, min_area=10)

# print(len(detections))
lines = word_segmentator.sort_multiline(detections)
# print(lines)
# fig.add_subplot(1, 1, 1)
plt.imshow(word_segmentator._prepare_img(test_img, word_segmentator.height), cmap='gray')

num_colors = 7
rows = len(lines)
columns = 0
colors = plt.cm.get_cmap('rainbow', num_colors)
for line_idx, line in enumerate(lines):
  columns = max(columns, len(line))
  for word_idx, det in enumerate(line):
    xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
    ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
    plt.plot(xs, ys, c=colors(line_idx % num_colors))
    plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

# for line_idx, line in enumerate(lines):
#   # if line_idx > 2:
#   #   break
#   for word_idx, det in enumerate(line):
#     fig.add_subplot(rows, columns, line_idx + word_idx + 1)
#     plt.imshow(det.img)
#     plt.axis('off')
#     plt.title(f'{line_idx}/{word_idx}')

plt.show()
plt.imshow(test_img, cmap='gray')
plt.show()