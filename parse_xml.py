import cv2
import xml.etree.ElementTree as ET

image = cv2.imread('a01-000u.png')

mytree = ET.parse('a01-000u.xml')
root = mytree.getroot()
handwritten_part = root[1]

img_width = int(root.attrib['width'])
img_height = int(root.attrib['height'])

bboxes = []
texts = []
print(image.shape)
for line in handwritten_part:
  minx = img_width
  miny = img_height
  maxx = 0
  maxy = 0
  for word in line:
    if word.tag == 'word':
      for cmp in word:
        minx = min(minx, int(cmp.attrib['x']))
        miny = min(miny, int(cmp.attrib['y']))

        maxx = max(maxx, int(cmp.attrib['x']) + int(cmp.attrib['width']))
        maxy = max(maxy, int(cmp.attrib['y']) + int(cmp.attrib['height']))

  bbox = {
    'x': minx,
    'y': miny,
    'h': maxy - miny,
    'w': maxx - minx
  }

  text = line.attrib['text']

  bboxes.append(bbox)
  texts.append(text)

for i in range(len(bboxes)):
  bbox = bboxes[i]
  text = texts[i]
  x = bbox['x']
  y = bbox['y']
  w = bbox['w']
  h = bbox['h']  
  print(x, y, w, h)

  cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
  cv2.putText(image, text, ( x, y + h ), cv2.FONT_HERSHEY_SIMPLEX, 
                  1, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('rectangles', image)
cv2.waitKey()

