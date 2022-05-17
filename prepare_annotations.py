import os 
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join

forms_path = '/Users/daniel/Documents/Anul3/licenta/text-lines/formsA-D'
xml_path = '/Users/daniel/Documents/Anul3/licenta/text-lines/xml'

filenames = [f for f in listdir(forms_path) if isfile(join(forms_path, f))]

def extract_bboxes(filename):

  mytree = ET.parse(filename)
  root = mytree.getroot()
  handwritten_part = root[1]

  bboxes = []
  texts = []

  img_width = int(root.attrib['width'])
  img_height = int(root.attrib['height'])

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
  norm_bboxes = []
  for i in range(len(bboxes)):
    bbox = bboxes[i]
    text = texts[i]
    x = bbox['x']
    y = bbox['y']
    w = bbox['w']
    h = bbox['h']
    centerx = x + w // 2
    centery = y + h // 2
    norm = (centerx / img_width, centery / img_height, w / img_width, h / img_height) 
    norm_bboxes.append(norm)
    # print(x, y, w, h)
    # print(norm)
  return norm_bboxes

if __name__ == '__main__':
  forms_path = '/Users/daniel/Documents/Anul3/licenta/text-lines/forms'
  xml_path = '/Users/daniel/Documents/Anul3/licenta/text-lines/xml'

  filenames = [f for f in listdir(forms_path) if isfile(join(forms_path, f))]
  print(len(filenames))
  for filename in filenames:
    true_name = filename.split('.')[0]
    xml_filename = os.path.join(xml_path, true_name + '.xml')
    bboxes = extract_bboxes(xml_filename)
    txt_filename = os.path.join(forms_path, true_name + '.txt')

    with open(txt_filename, 'a') as f:
      for bbox in bboxes:
        f.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')

  filenames = [f for f in listdir(forms_path) if isfile(join(forms_path, f))]
  print(len(filenames))
