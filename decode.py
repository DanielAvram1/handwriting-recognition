import numpy as np
import cv2

import requests
import io

response = requests.get('http://127.0.0.1:5000/api/test2')
response.raise_for_status()
data = np.load(io.BytesIO(response.content))  
print(data.shape)
# dtype = np.dtype('B')
# img = cv2.imread('./data/pages/test1.png')
# print(img.shape)
# try:
#   with open("test.txt", "rb") as f:
#     numpy_data = np.fromfile(f,dtype)
#   print(numpy_data.shape)
# except IOError:
#   print('Error While Opening the file!')    


# with open('test.txt', 'rb') as f:
#   image = np.asarray(bytearray(f.read()), dtype="uint8")
#   print(image.shape)
#   image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#   print(image.shape)
#   cv2.imshow('URL2Image',image)
#   cv2.waitKey()