from flask import Flask, request, send_file, Response, make_response
import numpy as np
import jsonpickle
import cv2
import base64
from server.detection import predict
from server.segmentation import Segmentator
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def index():
  return "hello flask"


@app.route('/api/test', methods=['POST'])
def test():
  r = request
  bytes_of_image = r.get_data()
  # convert string of image data to uint8
  nparr = np.fromstring(bytes_of_image, np.uint8)
  # with open('./server/image.jpeg', 'wb') as out:
  #   out.write(bytes_of_image)
  # return "Image read"
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  segmentator = Segmentator()
  detections = segmentator(image)
  detections.sort(key=lambda det: (det.bbox.y, det.bbox.x))
  

  image2 = image.copy()
  detected_text = ''
  for i, det in enumerate(detections):
    roi = det.img

    x, y, w, h = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
    rec, prob = predict(roi)
    detected_text += ' ' + rec
    # show ROI
    # cv2.imshow(f'{rec}, {prob}',roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.putText(image2, rec, ( x, y + h ), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.waitKey(0)




  #return send_file(img, mimetype='image/gif')
  # image2 =  cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  retval, buffer = cv2.imencode('.jpg', image2)
  data = buffer.tobytes()
  # print(type(base64.b64encode(data)))
  return jsonify({
    'image_base64': base64.b64encode(data).decode("ascii"),
    'detected_text': detected_text
  })

  # print(data)
  # return send_file('/out.jpg', mimetype='image/jpg')

  # response = make_response(data)
  # response.headers.set('Content-Type', 'image/jpeg')
  # # response.headers.set(
  # #   'Content-Disposition', 'attachment', filename='test.jpg')
  # return base64.b64encode(data)

@app.route('/api/test2', methods=['POST'])
def test2():

  img = cv2.imread('./data/pages/test1.png')

  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  # return str(img.shape)
  #return send_file(img, mimetype='image/gif')

  retval, buffer = cv2.imencode('.png', img)
  response = make_response(buffer.tobytes())
  return response

  # # build a response dict to send back to client
  # response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
  #             }
  # # encode response using jsonpickle
  # response_pickled = jsonpickle.encode(response)

  # return Response(response=resp`onse_pickled, status=200, mimetype="application/json")`