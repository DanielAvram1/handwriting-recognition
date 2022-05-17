import cv2
from path import Path

from .dataloader_iam import Batch
from .model import Model
from .preprocessor import Preprocessor
import os
dirname = os.path.dirname(__file__)

class FilePaths:
  """Filenames and paths to data."""
  fn_char_list = os.path.join(dirname, './model/charList.txt')
  fn_summary = os.path.join(dirname, './model/summary.json')
  # fn_corpus = './data/corpus.txt'


def char_list_from_file():# -> List[str]:
  with open(FilePaths.fn_char_list) as f:
    return list(f.read())

def get_img_height():# -> int:
  """Fixed height for NN."""
  return 32

def get_img_size(line_mode: bool = False):# -> Tuple[int, int]:
  """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
  if line_mode:
    return 256, get_img_height()
  return 128, get_img_height()


def infer(model: Model, fn_img: Path):# -> None:
  """Recognizes text in image provided by file path."""
  img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
  # blur = cv2.GaussianBlur(img,(5,5),0)
  # img,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  assert img is not None

  preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
  img = preprocessor.process_img(img)

  batch = Batch([img], None, 1)
  recognized, probability = model.infer_batch(batch, True)
  print(f'Recognized: "{recognized[0]}"')
  print(f'Probability: {probability[0]}')
  return recognized[0], probability[0]


model = Model(char_list_from_file(), must_restore=True, dump=False)

def predict(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  assert img is not None

  preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
  img = preprocessor.process_img(img)

  batch = Batch([img], None, 1)
  recognized, probability = model.infer_batch(batch, True)
  print(recognized)
  print(f'Recognized: "{recognized[0]}"')
  if probability:
    print(f'Probability: {probability[0]}')
    return recognized[0], probability[0]
  else:
    return recognized[0], -1


if __name__ == '__main__':
  model = Model(char_list_from_file(), must_restore=True, dump=False)
  infer(model, './data/lines/7.png')  