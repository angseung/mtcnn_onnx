import time
import cv2
from mtcnn import MTCNN as MTCNN_TF
from mtcnn_cv2 import MTCNN

detector = MTCNN()
detector_tf = MTCNN_TF()
test_pic = "t.jpg"

image = cv2.cvtColor(cv2.imread(test_pic), cv2.COLOR_BGR2RGB)

start = time.time()
result = detector.detect_faces(image)
elapsed_time = time.time() - start

start = time.time()
result_ori = detector_tf.detect_faces(image)
elapsed_time_tf = time.time() - start

# Result is an array with all the bounding boxes detected. Show the first.
print(f"ONNX Version: {elapsed_time: .6f}, TF Version: {elapsed_time_tf: .6f}")
