import cv2

import TensorflowLiteDetection

# cam = cv2.VideoCapture(2)
# ret, frame = cam.read()
#
from TensorflowLiteDetection import *
model = TensorflowLiteDetection.DetectionModel()
# while ret:
#     ref, frame = cam.read()
#     frame = cv2.resize(frame, (640, 488), interpolation=cv2.INTER_LANCZOS4)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.bitwise_not(frame)
#     frame = cv2.bilateralFilter(frame, 30, 15, 15)
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     frame_det, frame_orig = TensorflowLiteDetection.preprocess_image(frame)
#     output_data = TensorflowLiteDetection.detection(frame_det, model.interpreter, model.input_details, model.output_details)
#     frame_orig = TensorflowLiteDetection.draw_boxes_and_labels(output_data, image_orig=frame_orig)
#
#     # frame = cv2.addWeighted(frame, 0.5, result, 0.5, 0.0)
#     cv2.imshow('frame', frame_orig)
#     cv2.waitKey(20)

frame = cv2.imread('records/dual3/000011.jpg')
frame = cv2.resize(frame, (640, 488), interpolation=cv2.INTER_LANCZOS4)
frame_det, frame_orig = TensorflowLiteDetection.preprocess_image(frame)
output_data = TensorflowLiteDetection.detection(frame_det, model.interpreter, model.input_details, model.output_details)
frame_orig = TensorflowLiteDetection.draw_boxes_and_labels(output_data, image_orig=frame_orig)

# frame = cv2.addWeighted(frame, 0.5, result, 0.5, 0.0)
cv2.imshow('frame', frame_orig)
cv2.waitKey(0)
