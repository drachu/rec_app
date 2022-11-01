import cv2
import numpy as np
import tensorflow as tf


class DetectionModel():

    def __init__(self, model_dir_path="appResources/models/yolov7_model.tflite", class_names=['pedestrian'],
                 interpreter=None, input_details=None, output_details=None):
        self.model_dir_path = model_dir_path
        self.class_names = class_names
        self.interpreter = interpreter
        self.input_details = input_details,
        self.output_details = output_details
        if self.interpreter is None:
            self.interpreter, self.input_details, self.output_details = initialize_interpreter(self.model_dir_path)


def initialize_interpreter(path):
    _interp = tf.lite.Interpreter(model_path=path)
    _interp.allocate_tensors()
    _input_det = _interp.get_input_details()
    _output_det = _interp.get_output_details()
    return _interp, _input_det, _output_det


def preprocess_image(image):
    _image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LANCZOS4)
    _image_orig = _image
    _det_image = _image.transpose((2, 0, 1))
    _det_image = np.expand_dims(_det_image, 0)
    _det_image = np.ascontiguousarray(_det_image)
    _det_image = _det_image.astype(np.float32)
    _det_image /= 255
    return _det_image, _image_orig


def detection(det_image, interpreter_det, input_details_det, output_details_det):
    interpreter_det.set_tensor(input_details_det[0]['index'], det_image)
    interpreter_det.invoke()
    _output_d = interpreter_det.get_tensor(output_details_det[0]['index'])
    return _output_d


def draw_boxes(output, image_orig):
    _image_orig = image_orig
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(output):
        _score = round(float(score), 3)
        _box = [(round(x0), round(y0)), (round(x1), round(y1))]
        _name = 'pedestrian'
        _color = (219, 3, 252)
        _name += ' ' + str(_score)
        cv2.rectangle(_image_orig, _box[0], _box[1], _color, 2)
    return _image_orig

def draw_boxes_and_labels(output, image_orig):
    _image_orig = image_orig
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(output):
        _score = round(float(score), 3)
        _box = [(round(x0), round(y0)), (round(x1), round(y1))]
        _name = 'pedestrian'
        color = (219, 3, 252)
        _name += ' ' + str(_score)
        cv2.rectangle(image_orig, _box[0], _box[1], color, 2)
        cv2.putText(image_orig, _name, _box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
    return _image_orig
