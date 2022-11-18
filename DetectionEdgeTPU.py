import platform
import tflite_runtime.interpreter as tflite
import torch
import cv2
import numpy as np
import torchvision
import datetime

class DetectionModelEdgeTPU:
    TEST_TFLite = False

    def __init__(self, model_dir_path="appResources/models/yv5/yv5s_kco_uint8_384_512_edgetpu.tflite"):
        self.device = torch.device('cpu')
        self.initliazlie_interpreter(model_dir_path)

    def initliazlie_interpreter(self, path):
        self.delegate = {
            'Linux': 'libedgetpu.so.1',
            'Darwin': 'libedgetpu.1.dylib',
            'Windows': 'edgetpu.dll'}[platform.system()]
        if DetectionModelEdgeTPU.TEST_TFLite:
            self.interpreter = tflite.Interpreter("appResources/models/yv5/yv5s_kco_uint8_384_512.tflite")
        else:
            self.interpreter = tflite.Interpreter(path, experimental_delegates=[tflite.load_delegate(self.delegate)])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']

    def preproces_image_for_detect(self, frame):
        _orig_image = cv2.resize(frame, (512, 384), interpolation=cv2.INTER_LINEAR)
        _det_image = _orig_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        _det_image = np.ascontiguousarray(_det_image)  # contiguous
        _det_image = torch.from_numpy(_det_image).to(self.device)
        _det_image = _det_image.float()
        _det_image /= 255
        _det_image = _det_image[None]
        return _det_image, _orig_image

    def detection(self, frame):
        _batch, _channel, _height, _width = frame.shape
        _det_image = frame.permute(0, 2, 3, 1)
        _det_image = _det_image.cpu().numpy()
        _det_image = (_det_image / self.input_scale + self.input_zero_point).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], _det_image)
        self.interpreter.invoke()
        result = ((self.interpreter.get_tensor(self.output_details[0]['index'])).astype(np.float32) - self.output_zero_point) * self.output_scale
        result[..., :4] *= [_width, _height, _width, _height]
        return torch.from_numpy(result).to(self.device)

    def nms(self, predictions, conf_thres=0.45, iou_thres=0.30, max_det=300, nm=0):
        class_count = predictions.shape[2] - nm - 5  # number of classes
        prediction_candidates = predictions[..., 4] > conf_thres  # candidates

        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        mi = 5 + class_count  # mask start index

        _prediction = predictions[0][prediction_candidates[0]]  # confidence
        _prediction[:, 5:] *= _prediction[:, 4:5]  # confidence = obj_confidence * cls_confidence
        box = self.xywh2xyxy(_prediction[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = _prediction[:, mi:]  # zero columns if no masks
        conf, j = _prediction[:, 5:mi].max(1, keepdim=True)
        _prediction = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        n = _prediction.shape[0]
        if not n:  # no boxes
            return []
        elif n > max_nms:  # excess boxes
            _prediction = _prediction[_prediction[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            _prediction = _prediction[_prediction[:, 4].argsort(descending=True)]  # sort by confidence
        c = _prediction[:, 5:6] * max_wh  # classes
        boxes, scores = _prediction[:, :4] + c, _prediction[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        return _prediction[i].numpy()

    def xywh2xyxy(self, x):
        _box = x.clone()
        _box[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        _box[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        _box[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        _box[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return _box

    def draw_boxes_and_labels(self, output_data, frame):
        _image = frame
        for i, det in enumerate(output_data):
            _xy_min = (round(det[0]), round(det[1]))
            _xy_max = (round(det[2]), round(det[3]))
            _score = (round(det[4], 2))
            _image = cv2.rectangle(_image, _xy_min, _xy_max, (140, 8, 189), 2)
            _image = cv2.putText(_image, str(_score), (_xy_min[0], _xy_min[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 8, 189), 1, cv2.LINE_AA)
        return _image


if __name__ == '__main__':
    detection_model = DetectionModelEdgeTPU(model_dir_path="appResources/models/yv5/yv5s_kco_uint8_384_512_edgetpu.tflite")
    img = cv2.imread("appResources/images/test_image_00.jpg")

    img_det, orig_image = detection_model.preproces_image_for_detect(img)
    output = detection_model.detection(img_det)
    timer_start = datetime.datetime.now()
    
    output_nms = detection_model.nms(output)
    image = detection_model.draw_boxes_and_labels(output_nms, orig_image)

    timer_end = datetime.datetime.now()
    print("Detection time: " + str((timer_end-timer_start).microseconds/1000) + " ms")
    cv2.imshow('test_image', image)
    cv2.waitKey(0)