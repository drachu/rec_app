import os

import cv2
import torch
import datetime
class DetectionModelCUDA:
    def __init__(self, model_dir_path='AppResources/models/yv5/yv5n_ko.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir_path = model_dir_path
        self.model = torch.hub.load("yolov5", 'custom', path=model_dir_path, source='local')

    def draw_detections(self, results, image, labels=False):
        _results = results.pandas().xyxy[0].to_dict(orient="records")
        for _result in _results:
            _con = (round(_result['confidence'], 2))
            _xy_min = (int(_result['xmin']), int(_result['ymin']))
            _xy_max = (int(_result['xmax']), int(_result['ymax']))
            _image = cv2.rectangle(image, _xy_min, _xy_max, (140, 8, 189), 2)
            if labels and int(_result['xmax']) - int(_result['xmin']) > 40:
                _xy_top_right = (int(_result['xmax']), int(_result['ymin']))
                _image = cv2.rectangle(_image, (_xy_min[0], _xy_min[1] - 20), _xy_top_right, (140, 8, 189), -1)
                _image = cv2.putText(_image, str(_con), (_xy_min[0] + 5, _xy_min[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                     (255, 255, 255), 1, cv2.LINE_AA)
        return image
