import cv2
import torch
import pandas
class DetectionModelCUDA():
    def __init__(self, model_dir_path='appResources/models/kaist_camel_own_v5.pt', class_names=['pedestrian']):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir_path = model_dir_path
        self.class_names = class_names
        self.model = torch.hub.load("yolov5", 'custom', path=model_dir_path, source='local')

def draw_detections(results, image):
    _results = results.pandas().xyxy[0].to_dict(orient="records")
    for _result in _results:
        _con = _result['confidence']
        _cs = _result['class']
        _xy_min = (int(_result['xmin']), int(_result['ymin']))
        _xy_max = (int(_result['xmax']), int(_result['ymax']))
        cv2.rectangle(image, _xy_min, _xy_max, (140, 8, 189), 2)
    return image

if __name__ == '__main__':
    detection_model = DetectionModelCUDA()
    test_image = cv2.imread("appResources/images/test_image_00.jpg")
    results = detection_model.model(test_image)
    test_image = draw_detections(results, test_image)
    cv2.imshow('test_image', test_image)
    cv2.waitKey(0)