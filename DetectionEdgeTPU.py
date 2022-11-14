import tflite_runtime.interpreter as tflite
import torch
import cv2
import numpy as np
import torchvision

class DetectionModelEdgeTPU:
    def __init__(self, model_dir_path="appResources/models/kaist_camel_own_v5-384-512-int8.tflite"):
        self.device = torch.device('cpu')
        self.initliazlie_interpreter(model_dir_path)

    def initliazlie_interpreter(self, path):
        self.interpreter = tflite.Interpreter(path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']

    def preproces_image_for_detect(self, image):
        _orig_image = cv2.resize(image, (512, 384), interpolation=cv2.INTER_LINEAR)
        det_image = _orig_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        det_image = np.ascontiguousarray(det_image)  # contiguous
        det_image = torch.from_numpy(det_image).to(self.device)
        det_image = det_image.float()
        det_image /= 255
        det_image = det_image[None]
        return det_image, _orig_image

    def detection(self, detImage):
        b, ch, h, w = detImage.shape
        detImage = detImage.permute(0, 2, 3, 1)
        detImage = detImage.cpu().numpy()
        detImage = (detImage / self.input_scale + self.input_zero_point).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], detImage)
        self.interpreter.invoke()
        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output['index'])
            x = (x.astype(np.float32) - self.output_zero_point) * self.output_scale
            y.append(x)
        y[0][..., :4] *= [w, h, w, h]
        return torch.from_numpy(y[0]).to(self.device)

    def nms(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nm=0):
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        merge = False  # use merge-NMS
        mi = 5 + nc  # mask start index

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]  # confidence
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            n = x.shape[0]
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort(descending=True)]  # sort by confidence
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            output[xi] = x[i]
        return output[0].numpy()

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def draw_boxes_and_labels(self, output_data, image):
        _image = image
        for i, det in enumerate(output_data):
            _xy_min = (round(det[0]), round(det[1]))
            _xy_max = (round(det[2]), round(det[3]))
            _score = (round(det[4], 2))
            _image = cv2.rectangle(_image, _xy_min, _xy_max, (140, 8, 189), 2)
        return _image


if __name__ == '__main__':
    detection_model = DetectionModelEdgeTPU(model_dir_path="appResources/models/kaist_camel_own_v5-384-512-int8.tflite")
    img = cv2.imread("appResources/images/test_image_00.jpg")
    img_det, orig_image = detection_model.preproces_image_for_detect(img)
    output = detection_model.detection(img_det)
    output_nms = detection_model.nms(output)
    image = detection_model.draw_boxes_and_labels(output_nms, orig_image)
    cv2.imshow('test_image', image)
    cv2.waitKey(0)