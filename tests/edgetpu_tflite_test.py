import os

import cv2
import torch
from DetectionEdgeTPU import DetectionModelEdgeTPU
import glob
import datetime
import logging
LOGGER = logging.getLogger(__name__)

# pytest edgetpu_tflite_test.py::test_dataset_speed to run one test
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-5]
model_path_edgetpu = "AppResources/models/yv5/yv5n_ko_uint8_384_512_edgetpu.tflite"
dataset_test_path = "datasets/PedestrianPGETIs179985/test/images"
dataset_images_to_test = 1  # -1 will make test go through whole dataset
image_path = "tests/test_images/test_image_03.jpg"
DetectionModelEdgeTPU.TEST_TFLite = True

def test_dataset_speed():
    """
    Testing prediction speed on dataset passed with path.
    """
    os.chdir(ROOT_DIR)
    results = []
    detection_model = DetectionModelEdgeTPU(model_dir_path=model_path_edgetpu)
    print("\n")
    for i, image_path in enumerate(glob.glob(dataset_test_path + "/*jpg")):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_LINEAR)
        img_det, orig_image = detection_model.preproces_image_for_detect(img)
        detection_timer_start = datetime.datetime.now()
        output = detection_model.detection(img_det)
        detection_timer_stop = datetime.datetime.now()
        results.append((detection_timer_stop - detection_timer_start).microseconds / 1000)
        print(str(results[i]) + " ms | image: " + image_path)
        if i >= dataset_images_to_test: break
    print("Average detection time " + str(sum(results) / len(results)) + " ms")
    print("Used model: " + model_path_edgetpu)
    print("Used dataset: " + dataset_test_path)


def test_load_model():
    """
    Testing possibility of loading TFLite/Edge TPU model from path.
    """
    os.chdir(ROOT_DIR)
    detection_model = DetectionModelEdgeTPU(model_dir_path=model_path_edgetpu)
    input = detection_model.input_details[0]['shape']
    assert input[0] == 1 and input[1] == 384 and input[2] == 512 and input[3] == 3
    LOGGER.info("Valid input: %s", input)
    output = detection_model.output_details[0]['shape']
    assert output[0] == 1 and output[1] == 12096 and output[2] == 6
    LOGGER.info("Valid input: %s", output)

def test_preprocess():
    """
    Testing dimensions and type of preprocessed frame for TFLite/Edge TPU predcitions.
    """
    os.chdir(ROOT_DIR)
    detection_model = DetectionModelEdgeTPU(model_dir_path=model_path_edgetpu)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_LINEAR)
    LOGGER.info("\nImage before\nShape: %s\nType: %s", img.shape, type(img))

    preprocess_timer_start = datetime.datetime.now()
    img_det, orig_image = detection_model.preproces_image_for_detect(img)
    preprocess_timer_stop = datetime.datetime.now()
    assert type(img_det) == torch.Tensor
    assert img_det.shape == torch.Size([1, 3, 384, 512])
    LOGGER.info("\nImage after\nShape: %s\nType: %s", img_det.shape, type(img_det))
    print("\nPreprocess time: " + str((preprocess_timer_stop - preprocess_timer_start).microseconds / 1000) + " ms")

def test_nms():
    """
    Testing speed on Non-maximum suppression algorithm.
    """
    os.chdir(ROOT_DIR)
    detection_model = DetectionModelEdgeTPU(model_dir_path=model_path_edgetpu)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_LINEAR)
    img_det, orig_image = detection_model.preproces_image_for_detect(img)
    output = detection_model.detection(img_det)
    LOGGER.info("\nOutput before\nShape: %s\nType: %s", output.shape, type(output))

    nms_timer_start = datetime.datetime.now()
    output_nms = detection_model.nms(output)
    nms_timer_stop = datetime.datetime.now()
    LOGGER.info("\nOutput after\nShape: %s\nType: %s", output_nms.shape, type(output_nms))
    print("\nNMS time: " + str((nms_timer_stop - nms_timer_start).microseconds / 1000) + " ms")


def test_single_speed():
    """
    Testing prediction speed on single frame.
    """
    os.chdir(ROOT_DIR)
    detection_model = DetectionModelEdgeTPU(model_dir_path=model_path_edgetpu)
    assert detection_model is not None
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_LINEAR)

    preprocess_timer_start = datetime.datetime.now()
    img_det, orig_image = detection_model.preproces_image_for_detect(img)
    preprocess_timer_stop = datetime.datetime.now()

    detection_timer_start = datetime.datetime.now()
    output = detection_model.detection(img_det)
    assert output.shape == torch.Size([1, 12096, 6])
    detection_timer_stop = datetime.datetime.now()

    nms_timer_start = datetime.datetime.now()
    output_nms = detection_model.nms(output)
    image = detection_model.draw_detections(output_nms, orig_image, labels=True)
    nms_timer_stop = datetime.datetime.now()
    print("\nPreprocess time: " + str((preprocess_timer_stop - preprocess_timer_start).microseconds / 1000) + " ms")
    print("Detection time: " + str((detection_timer_stop - detection_timer_start).microseconds / 1000) + " ms")
    print("NMS and box drawing time: " + str((nms_timer_stop - nms_timer_start).microseconds / 1000) + " ms")
    cv2.imshow('test_image', image)
    cv2.waitKey(0)
