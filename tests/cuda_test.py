from DetectionCUDA import DetectionModelCUDA
import cv2
import glob
import datetime
import logging
import os
import pytest
LOGGER = logging.getLogger(__name__)

# pytest cuda_test.py::test_dataset_speed to run one test
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-5]
model_path_pytorch = "AppResources/models/yv5/yv5s_kco.pt"
dataset_test_path = "datasets/PedestrianPGETIs179985/test/images"
dataset_images_to_test = 5  # -1 will make test go through whole dataset
image_path = "tests/test_images/test_image_00.jpg"


@pytest.mark.unit
def test_load_model():
    os.chdir(ROOT_DIR)
    detection_model = DetectionModelCUDA(model_dir_path=model_path_pytorch)
    assert detection_model.model is not None
    print("\nSuccessfuly loaded model on device: " + str(detection_model.device))

def test_dataset_speed():
    os.chdir(ROOT_DIR)
    results = []
    detection_model = DetectionModelCUDA(model_dir_path=model_path_pytorch)
    print("\n")
    for i, image_path in enumerate(glob.glob(dataset_test_path + "/*jpg")):
        img = cv2.imread(image_path)
        detection_timer_start = datetime.datetime.now()
        output = detection_model.model(img)
        detection_timer_stop = datetime.datetime.now()
        results.append((detection_timer_stop - detection_timer_start).microseconds / 1000)
        print(str(results[i]) + " ms | image: " + image_path)
        if i >= dataset_images_to_test and i != -1: break
    print("Average detection time " + str(round(sum(results) / len(results), 2)) + " ms")
    print("Used model: " + model_path_pytorch)
    print("Used dataset: " + dataset_test_path)

def test_single_speed():
    os.chdir(ROOT_DIR)
    detection_model = DetectionModelCUDA(model_dir_path=model_path_pytorch)
    test_image = cv2.imread(image_path)
    timer_start = datetime.datetime.now()
    results = detection_model.model(test_image)
    test_image = detection_model.draw_detections(results, test_image, labels=True)
    timer_end = datetime.datetime.now()
    print("\nDetection time: " + str(round((timer_end - timer_start).microseconds / 1000, 2)) + " ms")
    print("\nLoaded model: " + model_path_pytorch + " on device: " + str(detection_model.device))
    # cv2.imshow('test_image', test_image)
    # cv2.waitKey(0)


