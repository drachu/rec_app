import os
import logging
from CamerasProcessing import CameraProcess, SynchronizationProcess, StereoCamera, CalibrationMapping
from multiprocessing import Queue
import cv2
import pytest
LOGGER = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-5]


def test_calibration_mapping():
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera(TEST=True)
    map_rgb, map_ir = StereoCamera.get_calibration_mapping()

def test_any_camera_connection():
    stereo_camera = StereoCamera(TEST=True)
    stereo_camera.rea
    map_rgb, map_ir = stereo_camera.get_calibration_mapping('AppResources/calibration/data/stereoMap.xml')
    for i in range(10):
        camera_process_RGB = CameraProcess("RGB", i, map_rgb, TEST=True)
        camera_process_RGB.start()
    frame = stereo_camera.video_queue_RGB.get(timeout=20)
    if frame in None:
        LOGGER.error("Could not get frame from any camera!")
        assert frame is not None


def test_whole_camera_connection():
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera()
    frame = StereoCamera.synchronization_queue.get(timeout=15)
    if frame is None:
        LOGGER.error("Frame has not been received!")
        assert frame is not None

