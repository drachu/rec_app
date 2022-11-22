import os
import logging
import platform
import time

from CamerasProcessing import CameraProcess, SynchronizationProcess, StereoCamera, CalibrationMapping
from multiprocessing import Queue
import cv2
import pytest
LOGGER = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-5]
RGB_CAMERA_ID = None
IR_CAMERA_ID = None

def test_calibration_mapping():
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera(TEST=True)
    map_rgb, map_ir = stereo_camera.get_calibration_mapping('AppResources/calibration/data/stereoMap.xml')


def test_any_camera_connection():
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera(TEST=True)
    stereo_camera.camera_switch.value = False
    map_rgb, map_ir = stereo_camera.get_calibration_mapping('AppResources/calibration/data/stereoMap.xml')
    processes = []
    for i in range(10):
        camera_process_RGB = CameraProcess("RGB", i, map_rgb, TEST=True)
        processes.append(camera_process_RGB)
        camera_process_RGB.start()
    try:
        frame = stereo_camera.video_queue_RGB.get(timeout=20)
    except:
        raise Exception('Could not get frame from IR camera!')
    if frame is None:
        LOGGER.error("Could not get frame from IR camera!")
        assert frame is not None
    kill_processes(processes)
    time.sleep(4)

def test_rgb_camera_connection():
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera(TEST=True)
    stereo_camera.camera_switch.value = False
    map_rgb, map_ir = stereo_camera.get_calibration_mapping('AppResources/calibration/data/stereoMap.xml')
    if RGB_CAMERA_ID is None:
        id_RGB = 1 if platform.system() == "Windows" else 2
        camera_process_RGB = CameraProcess("RGB", id_RGB, map_rgb, TEST=True)

    else:
        camera_process_RGB = CameraProcess("RGB", RGB_CAMERA_ID, map_rgb, TEST=True)
    camera_process_RGB.start()
    try:
        frame = stereo_camera.video_queue_RGB.get(timeout=20)
    except:
        raise Exception('Could not get frame from IR camera!')
    if frame is None:
        LOGGER.error("Could not get frame from IR camera!")
        assert frame is not None
    camera_process_RGB.kill()
    time.sleep(4)

def test_ir_camera_connection():
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera(TEST=True)
    map_rgb, map_ir = stereo_camera.get_calibration_mapping('AppResources/calibration/data/stereoMap.xml')
    if IR_CAMERA_ID is None:
        id_IR = 2 if platform.system() == "Windows" else 1
        camera_process_IR = CameraProcess("IR", id_IR, map_ir, TEST=True)

    else:
        camera_process_IR = CameraProcess("IR", IR_CAMERA_ID, map_ir, TEST=True)
    camera_process_IR.start()
    try:
        frame = stereo_camera.video_queue_IR.get(timeout=20)
    except:
        raise Exception('Could not get frame from IR camera!')
    if frame is None:
        LOGGER.error("Could not get frame from any camera!")
        assert frame is not None
    camera_process_IR.kill()
    time.sleep(4)

def test_whole_camera_connection():
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera()
    frame = StereoCamera.synchronization_queue.get(timeout=15)
    if frame is None:
        LOGGER.error("Frame has not been received!")
        assert frame is not None

def kill_processes(proc):
    for p in proc:
        p.kill()
        time.sleep(3)
