import os
import logging
import platform
import time

from CamerasProcessing import CameraProcess, StereoCamera
import cv2
LOGGER = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-5]
RGB_CAMERA_ID = None
IR_CAMERA_ID = None
FPS_TEST_TIME = 60  # seconds


def test_calibration_mapping():
    """
    Testing if calibrated frame have proper dimensions.
    """
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera(TEST=True)
    map_rgb, map_ir = stereo_camera.get_calibration_mapping('AppResources/calibration/data/stereoMap.xml')
    rgb_image = cv2.imread("tests/test_images/test_rgb_image.jpg")
    rgb_image = cv2.resize(rgb_image, (640, 488), interpolation=cv2.INTER_AREA)
    ir_image = cv2.imread("tests/test_images/test_ir_image.jpg")
    rgb_image = cv2.resize(ir_image, (640, 488), interpolation=cv2.INTER_AREA)
    rgb_image_mapped = cv2.remap(rgb_image, map_rgb.map_x,
                                 map_rgb.map_y,
                                 cv2.INTER_LANCZOS4,
                                 cv2.BORDER_CONSTANT, 0)
    assert rgb_image_mapped.shape == (488, 640, 3)
    ir_image_mapped = cv2.remap(ir_image, map_ir.map_x,
                                 map_ir.map_y,
                                 cv2.INTER_LANCZOS4,
                                 cv2.BORDER_CONSTANT, 0)
    assert ir_image_mapped.shape == (488, 640, 3)
    combined_frame = cv2.addWeighted(rgb_image_mapped, 0.5, ir_image_mapped, 0.5, 0.0)



def test_any_camera_connection():
    """
    Testing if application is able to establish connection with any camera connected to device.
    """
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
    """
    Testing if application is able to establish connection with any RGB camera.
    """
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
    """
    Testing if application is able to establish connection with any IR camera.
    """
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
    """
    Testing if application is able to establish connection with IR and RGB cameras using StereoCamera module.
    """
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera()
    frame = StereoCamera.synchronization_queue.get(timeout=15)
    if frame is None:
        LOGGER.error("Frame has not been received!")
        assert frame is not None

def test_frame_speed():
    """
    Testing application FPS that is possible to achieve.
    """
    os.chdir(ROOT_DIR)
    stereo_camera = StereoCamera()
    frame = StereoCamera.synchronization_queue.get(timeout=15)
    counter_end = time.time() + FPS_TEST_TIME
    frames = 0
    while time.time() < counter_end:
        frame = StereoCamera.synchronization_queue.get()
        frames += 1
    FPS = round(frames/60, 2)
    print("\nCamera achieved average of " + str(FPS) + " frames per second!")
    print("\nTest statistics:")
    print("\nFrames: " + str(frames))
    print("\nTime: " + str(FPS_TEST_TIME) + "s")


def kill_processes(proc):
    """
    Turning off processes used in tests.
    """
    for p in proc:
        p.kill()
        time.sleep(3)

