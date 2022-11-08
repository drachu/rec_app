import cv2
import numpy as np
import threading
from multiprocessing import Queue, Value, Process
from datetime import datetime
from ctypes import c_bool
import time
import platform

#from TensorflowLiteDetection import DetectionModel, preprocess_image, detection, draw_boxes, draw_boxes_and_labels

class StereoCamera():
    camera_log = []
    camera_errors = []
    camera_reading_RGB = Value(c_bool, True)
    camera_reading_IR = Value(c_bool, False)
    camera_RGB = Value(c_bool, False)
    camera_IR = Value(c_bool, False)
    receive_RGB = Value(c_bool, True)
    receive_IR = Value(c_bool, True)
    recording = Value(c_bool, False)
    detection = Value(c_bool, False)
    detection_boxes = Value(c_bool, False)
    detection_labels = Value(c_bool, False)
    cameras_reading = Value(c_bool, False)
    synchronization_queue = Queue()
    video_queue_IR = Queue()
    video_queue_RGB = Queue()
    count = Value("i", 0)
    synchronization_lock = threading.Lock()
    stereo_map_RGB_x = None
    stereo_map_RGB_y = None
    stereo_map_IR_x = None
    stereo_map_IR_y = None
    detection_model = None
    recorded_frames_RGB = []
    recorded_frames_IR = []

    def __init__(self):
        cv_file = cv2.FileStorage()
        cv_file.open('appResources/calibration/data/stereoMap.xml', cv2.FileStorage_READ)

        StereoCamera.stereo_map_RGB_x = cv_file.getNode('stereoMapRGB_x').mat()
        StereoCamera.stereo_map_RGB_y = cv_file.getNode('stereoMapRGB_y').mat()
        StereoCamera.stereo_map_IR_x = cv_file.getNode('stereoMapThermal_x').mat()
        StereoCamera.stereo_map_IR_y = cv_file.getNode('stereoMapThermal_y').mat()
        if platform.system() == "Windows":
            id_RGB = 1
            id_IR = 2
        else:
            id_RGB = 2
            id_IR = 1
        thread_RGB = CamThread("RGB", id_RGB, StereoCamera.stereo_map_RGB_x, StereoCamera.stereo_map_RGB_y)
        thread_RGB.daemon = True
        thread_IR = CamThread("IR", id_IR, StereoCamera.stereo_map_IR_x, StereoCamera.stereo_map_IR_y)
        thread_IR.daemon = True
        thread_synchronization = SynchronizationThread()
        thread_synchronization.daemon = True

        thread_RGB.start()
        thread_IR.start()
        thread_synchronization.start()

        #StereoCamera.detection_model = DetectionModel()


def resize_and_map(name, frameToCalibrate, stereoMap_x, stereoMap_y):
    #try:
        resize = cv2.resize(frameToCalibrate, (640, 488), interpolation=cv2.INTER_LANCZOS4)
        frame_mapped = cv2.remap(resize, stereoMap_x, stereoMap_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        if name == "RGB":
            translate_x, translate_y = -45, -40
            rows, cols = 488, 640
            M = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
            frame_mapped = cv2.warpAffine(frame_mapped, M, (cols, rows))
            frame_mapped = frame_mapped[0:488 - 70, 0:640 - 80]
            frame_mapped = cv2.resize(frame_mapped, (640, 488), interpolation=cv2.INTER_AREA)
        return frame_mapped
    #except:
        #StereoCamera.camera_errors.append("Cound not resize frame!")


class SynchronizationThread(Process):
    def __init__(self):
        Process.__init__(self, target=synchronization, args=(StereoCamera.camera_RGB,
        StereoCamera.camera_IR, StereoCamera.receive_RGB, StereoCamera.receive_IR, StereoCamera.recording,
        StereoCamera.detection, StereoCamera.cameras_reading, StereoCamera.synchronization_queue,
        StereoCamera.video_queue_IR, StereoCamera.video_queue_RGB))


def synchronization(camera_RGB, camera_IR, receive_RGB, receive_IR, recording, detection,
                    cameras_reading, synchronization_queue, video_queue_IR, video_queue_RGB):
    while True:
        if camera_RGB.value and camera_IR.value:
            #try:
                cameras_reading.value = True
                frame_IR = video_queue_IR.get()
                frame_RGB = video_queue_RGB.get()
            #except:
                #StereoCamera.camera_errors.append("Could not get IR or RGB frame for synchronization!")

            #try:
                frame_IR = cv2.cvtColor(frame_IR, cv2.COLOR_BGR2GRAY)
                frame_IR = cv2.bilateralFilter(frame_IR, 30, 15, 15)
                frame_IR = cv2.bitwise_not(frame_IR)
                frame_RGB = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2GRAY)
                combined_frame = cv2.addWeighted(frame_RGB, 0.3, frame_IR, 0.7, 0.0)
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_GRAY2BGR)
                frame_IR = cv2.cvtColor(frame_IR, cv2.COLOR_GRAY2BGR)
                frame_RGB = cv2.cvtColor(frame_RGB, cv2.COLOR_GRAY2BGR)
            #except:
                #StereoCamera.camera_errors.append("Image preprocess error!")

                if detection.value:
                    pass
                # try:
                #     image_det, image_orig = preprocess_image(combined_frame)
                #     output_data = detection(image_det, StereoCamera.detection_model.interpreter,
                #                             StereoCamera.detection_model.input_details,
                #                             StereoCamera.detection_model.output_details)
                #     if StereoCamera.detection_labels:
                #         image_orig = draw_boxes_and_labels(output_data, image_orig)
                #     elif StereoCamera.detection_boxes:
                #         image_orig = draw_boxes(output_data, image_orig)
                #     combined_frame = cv2.resize(image_orig, (640, 488), interpolation=cv2.INTER_LANCZOS4)
                # except:
                #     StereoCamera.camera_errors.append("Detection error!")
            #try:
                if receive_RGB.value and receive_IR.value:
                    synchronization_queue.put(combined_frame)
                elif receive_IR.value:
                    synchronization_queue.put(frame_IR)
                elif receive_RGB.value:
                    synchronization_queue.put(frame_RGB)
            #except:
                #StereoCamera.camera_errors.append("Could not send image to application!")
                if recording.value:
                #try:
                    frame_record_time = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
                    StereoCamera.recorded_frames_RGB.append([frame_RGB, frame_record_time])
                    StereoCamera.recorded_frames_IR.append([frame_IR, frame_record_time])
                #except:
                   # StereoCamera.camera_errors.append("Could not record frames!")


class CamThread(Process):

    def __init__(self, name, cam_id, stereo_map_x, stereo_map_y):
        Process.__init__(self, target=cam_view, args=(cam_id, name, stereo_map_x, stereo_map_y,
        StereoCamera.camera_reading_RGB, StereoCamera.camera_reading_IR, StereoCamera.camera_RGB,
        StereoCamera.camera_IR, StereoCamera.video_queue_IR,
        StereoCamera.video_queue_RGB))
        self.name = name
        self.cam_id = cam_id
        self.stereo_map_x = stereo_map_x
        self.stereo_map_y = stereo_map_y

def cam_view(cam_id, name, stereo_map_x, stereo_map_y, camera_reading_RGB, camera_reading_IR, camera_RGB,
             camera_IR, video_queue_IR, video_queue_RGB):
    StereoCamera.camera_log.append("Starting " + name)
    if name == "RGB" and platform.system() == "Windows":
        cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(cam_id)
    if cam.isOpened():  # try to get the first frame
        StereoCamera.camera_log.append(name + " camera found!")
        rval, frame = cam.read()
    else:
        StereoCamera.camera_log.append(name + " camera not found!")
        rval = False
        StereoCamera.camera_errors.append("Could not find " + name + " camera!")

    while rval:
        rval, frame = cam.read()
        if name == "RGB" and platform.system() == "Windows":
            time.sleep(1.0 / 8.7)
        frame = resize_and_map(name, frame, stereo_map_x, stereo_map_y)
        put_frame(name, frame, camera_reading_RGB, camera_reading_IR, camera_IR, camera_RGB, video_queue_IR, video_queue_RGB)


def put_frame(name, frame, camera_reading_RGB, camera_reading_IR, camera_IR, camera_RGB, video_queue_IR, video_queue_RGB):
    StereoCamera.synchronization_lock.acquire()
    if name == "IR" and camera_reading_RGB.value:
        camera_IR.value = True
        camera_reading_RGB.value = False
        video_queue_IR.put(frame)
        camera_reading_IR.value = True
    elif name == "RGB" and camera_reading_IR.value:
        camera_RGB.value = True
        camera_reading_IR.value = False
        video_queue_RGB.put(frame)
        camera_reading_RGB.value = True
    #except:
        #StereoCamera.camera_errors.append(name + " frame put error!")
    StereoCamera.synchronization_lock.release()
