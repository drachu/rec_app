import cv2
import numpy as np
import threading
from multiprocessing import Queue
from datetime import datetime
import time

#from TensorflowLiteDetection import DetectionModel, preprocess_image, detection, draw_boxes, draw_boxes_and_labels


class StereoCamera():
    camera_log = []
    camera_errors = []
    camera_reading_RGB = True
    camera_reading_IR = False
    camera_RGB = False
    camera_IR = False
    receive_RGB = True
    receive_IR = True
    recording = False
    detection = False
    detection_boxes = False
    detection_labels = False
    cameras_reading = False
    synchronization_queue = Queue()
    video_queue_IR = Queue()
    video_queue_RGB = Queue()
    synchronization_lock = threading.Lock()
    stereo_map_RGB_x = None
    stereo_map_RGB_y = None
    stereo_map_IR_x = None
    stereo_map_IR_y = None
    thread_RGB = None
    thread_IR = None
    thread_synchronization = None
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

        StereoCamera.thread_RGB = CamThread("RGB", 1, StereoCamera.stereo_map_RGB_x, StereoCamera.stereo_map_RGB_y)
        StereoCamera.thread_RGB.daemon = True
        StereoCamera.thread_IR = CamThread("IR", 2, StereoCamera.stereo_map_IR_x, StereoCamera.stereo_map_IR_y)
        StereoCamera.thread_IR.daemon = True
        StereoCamera.thread_synchronization = SynchronizationThread()
        StereoCamera.thread_synchronization.daemon = True

        StereoCamera.thread_RGB.start()
        StereoCamera.thread_IR.start()
        StereoCamera.thread_synchronization.start()

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


class SynchronizationThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        StereoCamera.camera_log.append("Sync thread start")
        synchronization()


def synchronization():
    while True:
        if StereoCamera.camera_RGB and StereoCamera.camera_IR:
            #try:
                StereoCamera.cameras_reading = True
                frame_IR = StereoCamera.video_queue_IR.get()
                frame_RGB = StereoCamera.video_queue_RGB.get()
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

                if StereoCamera.detection:
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
                if StereoCamera.receive_RGB and StereoCamera.receive_IR:
                    StereoCamera.synchronization_queue.put(combined_frame)
                elif StereoCamera.receive_IR:
                    StereoCamera.synchronization_queue.put(frame_IR)
                elif StereoCamera.receive_RGB:
                    StereoCamera.synchronization_queue.put(frame_RGB)
            #except:
                #StereoCamera.camera_errors.append("Could not send image to application!")
                if StereoCamera.recording:
                #try:
                    frame_record_time = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
                    StereoCamera.recorded_frames_RGB.append([frame_RGB, frame_record_time])
                    StereoCamera.recorded_frames_IR.append([frame_IR, frame_record_time])
                #except:
                   # StereoCamera.camera_errors.append("Could not record frames!")


class CamThread(threading.Thread):

    def __init__(self, name, cam_id, stereo_map_x, stereo_map_y):
        threading.Thread.__init__(self)
        self.name = name
        self.cam_id = cam_id
        self.stereo_map_x = stereo_map_x
        self.stereo_map_y = stereo_map_y

    def run(self):
        #try:
            StereoCamera.camera_log.append("Starting " + self.name)
            if self.name == "RGB":
                cam = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
            elif self.name == "IR":
                cam = cv2.VideoCapture(self.cam_id)
            else:
                cam = None
            cam_view(cam, self.name, self.stereo_map_x, self.stereo_map_y)
        #except:
           # StereoCamera.camera_errors.append("Could not start " + self.name + " thread!")


def cam_view(cam, name, stereo_map_x, stereo_map_y):
    if cam.isOpened():  # try to get the first frame
        StereoCamera.camera_log.append(name + " camera found!")
        rval, frame = cam.read()
    else:
        StereoCamera.camera_log.append(name + " camera not found!")
        rval = False
        StereoCamera.camera_errors.append("Could not find " + name + " camera!")

    while rval:
        rval, frame = cam.read()
        if name == "RGB":
            time.sleep(1.0 / 8.7)
        frame = resize_and_map(name, frame, stereo_map_x, stereo_map_y)
        put_frame(name, frame)


def put_frame(name, frame):
    StereoCamera.synchronization_lock.acquire()
    #try:
    if name == "IR" and StereoCamera.camera_reading_RGB:
            StereoCamera.camera_IR = True
            StereoCamera.camera_reading_RGB = False
            StereoCamera.video_queue_IR.put(frame)
            StereoCamera.camera_reading_IR = True
    elif name == "RGB" and StereoCamera.camera_reading_IR:
            StereoCamera.camera_RGB = True
            StereoCamera.camera_reading_IR = False
            StereoCamera.video_queue_RGB.put(frame)
            StereoCamera.camera_reading_RGB = True
    #except:
        #StereoCamera.camera_errors.append(name + " frame put error!")
    StereoCamera.synchronization_lock.release()
