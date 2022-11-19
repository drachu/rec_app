import cv2
import numpy as np
import threading
from multiprocessing import Queue, Value, Process, Array, Manager, freeze_support
from datetime import datetime
from ctypes import c_bool, c_char_p
import time
import platform

from DetectionEdgeTPU import DetectionModelEdgeTPU
from DetectionCUDA import DetectionModelCUDA, draw_detections


class StereoCamera:
    manager = None
    camera_log = None
    camera_errors = None
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
    recorded_frames_RGB = None
    recorded_frames_IR = None

    def __init__(self):
        StereoCamera.manager = Manager()
        StereoCamera.camera_log = StereoCamera.manager.list()
        StereoCamera.camera_errors = StereoCamera.manager.list()
        StereoCamera.recorded_frames_IR = StereoCamera.manager.list()
        StereoCamera.recorded_frames_RGB = StereoCamera.manager.list()
        cv_file = cv2.FileStorage()
        cv_file.open('appResources/calibration/data/stereoMap.xml', cv2.FileStorage_READ)

        StereoCamera.stereo_map_RGB_x = cv_file.getNode('stereoMapRGB_x').mat()
        StereoCamera.stereo_map_RGB_y = cv_file.getNode('stereoMapRGB_y').mat()
        StereoCamera.stereo_map_IR_x = cv_file.getNode('stereoMapThermal_x').mat()
        StereoCamera.stereo_map_IR_y = cv_file.getNode('stereoMapThermal_y').mat()

        synchronization_process = SynchronizationProcess()
        synchronization_process.daemon = True
        synchronization_process.start()
        if platform.system() == "Windows":
            id_RGB = 1
            id_IR = 2
        else:
            id_RGB = 2
            id_IR = 1
        camera_process_RGB = CameraProcess("RGB", id_RGB, StereoCamera.stereo_map_RGB_x, StereoCamera.stereo_map_RGB_y)
        camera_process_RGB.daemon = True
        camera_process_IR = CameraProcess("IR", id_IR, StereoCamera.stereo_map_IR_x, StereoCamera.stereo_map_IR_y)
        camera_process_IR.daemon = True
        camera_process_RGB.start()
        camera_process_IR.start()


def resize_and_map(name, frame_to_calibrate, stereo_map_x, stereo_map_y):
    resize = cv2.resize(frame_to_calibrate, (640, 488), interpolation=cv2.INTER_LANCZOS4)
    frame_mapped = cv2.remap(resize, stereo_map_x, stereo_map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    if name == "RGB":
        frame_mapped = cv2.warpAffine(frame_mapped, np.float32([[1, 0, -45], [0, 1, -40]]), (640, 488))[0:488 - 70, 0:640 - 80]
        frame_mapped = cv2.resize(frame_mapped, (640, 488), interpolation=cv2.INTER_AREA)
    frame_mapped = cv2.resize(frame_to_calibrate, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    return frame_mapped


class SynchronizationProcess(Process):
    def __init__(self):
        Process.__init__(self, target=synchronization, args=(StereoCamera.camera_RGB,
        StereoCamera.camera_IR, StereoCamera.receive_RGB, StereoCamera.receive_IR, StereoCamera.recording,
        StereoCamera.detection, StereoCamera.cameras_reading, StereoCamera.synchronization_queue,
        StereoCamera.video_queue_IR, StereoCamera.video_queue_RGB, StereoCamera.camera_log, StereoCamera.camera_errors,
        StereoCamera.recorded_frames_IR, StereoCamera.recorded_frames_RGB))



def synchronization(camera_RGB, camera_IR, receive_RGB, receive_IR, recording, detection,
                    cameras_reading, synchronization_queue, video_queue_IR, video_queue_RGB,
                    camera_log, camera_errors, recorded_frames_IR, recorded_frames_RGB):
    try:
        detection_model = DetectionModelEdgeTPU()
        camera_log.append("EdgeTPU model loaded")
    except:
        try:
            detection_model = DetectionModelCUDA()
            print("CUDA/CPU model loaded")
            camera_log.append("CUDA/CPU model loaded")
        except:
            detection_model = None
            camera_log.append("Could not load model")
    camera_log.append("Sync process started")
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
                # if platform.system() == "Windows":
                #     frame_IR = cv2.bilateralFilter(frame_IR, 30, 15, 15)
                frame_IR = cv2.bitwise_not(frame_IR)
                frame_RGB = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2GRAY)
                combined_frame = cv2.addWeighted(frame_RGB, 0.3, frame_IR, 0.7, 0.0)
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_GRAY2BGR)
                frame_IR = cv2.cvtColor(frame_IR, cv2.COLOR_GRAY2BGR)
                frame_RGB = cv2.cvtColor(frame_RGB, cv2.COLOR_GRAY2BGR)
            #except:
                #StereoCamera.camera_errors.append("Image preprocess error!")

                if detection.value and detection_model:
                        if isinstance(detection_model, DetectionModelEdgeTPU):
                            image_det, image_orig = detection_model.preproces_image_for_detect(combined_frame)
                            output_data = detection_model.detection(image_det)
                            output_nms = detection_model.nms(output_data)
                            if StereoCamera.detection_labels:
                                combined_frame = detection_model.draw_boxes_and_labels(output_nms, image_orig)
                            elif StereoCamera.detection_boxes:
                                combined_frame = detection_model.draw_boxes_and_labels(output_nms, image_orig)
                            # combined_frame = cv2.resize(image_orig, (640, 488), interpolation=cv2.INTER_LANCZOS4)
                        else:
                            results = detection_model.model(combined_frame)
                            if results:
                                combined_frame = draw_detections(results, combined_frame)
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
                    recorded_frames_RGB.append([frame_RGB, frame_record_time])
                    recorded_frames_IR.append([frame_IR, frame_record_time])
                #except:
                   # StereoCamera.camera_errors.append("Could not record frames!")


class CameraProcess(Process):

    def __init__(self, name, cam_id, stereo_map_x, stereo_map_y):
        Process.__init__(self, target=cam_view, args=(cam_id, name, stereo_map_x, stereo_map_y,
        StereoCamera.camera_reading_RGB, StereoCamera.camera_reading_IR, StereoCamera.camera_RGB,
        StereoCamera.camera_IR, StereoCamera.video_queue_IR, StereoCamera.video_queue_RGB, StereoCamera.camera_log,
        StereoCamera.camera_errors))
        self.name = name
        self.cam_id = cam_id
        self.stereo_map_x = stereo_map_x
        self.stereo_map_y = stereo_map_y

def cam_view(cam_id, name, stereo_map_x, stereo_map_y, camera_reading_RGB, camera_reading_IR, camera_RGB,
             camera_IR, video_queue_IR, video_queue_RGB, camera_log, camera_errors):
    if name == "RGB" and platform.system() == "Windows":
        cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(cam_id)
    if cam.isOpened():  # try to get the first frame
        camera_log.append(name + " camera found!")
        rval, frame = cam.read()
    else:
        camera_log.append(name + " camera not found!")
        rval = False
        camera_errors.append("Could not find " + name + " camera!")

    while rval:
        rval, frame = cam.read()
        if name == "RGB" and platform.system() == "Windows":
            time.sleep(1.0 / 8.7)
        elif name == "IR" and platform.system() == "Linux":
            time.sleep(1.0/1000.0)
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
