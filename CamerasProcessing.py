import multiprocessing
import cv2
import numpy as np
from multiprocessing import Queue, Value, Process, Manager
from datetime import datetime
from ctypes import c_bool
import time
import platform
from DetectionEdgeTPU import DetectionModelEdgeTPU
from DetectionCUDA import DetectionModelCUDA
from CamerasUtils import DisplayMode, DetectionMode, CalibrationMapping, EventLog, RecordingModule


class StereoCamera:

    display_mode = DisplayMode()
    detection_mode = DetectionMode()
    event_log = EventLog()
    recording_module = RecordingModule()

    camera_switch = Value(c_bool, True)
    cameras_reading = Value(c_bool, False)
    camera_colors_IR = Value(c_bool, False)

    synchronization_queue = Queue()
    video_queue_IR = Queue()
    video_queue_RGB = Queue()
    synchronization_lock = multiprocessing.Lock()


    def get_calibration_mapping(self, path):
        calibration_file = cv2.FileStorage()
        calibration_file.open(path, cv2.FileStorage_READ)
        map_RGB = CalibrationMapping(calibration_file.getNode('stereoMapRGB_x').mat(),
                                     calibration_file.getNode('stereoMapRGB_y').mat())
        map_IR = CalibrationMapping(calibration_file.getNode('stereoMapThermal_x').mat(),
                                    calibration_file.getNode('stereoMapThermal_y').mat())
        return map_RGB, map_IR

    def start(self, id_RGB = 1 if platform.system() == "Windows" else 2, id_IR = 2 if platform.system() == "Windows" else 1):
        stereo_map_RGB, stereo_map_IR = StereoCamera.get_calibration_mapping(self, 'AppResources/calibration/data/stereoMap.xml')
        synchronization_process = SynchronizationProcess()
        camera_process_RGB = CameraProcess("RGB", id_RGB, stereo_map_RGB)
        camera_process_IR = CameraProcess("IR", id_IR, stereo_map_IR)
        synchronization_process.start()
        camera_process_RGB.start()
        camera_process_IR.start()

    def __init__(self, TEST = False):
        cameras_data_manager = Manager()
        StereoCamera.event_log.camera_log = cameras_data_manager.list()
        StereoCamera.event_log.camera_errors = cameras_data_manager.list()
        StereoCamera.recording_module.recorded_frames_RGB = cameras_data_manager.list()
        StereoCamera.recording_module.recorded_frames_IR = cameras_data_manager.list()
        if not TEST:
            StereoCamera.start(self)


class SynchronizationProcess(Process):

    def load_detection_model(self, event_log):
        try:
            self.detection_model = DetectionModelEdgeTPU()
            event_log.put_log("EdgeTPU model loaded")
        except:
            pass
        self.detection_model = DetectionModelCUDA()
        event_log.put_log("CUDA/CPU model loaded")
        # self.detection_model = None
        # event_log.put_log("Could not load model")

    def preprocess_combine_frames(self, frame_IR, frame_RGB, camera_colors):
        _combined_frame = cv2.addWeighted(frame_RGB, 0.3, frame_IR, 0.7, 0.0)
        _combined_frame = cv2.cvtColor(_combined_frame, cv2.COLOR_GRAY2BGR)
        _frame_IR = cv2.cvtColor(frame_IR, cv2.COLOR_GRAY2BGR)
        _frame_RGB = cv2.cvtColor(frame_RGB, cv2.COLOR_GRAY2BGR)
        if camera_colors.value:
            _frame_IR = cv2.applyColorMap(_frame_IR, cv2.COLORMAP_JET)
        return _combined_frame, _frame_IR, _frame_RGB

    def __init__(self):
        self.detection_model = None
        Process.__init__(self, target=self.synchronization, args=(StereoCamera.detection_mode,
        StereoCamera.display_mode, StereoCamera.recording_module,
        StereoCamera.cameras_reading, StereoCamera.synchronization_queue, StereoCamera.video_queue_IR,
        StereoCamera.video_queue_RGB, StereoCamera.event_log, StereoCamera.camera_colors_IR), daemon=True)

    def synchronization(self, detection_mode, display_mode, recording_module,
                        cameras_reading, synchronization_queue, video_queue_IR, video_queue_RGB,
                        event_log, camera_colors):
        event_log.put_log("Sync process started")
        self.load_detection_model(event_log)
        while True:
                cameras_reading.value = True
                frame_IR = video_queue_IR.get()
                frame_IR = frame_IR[30: 488, 35: 640]
                frame_RGB = video_queue_RGB.get()
                frame_RGB = frame_RGB[30: 488, 35: 640]
                combined_frame, frame_IR, frame_RGB = self.preprocess_combine_frames(frame_IR, frame_RGB, camera_colors)

                if detection_mode.detection and self.detection_model:
                    if isinstance(self.detection_model, DetectionModelEdgeTPU):
                        image_det, image_orig = self.detection_model.preproces_image_for_detect(combined_frame)
                        output_data = self.detection_model.detection(image_det)
                        output_nms = self.detection_model.nms(output_data)
                        image_orig = self.detection_model.draw_detections(output_nms, image_orig, labels=detection_mode.labels)
                        combined_frame = image_orig
                    else:
                        results = self.detection_model.model(combined_frame)
                        if results:
                            combined_frame = self.detection_model.draw_detections(results, combined_frame, labels=detection_mode.labels)
                # try:
                if display_mode.IR and display_mode.RGB:
                    synchronization_queue.put(combined_frame)
                elif display_mode.IR:
                    synchronization_queue.put(frame_IR)
                elif display_mode.RGB:
                    synchronization_queue.put(frame_RGB)
                if recording_module.recording:
                    frame_record_time = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
                    recording_module.recorded_frames_RGB.append([frame_RGB, frame_record_time])
                    recording_module.recorded_frames_IR.append([frame_IR, frame_record_time])


class CameraProcess(Process):

    def __init__(self, name, cam_id, stereo_map, TEST=False):
        Process.__init__(self, target=self.start_camera, args=(StereoCamera.camera_switch,
        StereoCamera.video_queue_IR, StereoCamera.video_queue_RGB, StereoCamera.event_log), daemon=True)
        self.name = name
        self.cam_id = cam_id
        self.stereo_map = stereo_map
        self.TEST = TEST

    def resize_and_map(self, frame_to_calibrate):
        # przeskalowanie obu klatek do jednego rozmiaru
        resize = cv2.resize(frame_to_calibrate, (640, 488), interpolation=cv2.INTER_LANCZOS4)
        # wprowadzenie mapowania
        frame_mapped = cv2.remap(resize, self.stereo_map.map_x,
                                 self.stereo_map.map_y,
                                 cv2.INTER_LANCZOS4,
                                 cv2.BORDER_CONSTANT, 0)
        # ręczne zmiany tylko dla klatek kamery RGB
        if self.name == "RGB":
            #dodatkowa translacja oraz repozycja
            frame_mapped = cv2.warpAffine(frame_mapped,
                                          np.float32([[1, 0, -45], [0, 1, -40]]),
                                          (640, 488))[0:488 - 70, 0:640 - 80]
            # powtórne przeskalowanie do porządanej rozdzielczości
            frame_mapped = cv2.resize(frame_mapped, (640, 488), interpolation=cv2.INTER_AREA)
        return frame_mapped

    def start_camera(self, camera_switch,
                 video_queue_IR, video_queue_RGB, event_log):
        if self.name == "RGB" and platform.system() == "Windows":
            cam = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(self.cam_id)
        if cam.isOpened():  # try to get the first frame
            event_log.put_log(self.name + " camera found!")
            rval, frame = cam.read()
        else:
            event_log.put_log(self.name + " camera not found!")
            rval = False

        while rval:
            rval, frame = cam.read()
            if self.name == "RGB" and platform.system() == "Windows":
                time.sleep(1.0 / 8.7)
            elif self.name == "IR" and platform.system() == "Linux":
                time.sleep(1.0 / 1000.0)
            if not self.TEST:
                frame = self.resize_and_map(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.name == "IR" and camera_switch.value:
                frame = cv2.bitwise_not(frame)
                put_frame(self.name, frame, camera_switch, video_queue_IR, video_queue_RGB)
            elif self.name == "RGB" and not camera_switch.value:
                put_frame(self.name, frame, camera_switch, video_queue_IR, video_queue_RGB)


def put_frame(name, frame, camera_switch, video_queue_IR, video_queue_RGB):
    StereoCamera.synchronization_lock.acquire()
    if name == "IR":
        video_queue_IR.put(frame)
        camera_switch.value = False
    elif name == "RGB":
        video_queue_RGB.put(frame)
        camera_switch.value = True
    StereoCamera.synchronization_lock.release()
