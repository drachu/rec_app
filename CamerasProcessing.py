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
    """Main Class that represents whole stereo camera's module."""

    TEST_MODE = False
    """Depending on this field exceptions are raised and not showed on interface"""
    display_mode = DisplayMode()
    """
     Cameras display mode instance with multiprocessing variables.
     """
    detection_mode = DetectionMode()
    """
     Cameras detection mode instance with multiprocessing variables.
     """
    event_log = EventLog()
    """
     Cameras event log instance containing error and log Proxy Lists.
     """
    recording_module = RecordingModule()
    """
     Recording module instance with RGB and IR frames lists.
     """
    camera_switch = Value(c_bool, True)
    """
     Camera switch for cameras synchronization. Is changed whenever other process is using put_frame function.
     """
    cameras_reading = Value(c_bool, False)
    """
     Logic value telling if cameras starting their work.
     """
    camera_colors_IR = Value(c_bool, False)
    """
     Logic value - when set on true JET mapped IR frames are displayed.
     """
    synchronization_queue = Queue()
    """
     Queue with processed and combined frames ready to deploy to application.
     """
    video_queue_IR = Queue()
    """
     Queue with IR frames got from IR camera.
     """
    video_queue_RGB = Queue()
    """
     Queue with RGB frames got from RGB camera.
     """
    synchronization_lock = multiprocessing.Lock()
    """
     Synchronization multiprocessing log used for cameras synchronization in put_frame function.
     """
    error_lock = multiprocessing.Lock()
    """
     Lock that prevents using error function by more than one process.
     """

    def get_calibration_mapping(self, path, event_log=event_log):
        """
        Reading from file calibration mapping variables for IR and RGB cameras.
            :param path: Path to file with calibration mapping.
            :param event_log: Event log to record eventual errors or logs.
            :return:
        """
        try:
            calibration_file = cv2.FileStorage()
            calibration_file.open(path, cv2.FileStorage_READ)
            map_RGB = CalibrationMapping(calibration_file.getNode('stereoMapRGB_x').mat(),
                                         calibration_file.getNode('stereoMapRGB_y').mat())
            map_IR = CalibrationMapping(calibration_file.getNode('stereoMapThermal_x').mat(),
                                        calibration_file.getNode('stereoMapThermal_y').mat())
            return map_RGB, map_IR
        except Exception as error:
            raise_error(error, event_log, message="Could not get calibration mapping!")

    def start(self, id_RGB = 1 if platform.system() == "Windows" else 2, id_IR = 2 if platform.system() == "Windows" else 1, event_log=event_log):
        """
        Method that is starting RGB, IR and synchronization processes.
            :param id_RGB: Identification for RGB camera depending on system.
            :param id_IR: Identification for IR camera depending on system.
            :param event_log: Event log to record eventual errors or logs.
        """
        stereo_map_RGB, stereo_map_IR = StereoCamera.get_calibration_mapping(self, 'AppResources/calibration/data/stereoMap.xml', event_log=event_log)
        try:
            synchronization_process = SynchronizationProcess()
            camera_process_RGB = CameraProcess("RGB", id_RGB, stereo_map_RGB)
            camera_process_IR = CameraProcess("IR", id_IR, stereo_map_IR)
            synchronization_process.start()
            camera_process_RGB.start()
            camera_process_IR.start()
        except Exception as error:
            raise_error(error, event_log, message="Could not start processes!")

    def __init__(self):
        cameras_data_manager = Manager()
        StereoCamera.event_log.camera_log = cameras_data_manager.list()
        StereoCamera.event_log.camera_errors = cameras_data_manager.list()
        StereoCamera.recording_module.recorded_frames_RGB = cameras_data_manager.list()
        StereoCamera.recording_module.recorded_frames_IR = cameras_data_manager.list()
        if not StereoCamera.TEST_MODE:
            StereoCamera.start(self, event_log=StereoCamera.event_log)


class SynchronizationProcess(Process):
    """
    Synchronization process class that is responsible for combining IR and RGB frame, detection and passing frames to application.
    """

    def load_detection_model(self, event_log):
        """
        Loading detection model (CUDA or EDGE TPU) file depending on device's system and capabilities.
            :param event_log: Event log to record eventual errors or logs.
        """
        try:
            self.detection_model = DetectionModelEdgeTPU()
            print("EdgeTPU model loaded")
            event_log.put_log("EdgeTPU model loaded")
        except:
            try:
                self.detection_model = DetectionModelCUDA()
                event_log.put_log("CUDA/CPU model loaded")
                print("CUDA/CPU model loaded")
            except:
                self.detection_model = None
                event_log.put_log("Could not load model")
                print("Could not load model")

    def preprocess_combine_frames(self, frame_IR, frame_RGB, camera_colors, event_log):
        """
        Combining RGB and IR frames and applying JET map color mapping.
            :param frame_IR: Frame from IR camera.
            :param frame_RGB: Frame from RGB camera.
            :param camera_colors: Passed value that is telling if JET map should be used.
            :param event_log: Event log to record eventual errors or logs.
            :return: Combined frame from IR and RGB images.
        """
        try:
            _combined_frame = cv2.addWeighted(frame_RGB, 0.3, frame_IR, 0.7, 0.0)
            _combined_frame = cv2.cvtColor(_combined_frame, cv2.COLOR_GRAY2BGR)
            _frame_IR = cv2.cvtColor(frame_IR, cv2.COLOR_GRAY2BGR)
            _frame_RGB = cv2.cvtColor(frame_RGB, cv2.COLOR_GRAY2BGR)
            if camera_colors.value:
                _frame_IR = cv2.applyColorMap(_frame_IR, cv2.COLORMAP_JET)
            return _combined_frame, _frame_IR, _frame_RGB
        except Exception as error:
            raise_error(error, event_log, message="Could not preprocess frames!")

    def __init__(self):
        self.detection_model = None
        Process.__init__(self, target=self.synchronization, args=(StereoCamera.detection_mode,
        StereoCamera.display_mode, StereoCamera.recording_module,
        StereoCamera.cameras_reading, StereoCamera.synchronization_queue, StereoCamera.video_queue_IR,
        StereoCamera.video_queue_RGB, StereoCamera.event_log, StereoCamera.camera_colors_IR), daemon=True)

    def synchronization(self, detection_mode, display_mode, recording_module,
                        cameras_reading, synchronization_queue, video_queue_IR, video_queue_RGB,
                        event_log, camera_colors):
        """
        Main synchronization process looped method that is reading frames from RGB and IR frames,
        processing detection and saving frames to recording module lists.
            :param detection_mode: Detection mode module main process reference for multiprocessing shared memory.
            :param display_mode: Display mode module main process reference for multiprocessing shared memory.
            :param recording_module: Recording mode module main process reference for multiprocessing shared memory.
            :param cameras_reading: Cameras reading bool main process reference for multiprocessing shared memory.
            :param synchronization_queue: Synchronization queue reference for multiprocessing shared memory.
            :param video_queue_IR: IR camera queue reference for multiprocessing shared memory.
            :param video_queue_RGB: RGB camera queue reference for multiprocessing shared memory.
            :param event_log: Event log to record eventual errors or logs.
            :param camera_colors: Passed value that is telling if JET map should be used.
        """
        event_log.put_log("Sync process started")
        self.load_detection_model(event_log)
        while True:
                cameras_reading.value = True
                frame_IR = video_queue_IR.get()
                frame_IR = frame_IR[30: 488, 35: 640]
                frame_RGB = video_queue_RGB.get()
                frame_RGB = frame_RGB[30: 488, 35: 640]
                combined_frame, frame_IR, frame_RGB = self.preprocess_combine_frames(frame_IR, frame_RGB, camera_colors, event_log)

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
    """
    Dynamic class for IR and RGB cameras processes that is responsible for cameras serving.
    """
    def __init__(self, name, cam_id, stereo_map, TEST=False):
        Process.__init__(self, target=self.start_camera, args=(StereoCamera.camera_switch,
        StereoCamera.video_queue_IR, StereoCamera.video_queue_RGB, StereoCamera.event_log), daemon=True)
        self.name = name
        self.cam_id = cam_id
        self.stereo_map = stereo_map
        self.TEST = TEST

    def resize_and_map(self, frame_to_calibrate, event_log):
        """
        Preprocessing frames - calibrating and resizing for proper values that system is normalized to.
            :param frame_to_calibrate: Passed frame to preprocess.
            :param event_log: Event log to record eventual errors or logs.
            :return: Calibrated frame.
        """
        try:
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
        except Exception as error:
            raise_error(error, event_log, message="Could not resize and map frame!")

    def start_camera(self, camera_switch, video_queue_IR, video_queue_RGB, event_log):
        """
        Starting camera connection using OpenCV functions. Starting loop if cameras are detected and connection is established.
            :param camera_switch: Camera synchronization reference for shared multiprocessing memory.
            :param video_queue_IR: IR camera queue reference for multiprocessing shared memory.
            :param video_queue_RGB: RGB camera queue reference for multiprocessing shared memory.
            :param event_log: Event log to record eventual errors or logs
        """
        if self.name == "RGB" and platform.system() == "Windows":
            cam = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(self.cam_id)
        if cam.isOpened():  # try to get the first frame
            event_log.put_log(self.name + " camera found!")
            rval, frame = cam.read()
        else:
            event_log.put_log(self.name + " camera not found!")
            raise_error(error=None, event_log=event_log, message="Cameras not found!")
            rval = False

        while rval:
            rval, frame = cam.read()
            if self.name == "RGB" and platform.system() == "Windows":
                time.sleep(1.0 / 8.7)
            elif self.name == "IR" and platform.system() == "Linux":
                time.sleep(1.0 / 1000.0)
            if not self.TEST:
                frame = self.resize_and_map(frame, event_log)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.name == "IR" and camera_switch.value:
                frame = cv2.bitwise_not(frame)
                put_frame(self.name, frame, camera_switch, video_queue_IR, video_queue_RGB)
            elif self.name == "RGB" and not camera_switch.value:
                put_frame(self.name, frame, camera_switch, video_queue_IR, video_queue_RGB)


def put_frame(name, frame, camera_switch, video_queue_IR, video_queue_RGB):
    """
    Static synchronized function that is passing RGB or IR frame from camera to its queues.
        :param name: Camera type.
        :param frame: Frame to pass to queue.
        :param camera_switch: Camera synchronization switch reference for shared memory.
        :param video_queue_IR: IR camera queue reference for multiprocessing shared memory.
        :param video_queue_RGB: RGB camera queue reference for multiprocessing shared memory.
    """
    StereoCamera.synchronization_lock.acquire()
    if name == "IR":
        video_queue_IR.put(frame)
        camera_switch.value = False
    elif name == "RGB":
        video_queue_RGB.put(frame)
        camera_switch.value = True
    StereoCamera.synchronization_lock.release()


def raise_error(error, event_log, message):
    """
    Function that raising or putting error to list depending on test mode.
        :param error: Passed error exception.
        :param event_log: Event log to put error to.
        :param message: Error message.
    """
    StereoCamera.error_lock.acquire()
    if StereoCamera.TEST_MODE:
        print(f"Unexpected {error=}, {type(error)=}")
        raise
    else:
        event_log.put_error(message)
    StereoCamera.error_lock.release()
