import io
import shutil
from datetime import datetime
from zipfile import ZipFile
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
import threading
import cv2
from AppResources import AppDesign
from CamerasProcessing import StereoCamera, raise_error
from AppResources.AppDesign import generate_app_design, create_error_dialog
import os
from kivy.config import Config


class MainApp(MDApp):
    """
    Main application class representing program instance.
    """
    app_errors = []
    """
    List with recorded application errors.
    """
    error_dialog = None
    """
    Instance of error dialog.
    """
    camera_started = False
    """
     Logic value telling if cameras have started their work.
     """
    record_time = 0
    """
     Actual time passed while recording frames.
     """
    memory_space = None
    """
     Available space on device.
     """

    class VideoThread(threading.Thread):
        """
         Thread that is purpose is to connect synchronization process queue with application.
         """
        def __init__(self, app_self):
            threading.Thread.__init__(self)
            self.app_self = app_self

        def run(self):
            self.app_self.load_video()

    def load_video(self):
        """
        Looped by video thread method that is passing frames from synchronization queue to application.
        """
        while True:
            try:
                if not MainApp.camera_started:
                    stereoCamera = StereoCamera()
                    MainApp.camera_started = True
                    AppDesign.log_app_event(self, "Connection!")
                if StereoCamera.cameras_reading and not StereoCamera.synchronization_queue.empty():
                    frame = StereoCamera.synchronization_queue.get()
                    self.update_image(frame)
                if StereoCamera.event_log.new_camera_log.value:
                    self.update_camera_log(StereoCamera.event_log.camera_log)
                if StereoCamera.event_log.new_camera_error.value and not MainApp.error_dialog:
                    self.check_camera_errors(StereoCamera.event_log.camera_errors)
            except:
                MainApp.app_errors.append("Error loading frame from stereo cameras!")

    def build(self):
        """
        This method is building main application design and is starting video thread.
            :return: Screen instance with all children widgets passed to interface.
        """
        self.title = "Detection App"
        self.screen = generate_app_design(self)
        self.update_memory_space()
        videoThread = self.VideoThread(self)
        videoThread.daemon = True
        videoThread.start()
        return self.screen

    @mainthread
    def update_image(self, frame):
        """
        App loop scheduled method that is passing frames from Camera Processing to interface.
            :param frame: Passed frame from Camera Processing
        """
        if self.spinner.active:
            self.spinner.active = False
            self.screen_manager.current = "Camera"
        frame = cv2.resize(frame,
                           (int((self.image.height * 64) / 48),
                            int(self.image.height)),
                           interpolation=cv2.INTER_LANCZOS4)

        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

        if MainApp.app_errors:
            self.show_error_dialog()

    def check_camera_errors(self, logs):
        """
        This method is converting Proxy List camera errors to standard python list and is passing it to 'show_error_dialog' function.
            :param logs: Proxy List with Cameras Processing errors
        """
        error_logs = []
        for log in logs:
            error_logs.append(log)
        self.show_error_dialog(error_logs)

    def update_camera_log(self, logs):
        """
        This method converting Proxy List camera log to standard python list and is passing it to 'log_camera_event' function.
            :param logs: Proxy List with Cameras Processing logs
        """
        for log in logs:
            AppDesign.log_camera_event(self, log)
        logs[:] = []
        StereoCamera.event_log.new_camera_log.value = False

    def close_application(self, instance):
        """
        Closing whole application method.
            :param instance: Calling widget instance
        """
        MainApp.get_running_app().stop()

    def switch_camera_mode(self, instance):
        """
        This method is switching camera display mode depending on button instance calling it.
            :param instance: Calling button instance
        """
        instance.state = 'down'
        if instance.text == 'All':
            StereoCamera.display_mode.RGB, StereoCamera.display_mode.IR = True, True
        else:
            StereoCamera.display_mode.RGB, StereoCamera.display_mode.IR = self.button_camera_RGB.state == 'down', self.button_camera_IR.state == 'down'

    def switch_recording(self, instance):
        """
        This method is switching recording state located in StereoCamera module.
            :param instance: Calling button instance
        """
        if StereoCamera.cameras_reading.value:
            StereoCamera.recording_module.recording = not StereoCamera.recording_module.recording
            if StereoCamera.recording_module.recording:
                self.button_record.icon = 'stop-circle-outline'
                Clock.schedule_interval(self.update_record, 1)
            else:
                Clock.unschedule(self.update_record)
                self.button_record.icon = 'play-circle-outline'
                self.record_time.text = "0:00"
                self.recording_progress_bar.value = 0
                MainApp.record_time = 0
                AppDesign.log_app_event(self, "Record saved")
                save_process_rgb = threading.Thread(target=save_recording,
                                                    args=["RGB", StereoCamera.recording_module.recorded_frames_RGB, self], daemon=True)
                save_process_ir = threading.Thread(target=save_recording,
                                                   args=["IR", StereoCamera.recording_module.recorded_frames_IR, self], daemon=True)
                save_process_rgb.start()
                save_process_ir.start()

    def update_record(self, instance):
        """
        This is app looped function that is checking time which has passed. If it is more than 15 seconds recording is stopped.
            :param instance:
        """
        if MainApp.record_time < 15:
            MainApp.record_time += 1
            if len(str(MainApp.record_time)) < 2:
                self.record_time.text = "0:0" + str(MainApp.record_time)
            else:
                self.record_time.text = "0:" + str(MainApp.record_time)
            self.recording_progress_bar.value = MainApp.record_time
        elif MainApp.record_time >= 15:
            self.switch_recording(self)

    @mainthread
    def update_memory_space(self):
        """
        This method updated static field with left memory space on device and interface label.
        """
        bytes_per_GB = 1024 * 1024 * 1024
        total, used, free = shutil.disk_usage("/")
        MainApp.memory_space = used/free
        self.label_memory_state.text = str(round(float(used) / bytes_per_GB, 1)) + "GB/" + str(round(float(total) / bytes_per_GB, 1)) + "GB"
        self.memory_space_bar.value = used

    def switch_detection_mode(self, detection, labels):
        """
        This method switches detection mode displayed on interface card text and changes fields on StereoCamera.
            :param detection: Detection state
            :param labels: Labels displaying on predictions state
        """
        StereoCamera.detection_mode.detection, StereoCamera.detection_mode.labels = detection, labels
        self.detection_info_label.text = "Detection: on" if StereoCamera.detection_mode.detection else "Detection: off"
        if labels:
            self.detection_info_icon.icon = 'label-multiple-outline'
        elif detection:
            self.detection_info_icon.icon = 'vector-square'
        else:
            self.detection_info_icon.icon = 'square-off-outline'

    def switch_theme_mode(self, instance, state):
        """
        Method that is switching application visual theme mode - light or dark.
            :param instance: Calling widget instance
            :param state: Passed state - true (dark) or false (light)
        """
        if state:
            self.theme_cls.theme_style = "Dark"  # "Light"
            AppDesign.update_widgets_theme(self, [0.12941176470588237, 0.12941176470588237, 0.12941176470588237, 1.0],
                                           [1, 1, 1, 1])

        else:
            AppDesign.update_widgets_theme(self, [1.0, 1.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.87])

            self.theme_cls.theme_style = "Light"

    def show_ir_colors(self, instance):
        """
        Function called by button that is switching displaying IR frames in JET map color mode.
            :param instance: Calling widget state
        """
        if StereoCamera.camera_colors_IR.value:
            StereoCamera.camera_colors_IR.value = False
            self.show_colors_button.icon = "invert-colors-off"
        else:
            StereoCamera.camera_colors_IR.value = True
            self.show_colors_button.icon = "invert-colors"

    @mainthread
    def show_error_dialog(self, logs=[]):
        """
        Method that is displaying modal with errors from application and cameras.
            :param logs: Error logs from cameras
        """
        if not MainApp.error_dialog:
            MainApp.app_errors = list(dict.fromkeys(MainApp.app_errors))
            MainApp.error_dialog = create_error_dialog(self, logs, MainApp.app_errors)
            MainApp.app_errors = []
            MainApp.error_dialog.open()

    def close_error_dialog(self, instance=None):
        """
        Closing error dialog method.
            :param instance: Calling widget instance
        """
        MainApp.error_dialog.dismiss()
        StereoCamera.event_log.camera_errors[:] = []
        StereoCamera.event_log.new_camera_error.value = False
        MainApp.error_dialog = None

    def show_info_dialog(self, instance):
        """
        Showing information dialog method.
            :param instance: Calling widget instance
        """
        self.information_dialog.open()

    def close_info_dialog(self, instance):
        """
        Closing information dialog method.
            :param instance: Calling widget instance
        """
        self.information_dialog.dismiss()

    def try_restart_camera(self, instance):
        """
        This method is restarting stereo camera instance to again connect to cameras. Setting all needed values to default.
            :param instance: Calling widget instance
        """
        try:
            self.spinner.active = True
            self.screen_manager.current = "Loading"
            MainApp.camera_started = False
            self.close_error_dialog()
        except:
            raise_error(error=None, event_log=StereoCamera.event_log, message="Could not restart cameras processes!")
            self.show_error_dialog()


def save_recording(name, records, app_self):
    """
    Method that saves passed frames to file in zip format.
        :param name: Image type name
        :param records: List with records which includes frames and its names
        :param app_self: Application instance self
    """
    zip_file_bytes = io.BytesIO()
    record_time = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    with ZipFile(zip_file_bytes, 'w') as zip_file:
        for image, image_name in records:
            is_success, buffer = cv2.imencode(".jpg", image)
            io_buf = io.BytesIO(buffer)
            zip_file.writestr(image_name + ".jpg", io_buf.getvalue())

    with open('records/recorded' + name + '/' + name + '_' + record_time + '.zip', 'wb') as f:
        f.write(zip_file_bytes.getvalue())
    records[:] = []
    app_self.update_memory_space()


def run_app():
    """
    This static function is starting application and sets its configs.
    """
    os.environ['KIVY_HOME'] = "/config/"
    Config.set('kivy', 'window_icon', 'AppResources/images/app_logo.ico')
    Config.write()
    MainApp().run()
