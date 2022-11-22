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
from CamerasProcessing import StereoCamera
from AppResources.AppDesign import generate_app_design, create_error_dialog
import os

from kivy.config import Config

class MainApp(MDApp):
    app_errors = []
    error_dialog = None
    theme_cls = ThemeManager()
    camera_started = False
    record_time = 0
    memory_space = None

    class VideoThread(threading.Thread):
        def __init__(self, app_self):
            threading.Thread.__init__(self)
            self.app_self = app_self

        def run(self):
            self.app_self.load_video()

    def load_video(self):
        while True:
            # try:
                if not MainApp.camera_started:
                    stereoCamera = StereoCamera()
                    MainApp.camera_started = True
                    AppDesign.log_app_event(self, "Connection!")
                if StereoCamera.cameras_reading and not StereoCamera.synchronization_queue.empty():
                    frame = StereoCamera.synchronization_queue.get()
                    self.update_image(frame)
                if StereoCamera.event_log.new_camera_log.value:
                    self.update_camera_log(StereoCamera.event_log.camera_log)
            # except:
            #     MainApp.app_errors.append("Error loading frame from stereo cameras!")
            #     if not MainApp.error_dialog:
            #         self.check_camera_errors()

    def build(self):
        self.title = "Detection App"
        self.screen = generate_app_design(self)
        self.update_memory_space()
        videoThread = self.VideoThread(self)
        videoThread.daemon = True
        videoThread.start()
        return self.screen

    @mainthread
    def update_image(self, frame):
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
        #try:

    #except:
        #MainApp.app_errors.append("Connection with camera has been closed!")
        if MainApp.app_errors:
            self.show_error_dialog()

    @mainthread
    def check_camera_errors(self):
        if StereoCamera.camera_errors:
            self.show_error_dialog()

    def update_camera_log(self, logs):
        for log in logs:
            AppDesign.log_camera_event(self, log)
        logs[:] = []
        StereoCamera.event_log.new_camera_log.value = False

    def close_application(self, instance):
        # closing application
        MainApp.get_running_app().stop()

    def switch_camera_mode(self, instance):
        instance.state = 'down'
        if instance.text == 'All':
            StereoCamera.display_mode.RGB, StereoCamera.display_mode.IR = True, True
        else:
            StereoCamera.display_mode.RGB, StereoCamera.display_mode.IR = self.button_camera_RGB.state == 'down', self.button_camera_IR.state == 'down'

    def switch_recording(self, instance):
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
                                                   args=["IR", StereoCamera.recording_module.recorded_frames_RGB, self], daemon=True)
                save_process_rgb.start()
                save_process_ir.start()

    def update_record(self, instance):
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
        bytes_per_GB = 1024 * 1024 * 1024
        total, used, free = shutil.disk_usage("/")
        MainApp.memory_space = used/free
        self.label_memory_state.text = str(round(float(used) / bytes_per_GB, 1)) + "GB/" + str(round(float(total) / bytes_per_GB, 1)) + "GB"
        self.memory_space_bar.value = used


    def switch_detection_mode(self, detection, labels):
        StereoCamera.detection_mode.detection, StereoCamera.detection_mode.labels = detection, labels
        self.detection_info_label.text = "Detekcja: wł." if StereoCamera.detection_mode.detection else "Detekcja: wył."
        if labels:
            self.detection_info_icon.icon = 'label-multiple-outline'
        elif detection:
            self.detection_info_icon.icon = 'vector-square'
        else:
            self.detection_info_icon.icon = 'square-off-outline'

    def switch_theme_mode(self, instance, state):
        if state:
            self.theme_cls.theme_style = "Dark"  # "Light"
        else:
            self.theme_cls.theme_style = "Light"

    def show_ir_colors(self, instance):
        if StereoCamera.camera_colors_IR.value:
            StereoCamera.camera_colors_IR.value = False
            self.show_colors_button.icon = "invert-colors-off"
        else:
            StereoCamera.camera_colors_IR.value = True
            self.show_colors_button.icon = "invert-colors"

    def show_error_dialog(self):
        if not MainApp.error_dialog:
            StereoCamera.camera_errors = list(dict.fromkeys(StereoCamera.camera_errors))
            MainApp.app_errors = list(dict.fromkeys(MainApp.app_errors))
            MainApp.error_dialog = create_error_dialog(self, StereoCamera.camera_errors, MainApp.app_errors)
            StereoCamera.camera_errors = []
            MainApp.app_errors = []
            MainApp.error_dialog.open()

    def close_error_dialog(self, instance):
        MainApp.error_dialog.dismiss()
        MainApp.error_dialog = None

    def show_info_dialog(self, instance):
        self.info_dialog.open()

    def close_info_dialog(self, instance):
        self.info_dialog.dismiss()

    def try_restart_camera(self, instance):
        try:
            pass
        except:
            StereoCamera.camera_errors.append("Could not restart!")
            self.show_error_dialog()


def save_recording(name, records, app_self):
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
    os.environ['KIVY_HOME'] = "/config/"
    Config.set('kivy', 'window_icon', 'AppResources/images/app_logo.ico')
    Config.write()
    MainApp().run()
