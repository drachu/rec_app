import io
from datetime import datetime
from zipfile import ZipFile

from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
import threading
import cv2

import AppDesign
from CamThreading import StereoCamera
from AppDesign import generate_app_design, create_error_dialog


class MainApp(MDApp):
    app_errors = []
    error_dialog = None
    theme_cls = ThemeManager()
    camera_started = False
    record_time = 0

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
                if StereoCamera.cameras_reading and not StereoCamera.synchronization_queue.empty():
                    frame = StereoCamera.synchronization_queue.get()
                    self.update_image(frame)
            # except:
            #     MainApp.app_errors.append("Error loading frame from stereo cameras!")
                if not MainApp.error_dialog:
                    self.check_camera_errors()

    def build(self):
        self.screen = generate_app_design(self)
        videoThread = self.VideoThread(self)
        videoThread.daemon = True
        videoThread.start()
        return self.screen

    @mainthread
    def update_image(self, frame):
        if self.loading.active:
            self.loading.active = False
            self.loading.size_hint = (0, 0)
            self.right_layout.padding = 0
            self.bottom_card.size_hint = (1, 0.2)
            AppDesign.log_app_event(self, "Connection!")
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
        self.update_camera_log()

    @mainthread
    def check_camera_errors(self):
        if StereoCamera.camera_errors:
            self.show_error_dialog()

    def update_camera_log(self):
        if StereoCamera.camera_log:
            AppDesign.log_camera_event(self, StereoCamera.camera_log[0])
            StereoCamera.camera_log.pop(0)

    def close_application(self, instance):
        # closing application
        MainApp.get_running_app().stop()

    def switch_camera_mode(self, instance):
        if self.button_camera_all.state == 'down':
            self.button_camera_all.state = 'down'
            StereoCamera.receive_RGB.value = StereoCamera.receive_IR.value = True
        else:
            StereoCamera.receive_RGB.value, StereoCamera.receive_IR.value = self.button_camera_RGB.state == 'down', self.button_camera_IR.state == 'down'

    def switch_recording(self, instance):
        if StereoCamera.cameras_reading.value:
            StereoCamera.recording.value = not StereoCamera.recording.value
            if StereoCamera.recording.value:
                self.button_record.icon = 'stop-circle-outline'
                Clock.schedule_interval(self.update_record, 1)
            else:
                Clock.unschedule(self.update_record)
                self.button_record.icon = 'play-circle-outline'
                self.record_time.text = "0:00"
                self.recording_progress_bar.value = 0
                MainApp.record_time = 0
                self.save_recording()

    def save_recording(self):
        zip_file_bytes_RGB = io.BytesIO()
        zip_file_bytes_IR = io.BytesIO()
        record_time = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        with ZipFile(zip_file_bytes_RGB, 'w') as zip_file:
            for image, image_name in StereoCamera.recorded_frames_RGB:
                is_success, buffer = cv2.imencode(".jpg", image)
                io_buf = io.BytesIO(buffer)
                zip_file.writestr(image_name + ".jpg", io_buf.getvalue())

        with open('records/recordedRGB/RGB_' + record_time + '.zip', 'wb') as f:
            f.write(zip_file_bytes_RGB.getvalue())
        StereoCamera.recorded_frames_RGB = []

        with ZipFile(zip_file_bytes_IR, 'w') as zip_file:
            for image, image_name in StereoCamera.recorded_frames_IR:
                is_success, buffer = cv2.imencode(".jpg", image)
                io_buf = io.BytesIO(buffer)
                zip_file.writestr(image_name + ".jpg", io_buf.getvalue())
        with open('records/recordedIR/IR_' + record_time + '.zip', 'wb') as f:
            f.write(zip_file_bytes_IR.getvalue())
        StereoCamera.recorded_frames_IR = []

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


    def switch_detection_mode(self, detection, detection_boxes, detection_labels):
        StereoCamera.detection.value, StereoCamera.detection_boxes.value, StereoCamera.detection_labels.value =\
            detection, detection_boxes, detection_labels

    def switch_theme_mode(self, instance, state):
        if state:
            self.theme_cls.theme_style = "Dark"  # "Light"
        else:
            self.theme_cls.theme_style = "Light"

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


if __name__ == '__main__':
    MainApp().run()
