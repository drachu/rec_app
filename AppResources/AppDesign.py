import shutil

from kivy.clock import mainthread
from kivymd.material_resources import dp
from kivymd.uix.behaviors.toggle_behavior import MDToggleButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDRaisedButton, MDFlatButton, MDFloatingActionButtonSpeedDial, MDIconButton
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.list import MDList, OneLineIconListItem, IconLeftWidget
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.button import MDRoundFlatIconButton
from kivy.core.window import Window
from kivymd.uix.spinner.spinner import MDSpinner
from kivymd.uix.dialog.dialog import MDDialog
from kivymd.uix.label import MDLabel, MDIcon


def generate_left_card(self):
    _left_card = MDCard(elevation=3, radius=20, size_hint=(0.3, 1))
    self.left_layout = MDGridLayout(cols=1, spacing=20, padding=20)
    self.university_logo = Image(source=r"AppResources/images/pg_logo.png", size_hint=(1, 0.2))
    self.button_exit = MDRoundFlatIconButton(icon="power",
                                             size_hint=(1, 0.1),
                                             text="Exit",
                                             on_press=self.close_application)

    self.information_layout = generate_information_layout(self)
    self.record_card = generate_record_card(self)
    self.log_card = generate_log_card(self)
    self.memory_info_card = generate_memory_info_card(self)
    self.left_layout.add_widget(self.university_logo)
    self.left_layout.add_widget(self.button_exit)
    self.left_layout.add_widget(self.information_layout)
    self.left_layout.add_widget(self.record_card)
    self.left_layout.add_widget(self.log_card)
    self.left_layout.add_widget(self.memory_info_card)
    _left_card.add_widget(self.left_layout)
    return _left_card


def generate_log_card(self):
    _log_card = MDCard(size_hint=(1, 0.3), radius=20, elevation=2)
    self.log_scroll = MDScrollView()
    self.log_list = MDList(id="log")
    self.log_scroll.add_widget(self.log_list)
    log_app_event(self, "App started")
    _log_card.add_widget(self.log_scroll)
    return _log_card


def generate_information_layout(self):
    self.information_layout = MDGridLayout(cols=2, size_hint=(1, 0.1))
    self.about_instruction_image = Image(source=r"AppResources/images/about.jpg")
    self.information_button = MDIconButton(icon='information-outline', on_press=self.show_info_dialog)
    self.switch_theme = MDSwitch(active=True)
    self.switch_theme.bind(active=self.switch_theme_mode)
    self.information_layout.add_widget(self.information_button)
    self.information_layout.add_widget(self.switch_theme)
    return self.information_layout


def generate_record_card(self):
    self.record_card = MDCard(size_hint=(1, 0.1), radius=20, elevation=2)
    self.record_buttons_layout = MDBoxLayout(orientation='horizontal', spacing=5)
    self.button_record = MDIconButton(icon='play-circle-outline', size_hint_x=0.15,
                                      on_press=self.switch_recording, pos_hint={"center_y": 0.5, "center_x:": 0.5})
    self.recording_progress_bar = MDProgressBar(pos_hint={"center_y": 0.5, "center_x:": 0.5}, size_hint=(0.55, 0.1),
                                                max=15, value=0)
    self.record_time = MDLabel(text='0:00', size_hint_x=0.30,
                               pos_hint={"center_y": 0.5, "center_x:": 0.5}, halign="center")
    self.record_buttons_layout.add_widget(self.button_record)
    self.record_buttons_layout.add_widget(self.recording_progress_bar)
    self.record_buttons_layout.add_widget(self.record_time)
    self.record_card.add_widget(self.record_buttons_layout)
    return self.record_card




def generate_memory_info_card(self):
    total, used, free = shutil.disk_usage("/")
    self.memory_info_card = MDCard(size_hint=(1, 0.2), elevation=3, padding=10, radius=20)
    self.memory_info_layout = MDGridLayout(cols=1, spacing=10)
    self.label_memory = MDLabel(text="Memory space", pos_hint={'y': .5})
    self.label_memory_state = MDLabel()
    self.memory_space_bar = MDProgressBar(max=total, value=used, size_hint=(1, 0.1))
    self.memory_info_layout.add_widget(self.label_memory)
    self.memory_info_layout.add_widget(self.memory_space_bar)
    self.memory_info_layout.add_widget(self.label_memory_state)
    self.memory_info_card.add_widget(self.memory_info_layout)
    return self.memory_info_card

def generate_detection_info_card(self):
    self.detection_info_card = MDCard(size_hint=(None, None),
                                      size=(150, 40),
                                      elevation=3,
                                      padding=10,
                                      radius=20,
                                      pos_hint={'center_x': .5, 'center_y': .95})
    self.detection_info_layout = MDGridLayout(cols=2, spacing=10)
    self.detection_info_icon = MDIcon(icon="square-off-outline",
                                      valign='middle')
    self.detection_info_label = MDLabel(text="Detekcja: wył.",
                                        valign='middle')
    self.detection_info_layout.add_widget(self.detection_info_icon)
    self.detection_info_layout.add_widget(self.detection_info_label)
    self.detection_info_card.add_widget(self.detection_info_layout)
    return self.detection_info_card


@mainthread
def log_camera_event(self, event_text):
    self.log_list.add_widget(OneLineIconListItem(
        IconLeftWidget(icon='camera', icon_size=dp(20)),
        text=event_text, font_style='Caption'))

@mainthread
def log_app_event(self, event_text="Record saved"):
    self.log_list.add_widget(OneLineIconListItem(
        IconLeftWidget(icon='application-cog-outline', icon_size=dp(20)),
        text=event_text, font_style='Caption'))


class MyToggleButton(MDFlatButton, MDToggleButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_down = self.theme_cls.primary_color


class CameraScreen(Screen):
    def __init__(self, app_self):
        Screen.__init__(self)
        self.name = "Camera"
        app_self.camera_layout = MDFloatLayout(size_hint=(1, 1))
        app_self.detection_info_card = generate_detection_info_card(app_self)
        app_self.image = Image()
        app_self.camera_layout.add_widget(app_self.image)
        app_self.camera_layout.add_widget(app_self.detection_info_card)
        self.add_widget(app_self.camera_layout)


class LoadingScreen(Screen):
    def __init__(self, app_self):
        Screen.__init__(self)
        self.name = "Loading"
        app_self.spinner = MDSpinner(
            size_hint=(0.25, 0.25),
            pos_hint={'center_x': .5, 'center_y': .5},
            active=True)
        self.add_widget(app_self.spinner)



def generate_bottom_panel(self):
    self.bottom_card = MDCard(size_hint=(1, 0.2), radius=20, elevation=3)
    self.bottom_layout = MDGridLayout(cols=2, padding=20, spacing=20)

    self.camera_mode_layout = MDGridLayout(cols=3, radius=20, spacing=5,
                                           size_hint=(1, 1), md_bg_color=[0.0, 0.0, 0.0, 0.2])
    self.button_camera_IR = MyToggleButton(text="Thermal",
                                           size_hint=(1, 1),
                                           group="cameraMode",
                                           on_press=self.switch_camera_mode)
    self.button_camera_RGB = MyToggleButton(text="RGB",
                                            size_hint=(1, 1),
                                            group="cameraMode",
                                            on_press=self.switch_camera_mode)
    self.button_camera_all = MyToggleButton(text="All",
                                            size_hint=(1, 1),
                                            group="cameraMode",
                                            on_press=self.switch_camera_mode)
    self.button_camera_all.state = 'down'
    self.camera_mode_layout.add_widget(self.button_camera_IR)
    self.camera_mode_layout.add_widget(self.button_camera_all)
    self.camera_mode_layout.add_widget(self.button_camera_RGB)

    self.detection_mode = generate_detection_mode(self)
    self.screen.add_widget(self.detection_mode)

    self.bottom_layout.add_widget(self.camera_mode_layout)
    self.bottom_card.add_widget(self.bottom_layout)
    return self.bottom_card


def generate_detection_mode(self):
    self.detection_mode = MDFloatingActionButtonSpeedDial(data={
        'No detection': ['square-off-outline', "on_press", lambda x:
        self.switch_detection_mode(detection=False, labels=False)],
        'Boxes only': ['vector-square', "on_press", lambda x:
        self.switch_detection_mode(detection=True, labels=False)],
        'Boxes&labels': ['label-multiple-outline', "on_press", lambda x:
        self.switch_detection_mode(detection=True, labels=True)]},
        hint_animation=True,
        root_button_anim=True)
    return self.detection_mode


def create_error_dialog(self, camera_errors, app_errors):
    self.error_dialog = None
    errors = []
    errors.extend(camera_errors)
    errors.extend(app_errors)
    if "Connection with camera has been closed!" in camera_errors:
        self.error_dialog = MDDialog(buttons=[MDFlatButton(text="Try restart", on_press=self.try_restart_camera),
                                              MDRaisedButton(text="Exit", on_press=self.close_application)])
    else:
        self.error_dialog = MDDialog(buttons=[MDFlatButton(text="Discard", on_press=self.close_error_dialog),
                                              MDRaisedButton(text="Exit", on_press=self.close_application)])

    self.error_dialog.title = "Errors found!"
    self.error_dialog.text = "\n".join(errors)
    self.error_dialog.auto_dismiss = False
    return self.error_dialog


def create_information_dialog(self):
    self.information_dialog = None
    self.information_dialog = MDDialog(buttons=[MDRaisedButton(text="Close", on_press=self.close_info_dialog)],
                                       type='custom', size_hint=(0.90, 0.90),
                                       content_cls=MDBoxLayout(
                                          MDLabel(text="This application was designed to support pedestrian detection as part of the engineering diploma project \"" \
                                                       "Pedestrian detection software using multimodal imaging and machine learning\". After starting the application, the loading animation should end with showing the image from the cameras."
                                                       "Cameras must be connected, otherwise an error window will appear. If you wait for a long time for the cameras to turn on, restart the application by reconnecting the cameras before doing so. "
                                                       "Below are the menu instructions.",
                                                  size_hint = (1, 0.25)),
                                          Image(source='AppResources/images/about.jpg'),
                                                orientation='vertical',
                                                spacing=dp(12),
                                                size_hint_y=None,
                                                height=Window.height - 150,
                                                width=Window.width - 150))
    self.information_dialog.title = "About"
    self.information_dialog.auto_dismiss = False
    self.information_dialog.padding = 50
    return self.information_dialog

class InformationDialogContent(MDBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orientation = 'vertical',
        self.spacing = dp(12)
        self.size_hint_y = None
        self.height = dp(200)




def generate_app_design(self):
    Window.size = (900, 600)
    self.screen = Screen()
    self.screen.width = 1000
    self.theme_cls.theme_style = "Dark"  # "Light"
    self.theme_cls.primary_palette = "Purple"  # "Purple", "Red"
    self.theme_cls.primary_hue = "200"  # "500"
    self.layout = MDGridLayout()
    self.layout.cols = 2
    self.layout.size_hint = (0.95, 0.95)
    self.layout.pos_hint = {"center_x": 0.5, "center_y": 0.5}
    self.layout.spacing = 20

    self.info_dialog = create_information_dialog(self)

    self.right_layout = MDGridLayout()
    self.right_layout.cols = 1
    self.right_layout.spacing = 20
    self.right_layout.padding = 0

    self.left_card = generate_left_card(self)
    self.layout.add_widget(self.left_card)
    self.layout.add_widget(self.right_layout)
    self.screen.add_widget(self.layout)

    self.bottom_panel = generate_bottom_panel(self)

    self.screen_manager = ScreenManager()
    self.screen_manager.add_widget(CameraScreen(self))
    self.screen_manager.add_widget(LoadingScreen(self))
    self.screen_manager.current = "Camera"
    self.right_layout.add_widget(self.screen_manager)
    self.right_layout.add_widget(self.bottom_panel)

    return self.screen
