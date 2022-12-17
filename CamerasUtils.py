from multiprocessing import Value
from ctypes import c_bool

class CalibrationMapping:
    """Calibration module containing X nad Y mapping. Shared memory purpose."""
    def __init__(self, map_x, map_y):
        self.map_x = map_x
        self.map_y = map_y

class RecordingModule:
    """Recording module containing logic variable telling if recording is a fact.
    Contains IR and RGB frames Proxy List. Shared memory purpose."""
    def __init__(self):
        self.recorded_frames_RGB = None
        self.recorded_frames_IR = None
        self._recording = Value(c_bool, False)

    @property
    def recording(self):
        return self._recording.value

    @recording.setter
    def recording(self, value):
        self._recording.value = value


class EventLog:
    """Event log with log and errors Proxy Lists for shared memory purposes."""
    def __init__(self):
        self.camera_log = None
        self.new_camera_log = Value(c_bool, False)
        self.camera_errors = None
        self.new_camera_error = Value(c_bool, False)

    def put_log(self, event):
        self.camera_log.append(event)
        self.new_camera_log.value = True

    def put_error(self, event):
        self.camera_errors.append(event)
        self.new_camera_error.value = True


class DetectionMode:
    """Detection mode multiprocessing shared memory variable.
    Telling if detection is turned on and if labels should be displayed."""
    def __init__(self):
        self._detection = Value(c_bool, False)
        self._labels = Value(c_bool, False)

    @property
    def detection(self):
        return self._detection.value

    @detection.setter
    def detection(self, value):
        self._detection.value = value

    @property
    def labels(self):
        return self._labels.value

    @labels.setter
    def labels(self, value):
        self._labels.value = value


class DisplayMode:
    """Display mode multiprocessing shared memory variable.
    Depending on it RGB, IR or combined frames are displayed on interface."""
    def __init__(self):
        self._RGB = Value(c_bool, True)
        self._IR = Value(c_bool, True)

    @property
    def RGB(self):
        return self._RGB.value

    @RGB.setter
    def RGB(self, value):
        self._RGB.value = value

    @property
    def IR(self):
        return self._IR.value

    @IR.setter
    def IR(self, value):
        self._IR.value = value
