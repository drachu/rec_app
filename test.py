import threading
import cv2

class camThread(threading.Thread):
    def __init__(self):
        self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        super().__init__()


    def run(self):
        while True:
            ret, frame = self.camera.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(10)
# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()

cams = camThread()
# Start the camera
camera = cv2.VideoCapture(2)

# Start the cleaning thread
cam_cleaner = CameraBufferCleanerThread(camera)

# Use the frame whenever you want
while True:
    if cam_cleaner.last_frame is not None:
        cv2.imshow('The last frame', cam_cleaner.last_frame)
    cv2.waitKey(10)
