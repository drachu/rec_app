import cv2

from CamThreading import StereoCamera

if __name__ == '__main__':
    stereoCamera = StereoCamera()
    while True:
        frame = StereoCamera.synchronization_queue.get()
        cv2.imshow('frame', frame)
        cv2.waitKey(5)