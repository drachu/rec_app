import CamThreading
import cv2



if __name__ == '__main__':
    stereoCamera = CamThreading.StereoCamera()
    while True:
        frame = CamThreading.StereoCamera.synchronization_queue.get()
        cv2.imshow('frame', frame)
        cv2.waitKey(5)