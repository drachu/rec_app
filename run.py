import platform
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if platform.system() == "Windows":
        from App import run_app
        run_app()
    else:
        from CamerasProcessing import StereoCamera
        import cv2
        stereoCamera = StereoCamera()
        StereoCamera.detection.value = True
        StereoCamera.detection_labels.value = True
        StereoCamera.detection_boxes.value = True
        while True:
            frame = StereoCamera.synchronization_queue.get()
            cv2.imshow('frame', frame)
            cv2.waitKey(5)
