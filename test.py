import time

from CamerasProcessing import StereoCamera


FPS_TEST_TIME = 60  # seconds

def test_frame_speed():
    stereo_camera = StereoCamera()
    frame = StereoCamera.synchronization_queue.get(timeout=15)
    counter_end = time.time() + FPS_TEST_TIME
    frames = 0
    while time.time() < counter_end:
        frame = StereoCamera.synchronization_queue.get()
        frames += 1
    FPS = round(frames/60, 2)
    print("\nCamera achieved average of " + str(FPS) + " frames per second!")
    print("\nTest statistics:")
    print("\nFrames: " + str(frames))
    print("\nTime: " + str(FPS_TEST_TIME) + "s")