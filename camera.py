# Importing
import cv2 as cv

"""
Defines the camera object to use for capturing frames from the webcam
"""
class Camera:

    # Sets up the camera
    def __init__(self):
        self.camera = cv.VideoCapture(0)
        if not self.camera.isOpened():
            raise ValueError('Camera cannot open')

        # Width and height of the camera
        self.width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)

    # Gets rid of the camera when its closed
    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    # Stores the current frame of the webcam into a variable
    def getNextFrame(self):
        if self.camera.isOpened():
            returnval, frame = self.camera.read()

            if returnval:
                # Converts image from BGR format to RGB
                return (returnval, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return (returnval, None)

        else:
            return None

