import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

model_path = "blaze_face_short_range.tflite"
BaseOptions = mp.tasks.BaseOptions
FaceDetector=mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Actual face detector
options = FaceDetectorOptions(base_options = BaseOptions(model_asset_path=model_path), running_mode = VisionRunningMode.IMAGE)
with FaceDetector.create_from_options(options) as detector:
    pass
webcam = cv2.VideoCapture(0)


while True:
    retur, frame =webcam.read()
    frame = cv2.flip(frame, 1)
    if retur == True:
        cv2.imshow("My Camera", frame)
        key = cv2.waitKey(1)
        if key==ord("q"):
            break
webcam.release()
cv2.destroyAllWindows()
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = numpy_image)
face_detector_result = detector.detect(mp_image)

def main():

    if __name__ == "__main__":
        main()