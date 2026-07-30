import cv2

from jetson_inference import imageNet
from jetson_utils import (
    cudaFromNumpy
)

# -----------------------------
# Face Detection Function
# -----------------------------
def detectFaces(net, frame, conf_threshold=0.7):
    frameCopy = frame.copy()
    frameHeight = frameCopy.shape[0]
    frameWidth = frameCopy.shape[1]

    blob = cv2.dnn.blobFromImage(
        frameCopy,
        1.0,
        (300, 300),
        [104, 117, 123],
        True,
        False
    )

    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            faceBoxes.append([x1, y1, x2, y2])

            cv2.rectangle(
                frameCopy,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    return frameCopy, faceBoxes


# -----------------------------
# Load Models
# -----------------------------

faceProto = "/home/nvidia/Gender-and-Age-Detection/opencv_face_detector.pbtxt"
faceModel = "/home/nvidia/Gender-and-Age-Detection/opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(faceModel, faceProto)

ageNet = imageNet(
    model="/home/nvidia/jetson-inference/python/training/classification/models/Age_verifier1/resnet18.onnx",
    labels="/home/nvidia/jetson-inference/python/training/classification/models/Age_verifier1/labels.txt",
    input_blob="input_0",
    output_blob="output_0"
)
# Was trained on cropped images, so try less padding.
padding = 20 # Make this number lower to see if it improves accuracy. 

# -----------------------------
# Camera
# -----------------------------

video = cv2.VideoCapture(0)

while True:

    hasFrame, frame = video.read()

    if not hasFrame:
        break

    resultImg, faceBoxes = detectFaces(faceNet, frame)

    if not faceBoxes:
        cv2.imshow("Age Checker", resultImg)

        if cv2.waitKey(1) == 27:
            break

        continue

    for faceBox in faceBoxes:

        face = frame[
            max(0, faceBox[1]-padding):
            min(faceBox[3]+padding, frame.shape[0]-1),
            max(0, faceBox[0]-padding):
            min(faceBox[2]+padding, frame.shape[1]-1)
        ]

        if face.size == 0:
            continue

        face = cv2.resize(face, (224,224))

        face_cuda = cudaFromNumpy(face)

        class_id, confidence = ageNet.Classify(face_cuda)

        label = ageNet.GetClassDesc(class_id)

        confidence *= 100

        cv2.putText(
            resultImg,
            f"{label} ({confidence:.1f}%)",
            (faceBox[0], faceBox[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        print(f"Prediction: {label} ({confidence:.1f}%)")

    cv2.imshow("Age Checker", resultImg)

    if cv2.waitKey(1) == 27: # Press ESC to exit
        break

video.release()
cv2.destroyAllWindows()
