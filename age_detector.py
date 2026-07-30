from jetson_inference import detectNet, imageNet
from jetson_utils import videoSource, videoOutput, cudaToNumpy

import cv2
import os


# -----------------------------
# Load models
# -----------------------------

# Face detector
face_net = detectNet("facenet", threshold=0.5)

# Your custom age classifier
age_net = imageNet(
    model="/home/nvidia/jetson-inference/python/training/classification/models/Age_verifier1/resnet18.onnx",
    labels="/home/nvidia/jetson-inference/python/training/classification/models/Age_verifier1/labels.txt",
    input_blob="input_0",
    output_blob="output_0"
)


# -----------------------------
# Camera + display
# -----------------------------

camera = videoSource("/dev/video0")
display = videoOutput("display://0")


# Temporary file for face crops
temp_face = "/home/nvidia/AGE-CHECKER/temp_face.jpg"


# -----------------------------
# Main loop
# -----------------------------

while display.IsStreaming():

    img = camera.Capture()

    if img is None:
        continue


    # Detect faces
    detections = face_net.Detect(img)

    print("Faces detected:", len(detections))


    # Convert frame to OpenCV image
    frame = cudaToNumpy(img)


    for face in detections:

        # Get bounding box
        left = int(face.Left)
        top = int(face.Top)
        right = int(face.Right)
        bottom = int(face.Bottom)


        # Crop face
        face_crop = frame[top:bottom, left:right]


        # Skip bad crops
        if face_crop.size == 0:
            continue


        # Save cropped face
        cv2.imwrite(temp_face, face_crop)


        # Run age classifier
        class_id, confidence = age_net.Classify(temp_face)

        label = age_net.GetClassDesc(class_id)


        print(label, confidence)


        # Draw face box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )


        # Draw prediction
        cv2.putText(
            frame,
            f"{label}: {confidence:.1f}%",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


    # Render output
    display.Render(img)