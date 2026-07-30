import cv2

from jetson_utils import (
    videoSource,
    videoOutput,
    cudaToNumpy,
    cudaFromNumpy
)


# -----------------
# Load models
# -----------------

face_detection_model = cv2.dnn.readNet(
    "opencv_face_detector_uint8.pb",
    "opencv_face_detector.pbtxt"
)


age_detection_model = cv2.dnn.readNet(
    "age_net.caffemodel",
    "age_deploy.prototxt"
)


ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60+)"
]


camera = videoSource(
    "v4l2:///dev/video0"
)

display = videoOutput(
    "display://0"
)


while display.IsStreaming():

    img = camera.Capture()

    if img is None:
        continue


    frame = cudaToNumpy(img)


    # Detect faces

    h,w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300,300),
        [104,117,123]
    )

    face_detection_model.setInput(blob)

    detections = face_detection_model.forward()


    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.7:

            x1 = int(detections[0,0,i,3]*w)
            y1 = int(detections[0,0,i,4]*h)
            x2 = int(detections[0,0,i,5]*w)
            y2 = int(detections[0,0,i,6]*h)


            face = frame[y1:y2,x1:x2]


            if face.size == 0:
                continue


            ageBlob = cv2.dnn.blobFromImage(
                face,
                1,
                (227,227),
                (78.426,87.769,114.896)
            )


            age_detection_model.setInput(ageBlob)

            prediction = age_detection_model.forward()


            index = prediction[0].argmax()

            age = ageList[index]

            confidence = prediction[0][index]*100


            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )


            cv2.putText(
                frame,
                f"{age} {confidence:.1f}%",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )


    display.Render(
        cudaFromNumpy(frame)
    )