import cv2

from jetson_utils import (
    videoSource,
    videoOutput,
    cudaToNumpy,
    cudaFromNumpy
)


# Load face detection model
faceProto = "/home/nvidia/Gender-and-Age-Detection/opencv_face_detector.pbtxt"
faceModel = "/home/nvidia/Gender-and-Age-Detection/opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(
    faceModel,
    faceProto
)



# Load age prediction model
ageProto = "/home/nvidia/Gender-and-Age-Detection/age_deploy.prototxt"
ageModel = "/home/nvidia/Gender-and-Age-Detection/age_net.caffemodel"

ageNet = cv2.dnn.readNet(
    ageModel,
    ageProto
)



# Age categories the model can predict
ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]



# Detect faces in an image
def detectFaces(net, frame, confidence_threshold=0.7):

    h, w = frame.shape[:2]


    # Convert image into a format the face model understands
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300,300),
        [104,117,123],
        True,
        False
    )


    # Run face detection
    net.setInput(blob)
    detections = net.forward()


    boxes = []


    # Check every possible face detection
    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]


        if confidence > confidence_threshold:

            x1 = int(detections[0,0,i,3] * w)
            y1 = int(detections[0,0,i,4] * h)
            x2 = int(detections[0,0,i,5] * w)
            y2 = int(detections[0,0,i,6] * h)


            boxes.append(
                [x1,y1,x2,y2]
            )


    return boxes



# Open USB webcam
camera = videoSource(
    "v4l2:///dev/video0"
)


display = videoOutput(
    "display://0"
)



while display.IsStreaming():

    # Capture camera frame
    img = camera.Capture()


    if img is None:
        continue


    # Convert Jetson image to OpenCV image
    frame = cudaToNumpy(img)


    # Find faces
    faces = detectFaces(
        faceNet,
        frame
    )


    # Predict age for each face
    for box in faces:

        x1,y1,x2,y2 = box


        # Crop face area
        face = frame[
            max(0,y1):min(y2, frame.shape[0]),
            max(0,x1):min(x2, frame.shape[1])
        ]


        if face.size == 0:
            continue



        # Resize face to match model input
        face = cv2.resize(
            face,
            (227,227)
        )


        # Prepare face for age model
        blob = cv2.dnn.blobFromImage(
            face,
            1,
            (227,227),
            (78.426,87.769,114.896),
            swapRB=False
        )


        # Predict age range
        ageNet.setInput(blob)

        agePrediction = ageNet.forward()


        # Get highest confidence prediction
        ageIndex = agePrediction[0].argmax()

        age = ageList[ageIndex]

        confidence = agePrediction[0][ageIndex] * 100



        # Draw face box
        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        # Display age range and confidence
        cv2.putText(
            frame,
            f"{age} ({confidence:.1f}%)",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )



    # Display final output
    display.Render(
        cudaFromNumpy(frame)
    )