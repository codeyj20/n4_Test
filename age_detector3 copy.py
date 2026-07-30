import cv2

from jetson_utils import (
    videoSource,
    videoOutput,
    cudaToNumpy,
    cudaFromNumpy
)


# -----------------
# Load AI models
# -----------------

# Load the face detection model.
# This model finds where faces are located in the camera image.
faceNet = cv2.dnn.readNet(
    "opencv_face_detector_uint8.pb",
    "opencv_face_detector.pbtxt"
)


# Load the age prediction model.
# This model takes a face image and predicts an age range.
ageNet = cv2.dnn.readNet(
    "age_net.caffemodel",
    "age_deploy.prototxt"
)


# Labels that match the model's possible predictions.
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



# -----------------
# Set up camera and display
# -----------------

# Connect to the USB webcam.
camera = videoSource(
    "v4l2:///dev/video0"
)


# Create a window to display the AI results.
display = videoOutput(
    "display://0"
)



# Keep running while the display window is open.
while display.IsStreaming():


    # Capture one frame from the camera.
    img = camera.Capture()


    # If no image was captured, try again.
    if img is None:
        continue



    # Convert the Jetson camera image into a format OpenCV can process.
    frame = cudaToNumpy(img)



    # -----------------
    # Step 1: Detect faces
    # -----------------

    # Get image height and width.
    # These are needed to convert model coordinates into pixel locations.
    h,w = frame.shape[:2]


    # Convert the image into a "blob".
    # Neural networks cannot process normal images directly,
    # so the image must be resized and formatted first.
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300,300),
        [104,117,123]
    )


    # Send the image into the face detection model.
    faceNet.setInput(blob)


    # Run the AI model to find faces.
    detections = faceNet.forward()



    # Check every possible face detected by the model.
    for i in range(detections.shape[2]):


        # Get the confidence score for this detection.
        confidence = detections[0,0,i,2]


        # Only use detections that are confident enough.
        if confidence > 0.7:


            # Convert the model's coordinates into pixel locations.
            x1 = int(
                detections[0,0,i,3]*w
            )

            y1 = int(
                detections[0,0,i,4]*h
            )

            x2 = int(
                detections[0,0,i,5]*w
            )

            y2 = int(
                detections[0,0,i,6]*h
            )



            # -----------------
            # Step 2: Crop the face
            # -----------------

            # Remove everything except the detected face.
            # The age model only needs the face, not the entire image.
            face = frame[y1:y2,x1:x2]


            # Skip if the face crop failed.
            if face.size == 0:
                continue



            # -----------------
            # Step 3: Predict age range
            # -----------------

            # Resize and format the face image for the age model.
            ageBlob = cv2.dnn.blobFromImage(
                face,
                1,
                (227,227),
                (78.426,87.769,114.896)
            )


            # Send the face image into the age model.
            ageNet.setInput(ageBlob)


            # Run the age prediction.
            prediction = ageNet.forward()



            # Find which age category had the highest confidence.
            index = prediction[0].argmax()


            # Convert the prediction number into an age label.
            age = ageList[index]


            # Convert confidence score into a percentage.
            confidence = prediction[0][index]*100



            # -----------------
            # Step 4: Display results
            # -----------------

            # Draw a box around the detected face.
            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )


            # Display the predicted age range and confidence.
            cv2.putText(
                frame,
                f"{age} {confidence:.1f}%",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )



    # Convert the OpenCV image back into a Jetson image
    # and display the final result.
    display.Render(
        cudaFromNumpy(frame)
    )