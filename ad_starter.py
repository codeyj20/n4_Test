# Import libraries
# Jetson Utils - handles the camera and displaying the results
# OpenCV - processes images and loads the models

import cv2

from jetson_utils import (
    videoSource,
    videoOutput,
    cudaToNumpy,
    cudaFromNumpy
)


# Load models


# Create a list to store the possible age predictions from the model.
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




# Set up camera and display
camera = videoSource("v4l2:///dev/video0")     # USB webcam 
display = videoOutput("display://0") 





# MAIN PROGRAM LOOP
while display.IsStreaming():
    img = camera.Capture()

    if img is None:
        continue


    # Convert Jetson image into an OpenCV image.
    # OpenCV uses this format for image processing.
    frame = cudaToNumpy(img)



   
    # Face detection
  




    # Age prediction







    # Display results
  



    # Convert the OpenCV image back into
    # a Jetson image and display it.

    display.Render(
        cudaFromNumpy(frame)
    )