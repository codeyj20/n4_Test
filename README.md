## How It works

Add an explanation of the algorithm and how it works. Make sure to include details about how the code works, what it depends on, and any other relevant info. Add images or other descriptions for your project here. 

## Requirements
### Hardware
NVIDIA Jetson Orin Nano
USB webcam
Monitor
Power supply
Keyboard and mouse (optional, if using headed mode instead of NoMachine)
### Software/Libraries
Python 3
OpenCV
Install OpenCV with:
`pip3 install opencv-python`
Jetson Utils (included with Jetson Inference)
Jetson Inference can be installed from:  (GitHub link here)
NoMachine (optional, used to remotely access the Jetson desktop)

### Models
The project includes these pretrained OpenCV models.
Face Detection Model
opencv_face_detector.pbtxt
opencv_face_detector_uint8.pb
Age Prediction Model
age_deploy.prototxt
age_net.caffemodel

## Running this project

1. Connect the USB webcam, monitor, and power cable to the Jetson Orin Nano.
2. Open a terminal through NoMachine or use headed mode.
3. Clone the project repository.
`git clone https://github.com/student1/MyProject.git`
4. Navigate to the project folder.
`cd Estimation`
5. Run the detection program.
`python3 Verify.py`
6. Press the escape key to exit the program.

[View a video explanation here](video link)


