# face-detection-tracking-example
A simple example for detecting and tracking either a single face or multiple faces.

## Requirements:

* OpenCV with Python Bindings
* dlib
* numpy

## Quick-Start:

**single_face_detection_tracking.py** is for single face detection (largest size face will be detected).

1. Ensure that within the file you have set the path for your OpenCV installation's HaarCascade xml files.
2. Run the program: ```python single_face_detection_tracking.py```
3. Press 'q' at any time to quit.

**multi_face_detection_tracking.py** is for multiple faces. All faces will be tracked and detected.

1. Ensure that within the file you have set the path for your OpenCV installation's HaarCascade xml files.
2. Run the program: ```python multi_face_detection_tracking.py```
3. Press 'q' at any time to quit.
