
# ECE 220 Honors Project Fall 2020

## Description

Object and Motion detection algorithms are growing fields in Computer Vision with many scientists and industrialists working and researching ways to improve their efficiency and accuracy. The usage of said algorithms extend into various many use-cases, such as Number-Plate Detection, Facial Recognition, Vehicle Detection in Self Driving cars, and so on.

This honors project intends to explore the usage of such algorithms and their implementation in detecting and alerting infringement/encroachment of physical space or bounds by an animate object - a human and/or an animal.

An example use-case of this project would be to detect and alert if an infant or a child were to cross a boundary (like entering the balcony) they are not supposed to. Another would be to alert if employees are found to be entering office spaces, they are barred from doing so; however, the latter example will not be explored in this project due to the added complexity of facial recognition but is left to be considered in the future.

## config.json representation
* **video_stream_link**: Path to video_stream for OpenCV's VideoStream module. This either be a video file, a rstp url, or a number denoting the a physical camera connected to your device. 
* **distance_tolerance**: Maximum distance between the bounding boxes that would trigger an alert. 
* **resolution_width**: Width of the videostream for OpenCV's VideoStream module. 
* **resolution_height**: Height of the videostream for OpenCV's VideoStream module.
* **line_data**: Specifications for the restricted area's bounding box.
  * **color**: Storing the color of the bounding box.
    * **B**: Blue color: Value: [0, 255]
    * **G**: Green color: Value: [0, 255]
    * **R**: Red color: Value: [0, 255]
  * **line_thickness**: Thickness of the line drawn (in px)
  * **font_scale**: Font scaling for bounding box's text.
  * **error_message**: Message to print as an alert when the object is within the distance tolerance.
  * **is_initialized**: Whether the bounding box has been created. DO NOT MODIFY as any illegal values for the coordinated may cause the program to crash.
  * **x_coord_f**: Top-Left x-coordinate of the bounding box to be drawn.
  * **y_coord_f**: Top-Left y-coordinate of the bounding box to be drawn.
  * **x_coord_l**: Bottom-Right x-coordinate of the bounding box to be drawn.
  * **y_coord_l**: Bottom-Right y-coordinate of the bounding box to be drawn.
  
## Syntax
Usage: ` main.py [--reset]`, where:
   * `--reset` resets the previous restricted area's coordinates. 

## Execution
Follow the instructions printed in console as they're self-explanatory.
