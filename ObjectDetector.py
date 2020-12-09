# File adapted from Murtaza Hassan's tutorials on Object Detection
# URL: https://www.murtazahassan.com/courses/opencv-projects/
# Youtube Link: https://www.youtube.com/watch?v=Vg9rrOFmwHo

import cv2
import os

# Get current directory's location
base_dir = os.getcwd()

# Declare an array to store the names of the classes in 'coco.names'.
global class_names
class_names = []

# Get the location of the coco.names.
classFile = base_dir + '\\inference_files\\coco.names'

# Read the file and append to array.
with open(classFile, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Get the paths of both the inference config and weights.
configPath = base_dir + '\\inference_files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = base_dir + '\\inference_files\\frozen_inference_graph.pb'

# Declare variables and set parameters for the inference model.
threshold_value = 0.45  # Threshold to detect object
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# This function receives an image as a numpy array and runs it through the inference model
# and detects if any objects belonging to the array of classes is present in the image.
# It then draws the bounding box around the image and returns both the modified image
# and its bounding box coordinates.
#
# params:
# img - Input image as numpy array
# draw - Whether to draw the bounding boxes or not.
# class_list - List of classes that have to be recognized.
# json_config - json_configuration storing the coordinates of the classified area
def read_img(img, draw=True, class_list=[], json_config=None):
    # Run the image through the DNN model.
    (class_ids, confidence_values, bbox) = net.detect(img, confThreshold=threshold_value, nmsThreshold=0.1)

    # If the provided class_list array is empty, draw bounding boxes for detected claszses.
    if len(class_list) == 0:
        class_list = class_names

    # Array storing the bounding box info and the class name of the detected object.
    object_info = []

    # Proceed to draw the bounding boxes.
    if len(class_ids) != 0:
        for class_id, confidence_value, box in zip(class_ids.flatten(), confidence_values.flatten(), bbox):
            class_name = class_names[class_id - 1]
            if class_name in class_list:
                object_info.append([box, class_name])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, class_name.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence_value * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Draw the classified area's bounding box if the json_config is valid.
    if json_config is not None:
        cv2.rectangle(img,
                      (json_config["line_data"]["x_coord_f"], json_config["line_data"]["y_coord_f"]),
                      (json_config["line_data"]["x_coord_l"], json_config["line_data"]["y_coord_l"]),
                      (json_config["line_data"]["color"]["B"],
                       json_config["line_data"]["color"]["G"],
                       json_config["line_data"]["color"]["R"]),
                      json_config["line_data"]["line_thickness"])

    # Return the modified image and the object_info array.
    return img, object_info
