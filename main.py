# Contains code adapted from https://pyimagesearch.com
# URL: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

import imutils
import json
import math
import numpy
from ObjectDetector import *
from PIL import Image
from skimage.measure import compare_ssim
import sys

# Hardcode information about config file and delta file
config_file = base_dir + '\\config.json'
delta_file_name = base_dir + '\\modify_file.png'

with open(config_file) as json_file:
        global curr_json
        curr_json = json.loads(json_file.read())
        json_file.close()

# This function is used to obtain the bounding box for the area of interest, i.e.
# the area which will be marked as classified.
# params: capture_obj - OpenCV's VideoCapture Object.
def get_coord(capture_obj):
    # Execute until the first frame from the video stream is correctly obtained.
    while True:
        success_val, curr_snapshot = capture_obj.read()
        # If successfully read, break from the loop and continue.
        if success_val:
            break

    # Read the first frame and store it as an Image object.
    original_image_file = Image.fromarray(curr_snapshot)
    # Save the Image to a file with the name specified in delta_file_name
    original_image_file.save(delta_file_name)

    # Obtain the area of interest from the User.
    print("Modify the png with the name: " + delta_file_name + "\n")
    print("Draw a box around the area which must be safeguarded. \n")
    print("Enter 'Yes' when done. \n")

    # Loop until a valid input has been obtained.
    while True:
        complete_code = input()
        if complete_code != "Yes":
            print("Error! Wrong input. Valid inputs: 'Yes'\n")
        else:
            break

    # As the file has been modified, read it and store it in a variable.
    modified_image_file = numpy.array(Image.open(delta_file_name))

    # Grayscale the images to better obtain the differences in the images.
    gray_original = cv2.cvtColor(curr_snapshot, cv2.COLOR_BGR2GRAY)
    gray_modified = cv2.cvtColor(modified_image_file, cv2.COLOR_BGR2GRAY)

    # Get the similarity and difference values after running a comparison on the grayscale images.
    (score, diff) = compare_ssim(gray_original, gray_modified, full=True)

    # Convert difference score to unsigned 8-bit integer.
    diff = (diff * 255).astype("uint8")

    # Compute the threshold and hence the bounding boxes.
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Create an array to store all the bounding boxes.
    bounding_rect_details = []

    # Iterate over all the counters and generate bounding boxes.
    # Append them to the array.
    for c in contours:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        bounding_rect_details.append((x, y, w, h))
        cv2.rectangle(curr_snapshot, (x, y), (x + w, y + h), (0, 0, 255), 2)

    global idx_c
    idx_c = 0
    for i in range(0, len(bounding_rect_details)):
        tuple_c = bounding_rect_details[i]
        tuple_p = bounding_rect_details[idx_c]
        if (tuple_p[2] < tuple_c[2]) or (tuple_p[3] < tuple_c[3]):
            idx_c = i
        
    # Ask the user if the bounding boxes have been generated effectively.
    while True:
        print("Please check the opened window if the bounding rect as been draw correctly around the region of "
              "interest. \n")
        print("If yes, enter 'Yes'. Else, enter 'No'. \n")
        print("Please close the window before typing in your option. \n")

        # show the output images
        cv2.imshow("Original", curr_snapshot)
        cv2.waitKey(0)
        check_val = input()

        if check_val == "Yes":
            break
        elif check_val == "No":
            print("Restarting procedure for input. \n")
            get_coord(capture_obj)
            break

    # Return the bounding box array.
    return bounding_rect_details[idx_c]


# This function initializes the json object and calls get_coord if needed.
# params: capture_obj - OpenCV's VideoCapture Object.
def initialize_vars(capture_obj):
    if curr_json["line_data"]["is_initialized"] == "No":
        print("Coords not initialized. Opening script to initialize coordinates. \n")
        (x, y, w, h) = get_coord(capture_obj)

        curr_json["line_data"]["is_initialized"] = "Yes"
        curr_json["line_data"]["x_coord_f"] = x
        curr_json["line_data"]["y_coord_f"] = y
        curr_json["line_data"]["x_coord_l"] = x + w
        curr_json["line_data"]["y_coord_l"] = y + h

        with open(config_file, 'w') as new_json_file:
            json.dump(curr_json, new_json_file)


# This function checks if the distance between the two boxes is within the provided threshold.
# params:
# box_1 - coordinates of the first bounding box.
# box_2 - coordinates of the second bounding box.
# dist_threshold - threshold for the distance between the centers of both the bounding boxes.
def is_near(box_1, box_2, dist_threshold=0):
    x_mid_f = (box_1[0] + box_1[2]) / 2.0
    y_mid_f = (box_1[1] + box_1[3]) / 2.0

    x_mid_s = (box_2[0] + box_2[2]) / 2.0
    y_mid_s = (box_2[1] + box_2[3]) / 2.0

    distance_val = math.sqrt(pow((x_mid_s - x_mid_f), 2) + pow((y_mid_s - y_mid_f), 2))
    if distance_val <= dist_threshold:
        return True
    else:
        return False


# This function verifies if the config.json exists and is valid.
def check_config():
    if curr_json is None:
        print("Error: config.json missing. Please check the file location. \n")
        exit(0)

    if curr_json["video_stream_link"] is None:
        print("Error: Field 'video_stream_link' is missing. Please add the field.\n")
        exit(0)


# This function checks if the provided command-line arguments are valid.
def check_args(argv):
    argv_length = len(argv)
    if argv_length <= 1:
        if argv_length == 1:
            if argv[0] == "--reset":
                curr_json["line_data"]["is_initialized"] = "No"
            else:
                print("Illegal arguments provided. Usage: main.py [--reset]\n")
                exit(0)
    else:
        print("Illegal arguments provided. Usage: main.py [--reset]\n")
        exit(0)


# This function is the main body of the module.
# params: argv - provided command-line parameters.
def main(argv):
    # Check if argv is valid.
    check_args(argv)

    # Check if config.json is valid.
    check_config()

    # Initialize ObjectDetector.py's capture objects.
    cap = cv2.VideoCapture(curr_json["video_stream_link"])
    cap.set(3, curr_json["resolution_width"])
    cap.set(4, curr_json["resolution_height"])

    # Initialize script's variables.
    initialize_vars(cap)

    # Create a tuple to store the coordinates of the classified region's bounding box.
    classified_region = (curr_json["line_data"]["x_coord_f"],
                         curr_json["line_data"]["y_coord_f"],
                         curr_json["line_data"]["x_coord_l"],
                         curr_json["line_data"]["y_coord_l"])

    # Infinitely loop to process the videostream.
    while True:
        success, img = cap.read()

        # Call ObjectDetector.py's read_img function and run object detection on it.
        result, object_info = read_img(img, True, ['person'], curr_json)

        # For all recognized objects, draw bounding objects.
        for obj in object_info:
            if is_near(classified_region, obj[0], curr_json["distance_tolerance"]):
                cv2.putText(result,
                            curr_json["line_data"]["error_message"],
                            (int(cap.get(3) * 0.2), 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            curr_json["line_data"]["font_scale"],
                            (curr_json["line_data"]["color"]["B"],
                             curr_json["line_data"]["color"]["G"],
                             curr_json["line_data"]["color"]["R"]),
                            curr_json["line_data"]["line_thickness"])

        # Broadcast image onto OpenCV's output window.
        cv2.imshow("Output", result)
        cv2.waitKey(1)


if __name__ == "__main__":
    main(sys.argv[1:])
