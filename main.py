import json
import numpy
import math
from PIL import Image
from ObjectDetector import *
import imutils
from skimage.measure import compare_ssim

config_file = 'config.json'
delta_file_name = 'modify_file.png'
vid_stream = 0  # 'rtsp://192.168.1.34:8080/h264_pcm.sdp'

global curr_json
with open(config_file) as json_file:
    curr_json = json.loads(json_file.read())
    json_file.close()


def get_coord(capture_obj):
    while True:
        success_val, curr_snapshot = capture_obj.read()
        if success_val:
            break

    image_file = Image.fromarray(curr_snapshot)
    image_file.save(delta_file_name)

    print("Modify the png with the name: " + delta_file_name + "\n")
    print("Draw a box around the area which must be safeguarded. \n")
    print("Enter 'Yes' when done. \n")

    while True:
        complete_code = input()
        if complete_code != "Yes":
            print("Error! Wrong input. Valid inputs: 'Yes'\n")
        else:
            break

    modified_file = numpy.array(Image.open(delta_file_name))

    gray_base = cv2.cvtColor(curr_snapshot, cv2.COLOR_BGR2GRAY)
    gray_mod = cv2.cvtColor(modified_file, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(gray_base, gray_mod, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    bounding_rect_details = []

    for c in contours:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        bounding_rect_details.append((x, y, w, h))
        cv2.rectangle(curr_snapshot, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.rectangle(modified_file, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    cv2.imshow("Original", curr_snapshot)
    # cv2.imshow("Modified", modified_file)
    # cv2.imshow("Diff", diff)
    # cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

    while True:
        print("Please check the opened window if the bounding rect as been draw correctly around the region of "
              "interest. \n")
        print("If yes, enter 'Yes'. Else, enter 'No'. ")
        check_val = input()
        if check_val == "Yes":
            break
        elif check_val == "No":
            print("Restarting procedure for input. ")
            get_coord(capture_obj)
            break

    return bounding_rect_details[0]


def initialize_vars(capture_obj):
    if curr_json is None:
        print("Error: config.json missing. Please check the file location. \n")
        exit(0)

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


if __name__ == "__main__":
    cap = cv2.VideoCapture(vid_stream)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10,70)
    initialize_vars(cap)

    classified_region = (curr_json["line_data"]["x_coord_f"], curr_json["line_data"]["y_coord_f"],
                         curr_json["line_data"]["x_coord_l"], curr_json["line_data"]["y_coord_l"])

    while True:
        success, img = cap.read()
        result, object_info = read_img(img, True, ['person'], curr_json)
        for obj in object_info:
            if is_near(classified_region, obj[0], 200):
                cv2.putText(result, "PERSON IS VIOLATING SPACE", (int(cap.get(3) * 0.2), 50), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 0, 255), 1)

        cv2.imshow("Output", result)
        cv2.waitKey(1)
