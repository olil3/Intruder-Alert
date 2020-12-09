import cv2
import os
base_dir = os.getcwd()

global class_names
class_names = []

classFile = base_dir + '\\inference_files\\coco.names'
with open(classFile, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

configPath = base_dir + '\\inference_files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = base_dir + '\\inference_files\\frozen_inference_graph.pb'

threshold_value = 0.45  # Threshold to detect object
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def read_img(img, draw=True, object_list=[], json_config=None):
    class_ids, confidence_values, bbox = net.detect(img, confThreshold=threshold_value, nmsThreshold=0.1)
    if len(object_list) == 0:
        object_list = class_names

    object_info = []
    if len(class_ids) != 0:
        for class_id, confidence_value, box in zip(class_ids.flatten(), confidence_values.flatten(), bbox):
            class_name = class_names[class_id - 1]
            if class_name in object_list:
                object_info.append([box, class_name])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, class_name.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence_value * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    if json_config is not None:
        cv2.rectangle(img,
                      (json_config["line_data"]["x_coord_f"], json_config["line_data"]["y_coord_f"]),
                      (json_config["line_data"]["x_coord_l"], json_config["line_data"]["y_coord_l"]),
                      (0, 0, 255), 5)

    return img, object_info
