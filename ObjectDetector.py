import cv2

threshold_value = 0.45  # Threshold to detect object

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def read_img(img, draw=True, object_list=[], json_config=None):
    classIds, confs, bbox = net.detect(img, confThreshold=threshold_value, nmsThreshold=0.)
    # print(classIds,bbox)

    if len(object_list) == 0:
        object_list = classNames

    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in object_list:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    if json_config is not None:
        cv2.rectangle(img,
                      (json_config["line_data"]["x_coord_f"], json_config["line_data"]["y_coord_f"]),
                      (json_config["line_data"]["x_coord_l"], json_config["line_data"]["y_coord_l"]),
                      (255, 127, 0), 5)

    return img, objectInfo
