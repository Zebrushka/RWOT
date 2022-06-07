import cv2
import time
import sys
import numpy as np
from Tracker.Sort import *
import random



def build_model(is_cuda):
    net = cv2.dnn.readNet("weights/yolov5n.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableBackend (cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    return net

# задачем чтение кадров 2 раз в секунду
FPS = 1
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
RTSP = 'rtsp://administrator:PQRYgZpQcInNwrbOo7f9@192.168.0.48:554/stream1'
# RTSP = 'rtsp://fpst-strm2.kmv.ru:9020/rtsp/15236961/accf06fae6d30fd49914'
# RTSP = 'demo/people_walk.mp4'
# RTSP = 'rtsp://fpst-strm2.kmv.ru:9020/rtsp/15236929/06b719b715465b7eefd8' #face
# RTSP = 'demo/test_image.jpeg'


def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def load_capture(RTSP):
    capture = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_FPS, 2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return capture


def load_classes():
    class_list = []
    with open("label/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


class_list = load_classes()


def wrap_detection(input_image, output_data, tracker):
    mot_tracker = tracker
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    # прилетели готовы детекции после NMS

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    result_img = np.copy(frame)
    dets = []
    count_detection = 0

    color_list = []
    for j in range(1000):
        color_list.append(((int)(random.randrange(255)), (int)(random.randrange(255)), (int)(random.randrange(255))))

    for j in range(len(indexes)):

        name = class_list[result_class_ids[j]]

        if name == 'person':
            count_detection += 1

    track = []

    if count_detection > 0:
        detects = np.zeros((count_detection, 5))
        count = 0
        # Подготовим в формат который ест трекер
        for j in range(len(indexes)):
            b = result_boxes[j]

            name = class_list[result_class_ids[j]]

            if name == 'person':
                x1 = int(b[0])
                y1 = int(b[1])
                x2 = int((b[0] + b[2]))
                y2 = int((b[1] + b[3]))
                box = np.array([x1, y1, x2, y2, result_confidences[0]])
                detects[count, :] = box[:]
                count += 1
                
        # Подаём в трекер и сразу имеем результат!
        if len(detects) != 0:
            track = mot_tracker.update(detects)
            for d in track:
                result_img = cv2.rectangle(result_img, ((int)(d[0]), (int)(d[1])), ((int)(d[2]), (int)(d[3])),
                                           color_list[(int)(d[4])], 2)

                x = int(d[0])
                y = int(d[1])
                
                id_track = str(d[4])
                result_img = cv2.putText(result_img, id_track, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                
                # TODO отрисовка поинтов
                


    return result_img, track


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result



def get_center_bbox(tracks, result_img):

    if tracks is not None:
        
        for track in tracks:
            x1 = float(track[0])
            y1 = float(track[1])
            x2 = float(track[2])
            y2 = float(track[3])

            x, y, track_number = y2/2, x2/2, track[4]

            result_img = cv2.circle(result_img, (x,y), radius=0, color=(0, 0, 255), thickness=-1)

            return result_img
        

    return x1, y1, x2, y2


#colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

net = build_model(is_cuda)
capture = load_capture(RTSP)
mot_tracker = Sort()


start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1
prev = 0

while capture.isOpened():

    time_elapsed = time.time() - prev

    _, frame = capture.read()
    if frame is None:
        print("End of stream")
        break

    if time_elapsed > 1. / FPS:
        prev = time.time()

        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)

        result_img, tracks = wrap_detection(inputImage, outs[0], mot_tracker)



        frame_count += 1
        total_frames += 1
        #
        # for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        #     color = colors[int(classid) % len(colors)]
        #     cv2.rectangle(frame, box, color, 2)
        #     cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        #     cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        #
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()
        #
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(result_img, fps_label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Считаем количество текущих треков == количеству людей в кадре
        people_count = str(len(tracks))
        cv2.putText(result_img, people_count, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # TODO: сделать визуализация треков
        # TODO: сделать проверку отрисованных треков с существующеми треками, не существующие треки удалять
        # print('----------------------------------------------------')
        # print(' ')
        # print(' ')
        # print(len(tracks))
        # print(' ')
        # print(tracks)


        cv2.imshow('demo', result_img)
        cv2.waitKey(1)



        if cv2.waitKey(1) > -1:
            print("finished by user")
            break

print("Total frames: " + str(total_frames))