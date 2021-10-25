from PIL import Image
import cv2
import numpy as np
import signal
from yolov4.tf import YOLOv4
import os, sys, time

class Yolov4(object):
    def __init__(self):
        self.yolo = YOLOv4()
        self.yolo.input_size = 416
        self.yolo.batch_size = 64
        self.yolo.classes = "./data/classes/target.names"
        self.yolo.make_model()
        self.yolo.load_weights("./data/target.weights", weights_type="yolo")
        self.iou_threshold = 0.3
        self.score_threshold = 0.25
        self.origin_frame = None

        print("Done initialize Yolo configs.")

        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        cv2.destroyAllWindows()
        sys.exit(0)

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def obj_color(self):
        # Colored labels dictionary
        color_dict = {
            'target_a': [0, 255, 255], 'target_b': [238, 123, 158]
        }
        return color_dict

    def cvDrawBoxes(self, detections, img):
        color_dict = self.obj_color()
        for detection in detections:
            x, y, w, h = detection[0],\
                detection[1],\
                detection[2],\
                detection[3]
            name_tag = str(self.yolo.classes[detection[4]])
            name_tag = name_tag.replace("_", " ")

            for name_key, color_val in color_dict.items():
                if name_key == name_tag:
                    color = color_val
                    xmin, ymin, xmax, ymax = self.convertBack(
                        float(x), float(y), float(w), float(h))
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)
                    cv2.rectangle(img, pt1, pt2, color, 1)
                    cv2.putText(img,
                                name_tag +
                                " [" + str(round(detection[5] * 100, 2)) + "]",
                                (pt1[0], pt1[1] -
                                 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 2)
        return img

    # For vision.py module (Overall Integration)
    def predict(self, frame):
        print("start yolo detection!")
        frame_height, frame_width, channels = frame.shape
        self.origin_frame = frame.copy()
        detections = self.yolo.predict(
            frame, iou_threshold=self.iou_threshold, score_threshold=self.score_threshold).tolist()
        none_index = -1
        for i, d in enumerate(detections):
            # delete detection with zero probability
            if d[5] == 0.0:
                none_index = i
            # change from ratio(float) to pixel(int)
            d[0] = int(d[0]*frame_width)
            d[1] = int(d[1]*frame_height)
            d[2] = int(d[2]*frame_width)
            d[3] = int(d[3]*frame_height)
        if none_index != -1:
            detections.pop(none_index)
            none_index = -1
        return detections

    def display(self):
        frame = self.cvDrawBoxes(detections, self.origin_frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Target Detector', image)
        cv2.waitKey(1)

    
if __name__ == '__main__':
    yolo = Yolov4()
    for i in range(1,10):
        image_path = 'data/images/'+ str(i) + '.jpg'
        img = cv2.imread(image_path)
        detections = yolo.predict(img)
        print("detections:", detections)
        yolo.display()
