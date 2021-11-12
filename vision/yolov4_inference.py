# Import libraries
from ctypes import *
import math
import random
import cv2
import os
import numpy as np
import time
from darknet.build.darknet.x64 import darknet
import multiprocessing
import sys
from queue import Queue
from PIL import Image

# run yolo detection in new process


def yolo():
    # os.chdir(os.getcwd()+"/darknet/build/darknet/x64")
    p = multiprocessing.current_process()
    print("Starting:", p.name, p.pid)
    sys.stdout.flush()
    yolo_main = Yolov4()
    yolo_main.start()
    print("Exiting:", p.name, p.pid)
    sys.stdout.flush()


class Yolov4():
    def __init__(self):
        print("Initialize yolo.")
        self.target_size_ratio = 0.7
        self.time_delay = 0.1
        self.fps = None
        #self.count = 1
        self.filter = Queue(maxsize=10)
        #self.filter_count = 10
        #self.filter = []

        self.netMain = None
        self.metaMain = None
        self.altNames = None
        # print(os.path.abspath(__file__))
        # print(os.getcwd())
        # # Path to cfg
        # configPath = "./darknet/build/darknet/x64/cfg/yolov4.cfg"
        # # Path to weights
        # weightPath = "./darknet/build/darknet/x64/yolov4.weights"
        # # Path to meta data
        # metaPath = "./darknet/build/darknet/x64/cfg/coco.data"
        # Path to cfg
        configPath = "./darknet/build/darknet/x64/cfg/target_yolov4.cfg"
        # Path to weights
        weightPath = "./darknet/build/darknet/x64/target.weights"
        # Path to meta data
        metaPath = "./darknet/build/darknet/x64/cfg/target.data"
        # Checks whether file exists otherwise return ValueError
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath)+"`")
        # Checks the self.metaMain, NetMain and self.altNames. Loads it in script
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        #print("match: {}".format(match))
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                                #print("self.altNames: {}".format(self.altNames))
                    except TypeError:
                        pass
            except Exception:
                pass

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def obj_color(self):
        # Colored labels dictionary
        color_dict = {
            'person': [0, 255, 255], 'bicycle': [238, 123, 158], 'car': [24, 245, 217], 'motorbike': [224, 119, 227],
            'aeroplane': [154, 52, 104], 'bus': [179, 50, 247], 'train': [180, 164, 5], 'truck': [82, 42, 106],
            'boat': [201, 25, 52], 'traffic light': [62, 17, 209], 'fire hydrant': [60, 68, 169], 'stop sign': [199, 113, 167],
            'parking meter': [19, 71, 68], 'bench': [161, 83, 182], 'bird': [75, 6, 145], 'cat': [100, 64, 151],
            'dog': [156, 116, 171], 'horse': [88, 9, 123], 'sheep': [181, 86, 222], 'cow': [116, 238, 87], 'elephant': [74, 90, 143],
            'bear': [249, 157, 47], 'zebra': [26, 101, 131], 'giraffe': [195, 130, 181], 'backpack': [242, 52, 233],
            'umbrella': [131, 11, 189], 'handbag': [221, 229, 176], 'tie': [193, 56, 44], 'suitcase': [139, 53, 137],
            'frisbee': [102, 208, 40], 'skis': [61, 50, 7], 'snowboard': [65, 82, 186], 'sports ball': [65, 82, 186],
            'kite': [153, 254, 81], 'baseball bat': [233, 80, 195], 'baseball glove': [165, 179, 213], 'skateboard': [57, 65, 211],
            'surfboard': [98, 255, 164], 'tennis racket': [205, 219, 146], 'bottle': [140, 138, 172], 'wine glass': [23, 53, 119],
            'cup': [102, 215, 88], 'fork': [198, 204, 245], 'knife': [183, 132, 233], 'spoon': [14, 87, 125],
            'bowl': [221, 43, 104], 'banana': [181, 215, 6], 'apple': [16, 139, 183], 'sandwich': [150, 136, 166], 'orange': [219, 144, 1],
            'broccoli': [123, 226, 195], 'carrot': [230, 45, 209], 'hot dog': [252, 215, 56], 'pizza': [234, 170, 131],
            'donut': [36, 208, 234], 'cake': [19, 24, 2], 'chair': [115, 184, 234], 'sofa': [125, 238, 12],
            'pottedplant': [57, 226, 76], 'bed': [77, 31, 134], 'diningtable': [208, 202, 204], 'toilet': [208, 202, 204],
            'tvmonitor': [208, 202, 204], 'laptop': [159, 149, 163], 'mouse': [148, 148, 87], 'remote': [171, 107, 183],
            'keyboard': [33, 154, 135], 'cell phone': [206, 209, 108], 'microwave': [206, 209, 108], 'oven': [97, 246, 15],
            'toaster': [147, 140, 184], 'sink': [157, 58, 24], 'refrigerator': [117, 145, 137], 'book': [155, 129, 244],
            'clock': [53, 61, 6], 'vase': [145, 75, 152], 'scissors': [8, 140, 38], 'teddy bear': [37, 61, 220],
            'hair drier': [129, 12, 229], 'toothbrush': [11, 126, 158], 'target_a': [255, 0, 0], 'target_b': [0, 0, 255] 
        }
        return color_dict

    def cvDrawBoxes(self, detections, img):
        color_dict = self.obj_color()
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            name_tag = str(detection[0].decode())
            for name_key, color_val in color_dict.items():
                if name_key == name_tag:
                    color = color_val
                    xmin, ymin, xmax, ymax = self.convertBack(
                        float(x), float(y), float(w), float(h))
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)
                    cv2.rectangle(img, pt1, pt2, color, 1)
                    cv2.putText(img,
                                detection[0].decode() +
                                " [" + str(round(detection[1] * 100, 2)) + "]",
                                (pt1[0], pt1[1] -
                                 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 2)
        return img

    def focus_obj(self, detections, img):
        #color_dict = self.obj_color()
        H, W, channels = img.shape
        target_size = {"pt1": (int((1-self.target_size_ratio)*W/2), int((1-self.target_size_ratio)*H/2)),
                       "pt2": (int((1+self.target_size_ratio)*W/2), int((1+self.target_size_ratio)*H/2))}
        cv2.rectangle(img, target_size["pt1"],
                      target_size["pt2"], (0, 255, 0), 1)
        in_target = []
        for detection in detections:
            x, y = detection[2][0],\
                detection[2][1]

            # detection within target box
            if target_size["pt1"][0] <= x <= target_size["pt2"][0] and target_size["pt1"][1] <= y <= target_size["pt2"][1]:
                in_target.append(detection)

        # get object with max of sum of width and height of object box
        target_dict = {}
        for target in in_target:
            name = str(target[0].decode())
            w, h = target[2][2],\
                target[2][3]
            target_dict[name] = w+h

        # ignore person, just detect object
        if target_dict:
            sorted_target_list = [k for k, v in sorted(
                target_dict.items(), key=lambda item: item[1], reverse=True)]
            final_target = sorted_target_list[0]
            if final_target == "person":
                if len(sorted_target_list) > 1:
                    final_target = sorted_target_list[1]
                else:
                    final_target = "None"
            print("sorted_target_dict: {}".format(sorted_target_list))
            #print("final_target: {}".format(final_target))
        else:
            final_target = "None"

        target_txt = "Target Object: " + final_target
        img = cv2.putText(img, target_txt, (W//20, H//15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # filter target object using queue, eliminate noise
        if self.filter.full() == True:
            self.filter.get()
        self.filter.put(final_target)

        if self.filter.empty():
            return img
        queue_filter = list(self.filter.queue)
        final_filter_target = self.max_freq(queue_filter)
        target_txt = "Filtered Target Object: " + final_filter_target
        img = cv2.putText(img, target_txt, (W//20, H//15*2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, lineType=cv2.LINE_AA)

        return img

    # find the frequent element in list
    def max_freq(self, lst):
        return max(set(lst), key=lst.count)

    def start(self):
        print("start yolo!")
        # Uncomment to use Webcam
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("test2.mp4")                             # Local Stored video detection - Set input video
        # Returns the width and height of capture video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # Set out for video writer
        out = cv2.VideoWriter(                                          # Set the Output path for video writer
            "./darknet/build/darknet/x64/Demo/output.avi", cv2.VideoWriter_fourcc(
                *"MJPG"), 10.0,
            (frame_width, frame_height))

        print("Starting the YOLO loop...")

        # Create an image we reuse for each detect
        # Create image according darknet for compatibility of network
        darknet_image = darknet.make_image(frame_width, frame_height, 3)
        # Load the input frame and write output frame.
        while True:
            prev_time = time.time()
            # Capture frame and return true if frame present
            ret, frame_read = cap.read()
            # For Assertion Failed Error in OpenCV
            # Check if frame present otherwise he break the while loop
            if not ret:
                break

            # Convert frame into RGB from BGR and resize accordingly
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (frame_width, frame_height),
                                       interpolation=cv2.INTER_LINEAR)

            # Copy that frame bytes to darknet_image
            darknet.copy_image_from_bytes(
                darknet_image, frame_resized.tobytes())

            # Detection occurs at this line and return detections, for customize we can change the threshold.
            detections = darknet.detect_image(
                self.netMain, self.metaMain, darknet_image, thresh=0.25)
            # Call the function self.cvDrawBoxes() for colored bounding box per class
            #frame_resized = self.cvDrawBoxes(detections, frame_resized)
            image = self.focus_obj(detections, frame_resized)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            time.sleep(self.time_delay)

            self.fps = 1/(time.time()-prev_time)
            print("FPS: {}".format(self.fps))
            # Display Image window
            cv2.imshow('Demo', image)
            cv2.waitKey(3)
            # Write that frame into output video
            out.write(image)
        # For releasing cap and out.
        cap.release()
        out.release()
        print(":::Video Write Completed")

    def predict(self, image):
        #print("image.size: ", image.size)
        if image.size == 0:
            return None
        image_h, image_w, _ = image.shape
        darknet_image = darknet.make_image(image_w, image_h, 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(
                darknet_image, image_rgb.tobytes())
        detections = darknet.detect_image(
                self.netMain, self.metaMain, darknet_image, thresh=0.70)
        return detections
    
    def display(self, detections, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_drawn = self.cvDrawBoxes(detections, image)
        
        image_h, image_w, _ = image_drawn.shape
        img_cen_x = image_w / 2
        img_cen_y = image_h / 2

        # Check if the center of the detection box is within the bounding box 'fbbox' 
        fbbox = {
            'xmin': int(img_cen_x - (image_w * 0.7 / 2)), # Xmin
            'xmax': int(img_cen_x + (image_w * 0.7 / 2)), # Xmax
            'ymin': int(img_cen_y - (image_h * 0.7 / 2)), # Ymin
            'ymax': int(img_cen_y + (image_h * 0.7 / 2))  # Ymax
        }
        cv2.rectangle(image_drawn, (fbbox["xmin"], fbbox["ymin"]),
                      (fbbox["xmax"], fbbox["ymax"]), (0, 255, 0), 1)

        image = Image.fromarray(image_drawn.astype(np.uint8))
        image.show()

if __name__ == "__main__":
    pass
    # yolo_main = Yolo()                                                           # Calls the main function YOLO()