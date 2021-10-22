import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import signal

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/images/1.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

class Yolov4(object):
    def __init__(self):
        #self.input_size = [416, 416]
        #self.weights = './checkpoints/yolov4-416'
        self.input_size = FLAGS.size
        #self.iou = 0.45
        #self.score = 0.25
        self.nms_max_overlap = 0.75
        self.pred_bbox = None
        self.origin_img = None
        self.allowed_classes = None
        self.infer = None
        self.load_model()
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        cv2.destroyAllWindows()
        sys.exit(0)
        
    def load_model(self):
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']
        print("Done loading model!")

    def predict(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.origin_img = frame.copy()
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        #images_data = image_data[np.newaxis, ...].astype(np.float32)
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        self.batch_data = tf.constant(images_data)
        self.pred_bbox = self.infer(self.batch_data)
        for key, value in self.pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        self.pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        self.allowed_classes = list(class_names.values())  

        return self.pred_bbox      

    def display(self):
        image = utils.draw_bbox(self.origin_img, self.pred_bbox, allowed_classes = self.allowed_classes)
        image = Image.fromarray(image.astype(np.uint8))
        image.show()
        #image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        #cv2.imwrite(self.output, image)
        #cv2.imshow('Yolov4 Inference', image)
        #cv2.waitKey(1)

def main(_argv):
    yolo = Yolov4()
    for i in range(1,10):
        image_path = 'data/images/'+ str(i) + '.jpg'
        img = cv2.imread(image_path)
        pred_bbox = yolo.predict(img)
        print("pred_bbox:", pred_bbox)
        yolo.display()
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
