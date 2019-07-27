import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
from model.yolo import Yolo_v3
import argparse
from utils.detect import load_images, load_class_names, draw_boxes, load_weights
from utils.stub import word_dictionary, pre_final_dictionary, final_dictionary, is_are, comma_and_placement
import emoji


def paragraph_generator(contents):

    with open('answer.txt', 'w') as answer:

        if contents == '':

            answer.write('No tag detected in the image.')

        else:

            answer.write(' '.join([str(item) for item in and_tag]) +
                         ' ' + is_tag + ' there in the given image.')


_MODEL_SIZE = (416, 416)
parser = argparse.ArgumentParser()
parser.add_argument(
    'images',
    nargs='*',
    help='Must take in an image',
    type=str)

args = parser.parse_args()
image = vars(args)

image_list = image['images']


img = image_list[0]

img_names = glob.glob(img)


batch_size = len(img_names)
batch = load_images(img_names, model_size=_MODEL_SIZE)
class_names = load_class_names('./labels/coco.names')
n_classes = len(class_names)
max_output_size = 10
iou_threshold = 0.5
confidence_threshold = 0.7

model = Yolo_v3(
    n_classes=n_classes,
    model_size=_MODEL_SIZE,
    max_output_size=max_output_size,
    iou_threshold=iou_threshold,
    confidence_threshold=confidence_threshold)

inputs = tf.placeholder(tf.float32, [batch_size, 416, 416, 3])

detections = model(inputs, training=False)

model_vars = tf.global_variables(scope='yolo_v3_model')
assign_ops = load_weights(model_vars, './weights/yolov3.weights')

with tf.Session() as sess:
    sess.run(assign_ops)
    detection_result = sess.run(detections, feed_dict={inputs: batch})

draw_boxes(img_names, detection_result, class_names, _MODEL_SIZE)
tf.reset_default_graph()


file = open('tag.txt', 'r')

contents = file.read()

words = contents.split('\n')

words = words[:-1]

sorted_words = sorted(words)

word_dictionary = word_dictionary(sorted_words)

pre_final_dictionary = pre_final_dictionary(word_dictionary)

final_dictionary = final_dictionary(pre_final_dictionary)

is_tag = is_are(final_dictionary)

and_tag = comma_and_placement(final_dictionary)


paragraph_generator(contents)

print('\n\n\nAnswer generated Succesfully {}'.format(
    emoji.emojize(":grinning_face_with_big_eyes:")))
