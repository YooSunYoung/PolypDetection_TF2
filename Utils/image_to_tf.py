import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree


flags.DEFINE_string('train_data_dir', '../data/OneImage_aug', 'path to raw polyp images for training')
flags.DEFINE_string('valid_data_dir', '../data/OneImage', 'path to raw polyp images for validation')
flags.DEFINE_string('train_output_file', '../data/polyp_train.tfrecord', 'path to dataset for training in tf format')
flags.DEFINE_string('valid_output_file', '../data/polyp_valid.tfrecord', 'path to dataset for validation in tf format')
flags.DEFINE_string('classes', '../data/polyp.names', 'classes file')


def build_example(data_dir, annotation, class_map):
    img_path = (data_dir + '/' + annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    x_min, y_min = [], []
    x_max, y_max = [], []
    classes, classes_text = [], []
    truncated, views, difficult_obj = [], [], []

    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            x_min.append(float(obj['bndbox']['xmin']) / width)
            y_min.append(float(obj['bndbox']['ymin']) / height)
            x_max.append(float(obj['bndbox']['xmax']) / width)
            y_max.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            #classes.append(class_map[obj['name']])
            classes.append(class_map[obj['name']]+1)
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=x_min)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=x_max)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=y_min)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=y_max)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def image_to_tfrecord(classes, data_dir, output_file):
    class_map = {name: idx for idx, name in enumerate(
        open(classes).read().splitlines())}
    n_classes = len(class_map)
    logging.info("Class mapping loaded: %s", class_map)
    writer = tf.io.TFRecordWriter(output_file)
    image_list = []
    xml_list = []
    files = os.listdir(data_dir)
    for file in files:
        if file.split(".")[-1] == "jpg":
            image_list.append((data_dir + "/" + file))
            xml_list.append((data_dir + "/" + file.replace("jpg", "xml")))
        elif file.split(".")[-1] == "bmp":
            image_list.append((data_dir + "/" + file))
            xml_list.append((data_dir + "/" + file.replace("bmp", "xml")))

    for xml_file in xml_list:
        annotation_xml = lxml.etree.fromstring(open(xml_file).read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(data_dir, annotation, class_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    logging.info("Done")


def main(_argv):
    for data, output in [[FLAGS.train_data_dir, FLAGS.train_output_file],
                         [FLAGS.valid_data_dir, FLAGS.valid_output_file]]:
        image_to_tfrecord(FLAGS.classes, data, output)


if __name__ == '__main__':
    app.run(main)
