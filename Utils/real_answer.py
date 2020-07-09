import lxml
from absl import app, flags
from absl.flags import FLAGS
import cv2
from Utils.image_to_tf import parse_xml

flags.DEFINE_string('image', '../data/SimpleDataset/226.jpg', 'path to input image')
flags.DEFINE_string('output', '../answer.jpg', 'path to output image')


def draw_answer(img, answers):
    for name, x1y1, x2y2 in answers:
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 2)
        img = cv2.putText(img, '{}'.format(name),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def get_answer(xml_path):
    annotation_xml = lxml.etree.fromstring(open(xml_path).read())
    annotation = parse_xml(annotation_xml)['annotation']
    x_min, y_min = [], []
    x_max, y_max = [], []
    classes_text = []
    if 'object' in annotation:
        for obj in annotation['object']:
            x_min.append(int(obj['bndbox']['xmin']))
            y_min.append(int(obj['bndbox']['ymin']))
            x_max.append(int(obj['bndbox']['xmax']))
            y_max.append(int(obj['bndbox']['ymax']))
            classes_text.append(obj['name'])
    answers = []
    for name, x1, y1, x2, y2 in zip(classes_text, x_min, y_min, x_max, y_max):
        answers.append([name, (x1, y1), (x2, y2)])
    return answers


def main(_argv):
    img = cv2.imread(FLAGS.image)
    xml_path = FLAGS.image.replace("jpg","xml")
    answers = get_answer(xml_path)
    img = draw_answer(img, answers)
    cv2.imwrite(FLAGS.output, img)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
