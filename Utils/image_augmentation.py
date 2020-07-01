import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import xml.etree.cElementTree as ET
import cv2
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from Utils.common_functions import make_and_clean_dir
ia.seed(1)

flags.DEFINE_string('data_train_dir', '../data/OneImage_train/', 'path to the all images')
flags.DEFINE_string('output_dir', '../data/OneImage_aug/', 'path to output images')


def save_image_and_boundbox(directory_path, file_prefix, file_names,
                            image_aug, bbs_aug,
                            image_width, image_height):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i in range(len(image_aug)):
        #imageio.imwrite(fileFolder+"/"+fileNames[i], image_aug[i])
        new_file_name = file_prefix + "_" + file_names[i]
        cv2.imwrite(directory_path + "/" + new_file_name, image_aug[i])

        bb = bbs_aug[i].bounding_boxes[0]
        bb.x1 = max(bb.x1, 3)
        bb.y1 = max(bb.y1, 3)
        bb.x2 = min(bb.x2, image_width-1)
        bb.y2 = min(bb.y2, image_height-1)

        if (bb.x2 < bb.x1+4) or (bb.y2 < bb.y1+4):
            print("Bounding box in "+new_file_name+" is too small and is not saved!")
            return

        file = open(directory_path + "/" + new_file_name.replace(".jpg", ".xml"), "w")
        file.write("<annotation>\n"
                   "\t<folder>TrainDataSet</folder>\n"
                   "\t<filename>{}</filename>\n"
                   "\t<path>{}</path>\n"
                   "\t<source>\n"
                   "\t\t<database>Unknown</database>\n"
                   "\t</source>\n"
                   "\t<size>\n"
                   "\t\t<width>{}</width>\n"
                   "\t\t<height>{}</height>\n"
                   "\t\t<depth>3</depth>\n"
                   "\t</size>\n"
                   "\t<segmented>0</segmented>\n"
                   "\t<object>\n"
                   "\t\t<name>Polyp</name>\n"
                   "\t\t<pose>Unspecified</pose>\n"
                   "\t\t<truncated>0</truncated>\n"
                   "\t\t<difficult>0</difficult>\n"
                   "\t\t<bndbox>\n"
                   "\t\t\t<xmin>{}</xmin>\n"
                   "\t\t\t<ymin>{}</ymin>\n"
                   "\t\t\t<xmax>{}</xmax>\n"
                   "\t\t\t<ymax>{}</ymax>\n"
                   "\t\t</bndbox>\n"
                   "\t\t</object>\n"
                   "</annotation>".format(new_file_name, directory_path + "/" + new_file_name, image_width,
                                          image_height, int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)))
        file.close()

        img = cv2.UMat(image_aug[i]).get()
        cv2.rectangle(img, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), (255, 255, 0), 2)
        cv2.imwrite("temp/ImgAugOutput/" + file_names[i].split(".")[0] + "_" + file_prefix + ".jpg", img)


def data_augmentation(data_train_path, output_dir):
    images = []
    bounding_boxes = []
    file_names = []
    image_width, image_height = 0, 0

    step = 0
    for filename in os.listdir(data_train_path):
        if ".jpg" in filename or ".bmp" in filename:
            fex = "." + str(filename.split(".")[-1])
            img = cv2.imread(data_train_path + "/" + filename, cv2.IMREAD_COLOR)
            images.append(img)
            file_names.append(filename)

            # Read bounding box
            file_path = os.path.join(data_train_path, filename.replace(fex, ".xml"))
            tree = ET.ElementTree(file=file_path)
            xmin, ymin, xmax, ymax = 0, 0, 0, 0
            for elem in tree.iter():
                if elem.tag == "xmin":
                    xmin = int(elem.text)
                if elem.tag == "ymin":
                    ymin = int(elem.text)
                if elem.tag == "xmax":
                    xmax = int(elem.text)
                if elem.tag == "ymax":
                    ymax = int(elem.text)

            bounding_box = BoundingBoxesOnImage([BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)], shape=img.shape)
            bounding_boxes.append(bounding_box)
            step += 1

    if len(bounding_boxes) is not 0:
        image_width = bounding_boxes[0].width
        image_height = bounding_boxes[0].height

    ia.seed(1)
    seq = iaa.Sequential([
        iaa.Affine(scale={"x": (0.95, 1.05)}),
        iaa.Affine(scale={"y": (0.95, 1.05)}),
        iaa.Affine(translate_percent={"x": (-0.05, 0.05)}),
        iaa.Affine(translate_percent={"y": (-0.05, 0.05)}),
        iaa.Affine(rotate=(-15, 15)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ], random_order=True)

    for step in range(20):  # This number should be adjusted
        print("Run epoch {}".format(step))
        image_aug, bbs_aug = seq(images=images, bounding_boxes=bounding_boxes)
        save_image_and_boundbox(directory_path=output_dir, file_prefix="epoch{}".format(step),
                                file_names=file_names, image_aug=image_aug, bbs_aug=bbs_aug,
                                image_width=image_width, image_height=image_height)


def main(_args):
    make_and_clean_dir(FLAGS.output_dir)
    data_augmentation(FLAGS.data_train_dir, FLAGS.output_dir)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
