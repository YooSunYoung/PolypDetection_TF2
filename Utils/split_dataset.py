import shutil
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from Utils.common_functions import make_and_clean_dir


flags.DEFINE_string('data_dir', "../data/OneImage",
                    'path to Polyps data set for training, validation and test')
flags.DEFINE_string('output_dir_prefix', '../data/OneImage', 'prefix of the path for output directories')
flags.DEFINE_string('fraction', '70:30:0', 'fraction as train:valid:test')


def split_data_set(data_dir, fraction, output_json=None): # fraction as train:valid:test
    import json
    import random
    import glob
    jpg_data_list = glob.glob(os.path.join(data_dir, "*.jpg"))
    bmp_data_list = (glob.glob(os.path.join(data_dir, "*.bmp")))
    jpg_data_list = [[img_path, img_path.replace(".jpg", ".xml")] for img_path in jpg_data_list]
    bmp_data_list = [[img_path, img_path.replace(".bmp", ".xml")] for img_path in bmp_data_list]
    data_list = jpg_data_list + bmp_data_list
    sum_frac, data_list_len = sum(fraction), len(data_list)
    num_list = [int(data_list_len*frac/sum_frac) for frac in fraction]
    if sum(num_list) < data_list_len: num_list[0] += 1
    random.shuffle(data_list)
    data_set_list = {"train": data_list[:num_list[0]],
                     "valid": data_list[num_list[0]:num_list[0] + num_list[1]],
                     "test": data_list[num_list[0] + num_list[1]:]}
    logging.info(
        "\n# of train data: {}\n # of valid data: {}\n # of test data: {}\n".format(
            len(data_set_list["train"]), len(data_set_list["valid"]), len(data_set_list["test"])
        )
    )
    if output_json is not None:
        try:
            with open(output_json, 'w') as fp:
                json.dump(data_set_list, fp, indent=4)
        except FileNotFoundError:
            logging.info("Could not find the file to write the summary. Please check the name of the file path again.")
            logging.info("Proceed without recording the summary.")
    return data_set_list


def copy_files_into_sub_dataset(output_data_dir, sub_dataset_name, data_set_list):
    file_dir = os.path.join(output_data_dir)
    file_dir = file_dir + "_" + sub_dataset_name
    make_and_clean_dir(file_dir)

    for file in data_set_list[sub_dataset_name]:
        shutil.copy(file[0], file_dir)
        shutil.copy(file[1], file_dir)


def main(_argv):
    fraction = FLAGS.fraction.split(":")
    fraction = [float(frac) for frac in fraction]
    data_set_list = split_data_set(FLAGS.data_dir, fraction)
    for name in ["train", "valid", "test"]:
        copy_files_into_sub_dataset(output_data_dir=FLAGS.output_dir_prefix,
                                    sub_dataset_name=name,
                                    data_set_list=data_set_list)


if __name__ == '__main__':
    #os.chdir('/opt/project')
    app.run(main)
