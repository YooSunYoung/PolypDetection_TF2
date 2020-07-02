import os
import shutil


def make_and_clean_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)
        os.mkdir(dir)
