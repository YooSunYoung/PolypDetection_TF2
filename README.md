### Overview
This project consists of some tools for polyp detection.

```
VitisAIForPolyp
    ├───EnvironmentSettings
    ├───Models
    ├───Training
    └───Utils (data pre-processing,
               evaluation,
               train results post-processing)
```
After running data pre-processing and training, more directories will be added like below.
```
    ├───data
    │     ├───PolypImages
    │     ├───PolypImages_train
    │     ├───PolypImages_aug
    │     └───PolypImages_valid
    └───results
          ├───checkpoints
          ├───logs
          └───TfliteModel
```
### Environment Setting
`EnvironmentSettings/` consists of the `.yml` files for conda environments.
You can create the virtual environment via command below regarding your hardware(CPU/GPU).
There are also pip recipes.
#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate polyp-tf2-cpu
```

<!-- 
```
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```
-->

#### Pip

```bash
pip install -r pip_requirements_cpu.txt
```

### Data
This project is for polyp detection so we are using the images taken by the endoscopy.
The size and the number of the channels(color) of images are stated in the `Models/config.py` as a dictionary.
The dataset should be a set of image(`.bmp` or `.jpg`) and label(`.xml`) pairs.
In the label(`.xml`) file, it should include the `filename`, image `size` (which consists of `width`, `height`, `channel`)
and the `object` which is the information of polyp in this case.

`object` should includes the `name` of the object such as 'Polyp' in this case.
It also has to include the location of the bounding box around the polyp as `xmin`, `ymin`, `xmax` and `ymax`.

The image files and the xml files should have same name before their file execution tag.
It would be easier to put all the data in one directory. For example,
```
├───data
      ├───PolypImages
            ├─── 001.bmp
            ├─── 001.xml
            ├─── 002.bmp
            ├─── 002.xml
                    .
                    .
                    .
```

### Data Pre-processing
#### 1. Split data into Train/Valid/Test set

Here is how to use `Utils/split_dataset.py` to split dataset randomly into `training`, `validation` and `test` sets.
Before you split the dataset all the dataset including images and the label files(usually .xml files) should be in one directory.
It doesn't remove the files in the `data_dir` but copy them in to other directories. 

- --data_dir : path to the directory which has all the dataset
- --output_dir_prefix : output files will be copied into `${output_dir_prefix}_train`, `${output_dir_prefix}_valid` and `${output_dir_prefix}_test`directory.                    
- --fraction : proportional expression for the number of train, valid and test images.
```bash
python Utils/split_dataset.py \
                        --data_dir ./data/PolypImages \
                        --output_dir_prefix ./data/PolypImages \
                        --fraction 5:3:2
```
> For example, if you execute the command line above, 
> and you have 10 images and labels in the `./data/PolypImages` directory,
> you will have 5 files in the `./data/PolypImages_train` directory,
> 3 files in the `./data/PolypImages_valid` directory and
> 2 files in the `./data/PolypImages_test` directory as a result.

#### 2. Image Augmentation
As the lack of polyp images we did image augmentation for the training dataset.
All the images should have the same size.

```bash
python Utils/data_augmentation.py \
                        --train_data_dir ./data/PolypImages_train \
                        --output_dir ./data/PolypImages_aug
```
> For now, it scales and rotates the images for certain amount.

#### 3. Put images and labels into tf.record file
Here is how to use `Utils/split_dataset.py` to put the dataset into tf.record file.
The path to the training and validation dataset should be specified by the option.
Test dataset doesn't have to be saved in tf.record format.
```bash
python Utils/image_to_tf.py \
                        --train_data_dir ./data/PolypImages_aug \
                        --valid_data_dir ./data/PolypImages_valid \
                        --train_output_file ./data/polyp_train.tfrecord \
                        --valid_output_file ./data/polyp_valid.tfrecord \
                        --classes ./data/polyp.names
```
> I put `./data/PolypImages_aug` into the `--train_data_dir` tag since I used the augmented images as a training dataset.
 
### Models
<!-- description for the models needed-->
- `Models/config.py` contains the information as a dictionary variable `configuration`.
  If you change the content in the `configuration` and it would affect all other processes.
  You may also put the information as arguments to the specific methods.
- `Models/dataset.py` contains the `PolypDataset` class and its static methods.
- `Models/models.py` contains the `PolypDetectionModel` class.
  For now, the only model is modified squeezenet.

### Training

```bash
python Training/train.py \
                        --dataset ./data/polyp_train.tfrecord \
                        --val_dataset ./data/polyp_valid.tfrecord \
                        --classes ./data/polyp.names \
                        --checkpoint_dir_path ./results/checkpoints \
                        --log_dir_path ./results/log \
                        --tflite_model_dir_path ./results/TfliteModel \
                        --epochs 1000 \
                        --save_points 50 \
                        --batch_size 1 \
                        --learning_rate 1e-3 
```

### Evaluation
```
python Utils/detect.py \
                    --classes ./data/polyp.names \ 
                    --weights ./results/checkpoints/polyp_train_50.tf \
                    --image ./data/PolypImages_test/001.bmp \
                    --output ./output.jpg
```
### Postprocessing (Weight and Model files) 

