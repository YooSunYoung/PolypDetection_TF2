import numpy as np
import math
import logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    Lambda,
    ReLU,
    MaxPool2D
)
from Models import config
import Models.dataset as dataset
from Models.dataset import PolypDataset as polyp_dataset

logging.basicConfig(level=logging.INFO)


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def half_fire(x, num_squeeze_filters=12, num_expand_3x3_filters=12, batch_norm_flag=False):
    x = Conv2D(filters=num_squeeze_filters, kernel_size=1, strides=1, padding='same')(x)
    if batch_norm_flag: x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=num_expand_3x3_filters, kernel_size=3, strides=1, padding='same')(x)
    if batch_norm_flag: x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


class ObjectDetectionModel:
    def __init__(self, **kwargs):
        self.image_height = kwargs.get("image_height", 227)
        self.image_width = kwargs.get("image_width", 227)
        self.size = [self.image_height, self.image_width]
        self.n_channels = kwargs.get("n_channels", 3)
        self.n_boxes = kwargs.get("n_boxes", 3)
        self.n_grid = kwargs.get("grid_size", 4)
        self.n_classes = kwargs.get("n_classes", 0)
        self.model_name = kwargs.get("model_name", "squeeze_tiny")

    @staticmethod
    def default_grid(n_grid):
        grid = tf.meshgrid(tf.range(n_grid), tf.range(n_grid))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # the grid shape becomes (gridSize,gridSize,1,2)
        grid = tf.tile(grid, tf.constant([1, 1, 3, 1], tf.int32))  # the grid shape becomes (gridSize,gridSize,3,2)
        grid = tf.cast(grid, tf.float32)
        return grid

    def input_layer(self, name="Input"):
        return Input([self.size[0], self.size[1], self.n_channels], name=name)

    def output_layer(self, inputs):
        output = Conv2D(filters=self.n_boxes*5, kernel_size=1, strides=1, padding="same")(inputs)
        output = Lambda(lambda x: tf.reshape(x, (-1, self.n_grid, self.n_grid, self.n_boxes, 5)))(output)  #5, is objProb,cx,cy,w,h
        return output

    @staticmethod
    def compute_iou(boxes1, boxes2):
        boxes_t = []
        for boxes in [boxes1, boxes2]:
            boxes_t.append(tf.stack([boxes[..., 0] - boxes[..., 2] / 2.0,
                                     boxes[..., 1] - boxes[..., 3] / 2.0,
                                     boxes[..., 0] + boxes[..., 2] / 2.0,
                                     boxes[..., 1] + boxes[..., 3] / 2.0],
                                    axis=-1))
        boxes1_t, boxes2_t = boxes_t

        lu = tf.maximum(tf.cast(boxes1_t[..., :2], dtype=tf.float32), tf.cast(boxes2_t[..., :2], dtype=tf.float32))
        rd = tf.minimum(tf.cast(boxes1_t[..., 2:], dtype=tf.float32), tf.cast(boxes2_t[..., 2:], dtype=tf.float32))

        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(
            tf.cast(square1, dtype=tf.float32) + tf.cast(square2, dtype=tf.float32) - tf.cast(inter_square,
                                                                                              dtype=tf.float32), 1e-10)
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def get_loss(self, y_true=None, y_pred=None, training=False):
        grid = self.default_grid(self.n_grid)
        grid_size = self.n_grid
        n_boxes = self.n_boxes
        pred_obj_conf = y_pred[:, :, :, :, 0]
        pred_box_offset_coord = y_pred[:, :, :, :, 1:]

        pred_box_normalized_coord = tf.concat([(pred_box_offset_coord[:, :, :, :, 0:2] + grid) / grid_size,
                                               tf.square(pred_box_offset_coord[:, :, :, :, 2:])], axis=-1)

        target_obj_conf = y_true[:, :, :, 0]
        target_obj_conf = tf.reshape(target_obj_conf, shape=[-1, grid_size, grid_size, 1])

        target_box_coord = y_true[:, :, :, 1:]
        target_box_coord = tf.reshape(target_box_coord, shape=[-1, grid_size, grid_size, 1, 4])
        target_box_normalized_coord = tf.tile(target_box_coord, multiples=[1, 1, 1, n_boxes, 1])

        target_box_offset_coord = tf.concat(
            [tf.cast(target_box_normalized_coord[:, :, :, :, 0:2] * grid_size, dtype=tf.float32) - grid,
             tf.cast(tf.sqrt(target_box_normalized_coord[:, :, :, :, 2:]), dtype=tf.float32), ]
            , axis=-1)
        pred_ious = self.compute_iou(target_box_normalized_coord, pred_box_normalized_coord)
        predictor_mask_max = tf.reduce_max(pred_ious, axis=-1, keepdims=True)
        predictor_mask = tf.cast(pred_ious >= tf.cast(predictor_mask_max, dtype=tf.float32),
                                 tf.float32) * target_obj_conf
        no_obj_mask = tf.ones_like(predictor_mask) - predictor_mask

        # computing the confidence loss
        obj_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(predictor_mask * (pred_obj_conf - predictor_mask)), axis=[1, 2, 3]))
        no_obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(no_obj_mask * pred_obj_conf), axis=[1, 2, 3]))

        # computing the localization loss
        predictor_mask_none = predictor_mask[:, :, :, :, None]
        loc_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(predictor_mask_none * (target_box_offset_coord - pred_box_offset_coord)),
                          axis=[1, 2, 3]))

        if (math.isnan(obj_loss) is True) or (math.isnan(no_obj_loss) is True) or (math.isnan(loc_loss) is True):
            pass

        loss = 10 * loc_loss + 2 * obj_loss + 0.5 * no_obj_loss

        if training is True:
            tf.summary.scalar("loc_loss", np.sum(10 * loc_loss))
            tf.summary.scalar("obj_loss", np.sum(2 * obj_loss))
            tf.summary.scalar("non_obj_loss", np.sum(0.5 * no_obj_loss))
            # print("loss is:{}".format(loss))
        return loss


class PolypDetectionModel(ObjectDetectionModel):
    def __init__(self, **kwargs):
        if len(kwargs) is 0:
            kwargs = config.configuration
        ObjectDetectionModel.__init__(self, **kwargs)
        self.train_dataset, self.valid_dataset = None, None
        self.model = None

    def polyp_image_input_layer(self, name=None):
        return Input([self.size[0], self.size[1], self.n_channels], name=name)

    def polyp_image_output_layer(self, inputs):
        output = Conv2D(filters=self.n_boxes*5, kernel_size=1, strides=1, padding="same")(inputs)
        output = Lambda(lambda x: tf.reshape(x, (-1, self.n_grid, self.n_grid, self.n_boxes, 5)))(output)  #5, is objProb,cx,cy,w,h
        return output

    def reshape_output(self, outputs):
        grid = PolypDetectionModel.default_grid(self.n_grid)
        objectness_net_out, pred_box_offset_coord = tf.split(outputs, (1, 4), axis=-1)

        pred_box_normalized_coord_CxCy = (pred_box_offset_coord[:, :, :, :, 0:2] + grid) / self.n_grid
        pred_box_normalized_coord_wh = tf.square(pred_box_offset_coord[:, :, :, :, 2:])

        box_x1y1 = pred_box_normalized_coord_CxCy - pred_box_normalized_coord_wh / 2
        box_x2y2 = pred_box_normalized_coord_CxCy + pred_box_normalized_coord_wh / 2
        box_x1y1x2y2_withLUAs0_scale01 = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return box_x1y1x2y2_withLUAs0_scale01, objectness_net_out

    def reshape_output_for_prediction(self, outputs):
        b, c = [], []
        o = outputs
        num_tot_boxes = self.n_grid*self.n_grid*self.n_boxes
        bbox = tf.reshape(o[0], (1, num_tot_boxes, 1, 4))
        confidence = tf.reshape(o[1], (1, num_tot_boxes, 1))
        scores = confidence
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=bbox,
            scores=confidence,
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=0.5,
            score_threshold=0.5
        )
        return boxes, scores, classes, valid_detections

    def get_dataset(self, train_dataset_path=None, val_dataset_path=None, classes_path=None, batch_size=1):
        train_data_val_data = []
        for path in [train_dataset_path, val_dataset_path]:
            input_dataset = dataset.load_fake_dataset()
            if path:
                input_dataset = dataset.load_tfrecord_dataset(path, classes_path)
            input_dataset = input_dataset.shuffle(buffer_size=512)
            input_dataset = input_dataset.batch(batch_size)
            input_dataset = input_dataset.map(lambda x, y: (
                polyp_dataset.transform_images(x),
                polyp_dataset.transform_targets(y)))
            train_data_val_data.append(input_dataset)
        train_data_val_data[0] = train_data_val_data[0].prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.train_dataset, self.valid_dataset = train_data_val_data
        return train_data_val_data

    def squeeze_net_tiny(self, inputs=None, name=None, batch_norm_flag=False):
        if inputs is None:
            inputs = self.polyp_image_input_layer()
        x = inputs
        x = Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(x)
        if batch_norm_flag: x = BatchNormalization()(x)
        x = ReLU()(x)
        for i in range(6):
            x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
            x = half_fire(x)  # fire1
        x = half_fire(x)  # fire7
        return tf.keras.Model(inputs, x, name=name)

    def build_model(self, training=False):
        tmp = inputs = self.polyp_image_input_layer(name='Input')
        if self.model_name == "squeeze_tiny":
            tmp = self.squeeze_net_tiny(name=self.model_name)(tmp)
        else:
            logging.info("There is no model with name {a}. Please choose one of squeeze_tiny and squeeze"
                         .format(a=self.model_name))
        outputs = self.polyp_image_output_layer(tmp)
        if training: self.model = Model(inputs, outputs, name="MyModel")
        boxes_0 = Lambda(lambda x: self.reshape_output(x), name='yolo_boxes_0')(outputs)
        outputs = Lambda(lambda x: self.reshape_output_for_prediction(x), name='yolo_nms')(boxes_0)
        self.model = Model(inputs, outputs, name='MyModel')

    def get_model(self):
        if self.model is None:
            self.build_model()
            logging.info("Model was not built yet so build a new model with training=False")
        return self.model

    def load_weights(self, weight_file_path):
        if self.model is None:
            self.build_model()
        self.model.load_weights(weight_file_path).expect_partial()
        logging.info('weights loaded')


if __name__ == '__main__':
    my_model = PolypDetectionModel(**config.configuration)
    my_model.get_model().summary()
