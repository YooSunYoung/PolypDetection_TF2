from absl import app, flags, logging
from absl.flags import FLAGS

from Training import training_recipe
from Utils.common_functions import make_and_clean_dir

from Models.models import PolypDetectionModel, get_loss

import tensorflow as tf


def set_up_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def set_up_directories():
    # returns the train log directory
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'results/logs/' + current_time
    make_and_clean_dir(train_log_dir)
    make_and_clean_dir("../results/checkpoints")
    make_and_clean_dir("../results/TfliteModel")
    return train_log_dir


def main(_argv):
    training_recipe.set_settings(flags)
    set_up_gpu()
    train_log_dir = set_up_directories()
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_summary_writer.set_as_default()

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    model = PolypDetectionModel()
    training_model = model.get_model(True)

    train_dataset, val_dataset = model.get_dataset()  # prepare for the train dataset and validation dataset

    step=0
    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                tf.summary.experimental.set_step(step)
                step = step + 1

                with tf.GradientTape() as tape:
                    outputs = training_model(images, training=True)
                    pred_loss = get_loss(labels, outputs, training=True)
                    total_loss = tf.reduce_sum(pred_loss)

                grads = tape.gradient(total_loss, training_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, training_model.trainable_variables))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = training_model(images, training=True)
                regularization_loss = tf.reduce_sum(training_model.losses)
                pred_loss = get_loss(labels, outputs, training=False)
                total_loss = tf.reduce_sum(pred_loss)
                avg_val_loss.update_state(total_loss)

            logging.info("epoch:{}, average train/valid loss per batch: {}/{}".format(epoch,
                                                                                      avg_loss.result().numpy(),
                                                                                      avg_val_loss.result().numpy()))
            tf.summary.scalar("average train loss per batch", avg_loss.result().numpy())
            tf.summary.scalar("average valid loss per batch", avg_val_loss.result().numpy())
            avg_loss.reset_states()
            avg_val_loss.reset_states()

            if epoch % FLAGS.save_points == 0 or epoch == FLAGS.epochs:
                logging.info("-----------------------------------------------------------------------------")
                training_model.save_weights('results/checkpoints/yolov3_train_{}.tf'.format(epoch))
                converter = tf.lite.TFLiteConverter.from_keras_model(training_model)
                tflite_model = converter.convert()
                open("results/TfliteModel/model{}.tflite".format(epoch), "wb").write(tflite_model)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
