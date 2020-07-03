import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from Models.models import PolypDetectionModel

flags.DEFINE_string('model_name', 'squeeze_tiny', 'currently not used')
# flags.DEFINE_string('tf_weight_input', './results/checkpoints/polyp_train_500.tf', 'currently not used')
flags.DEFINE_string('tf_weight_input', '../results/checkpoints/polyp_train_1.tf', 'path to the .tf weight file')
flags.DEFINE_string('pb_weight_output', '../results/pbModel', 'path to the output .pb file')


def convert(model_name, tf_weights_input, pb_weights_output):
    model = PolypDetectionModel(model_name=model_name)
    model.build_model(True)
    model.load_weights(tf_weights_input)
    tf_model = model.get_model()
    tf.saved_model.save(tf_model, pb_weights_output)


def main(_args):
    convert(FLAGS.model_name, FLAGS.tf_weight_input, FLAGS.pb_weight_output)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
