def set_settings(flags):
    flags.DEFINE_string('dataset', 'data/polyp_train001bmp.tfrecord', 'path to dataset')
    flags.DEFINE_string('val_dataset', 'data/polyp_train001bmp.tfrecord', 'path to validation dataset')
    flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
    flags.DEFINE_boolean('LoadWeights', False, 'yolov3 or yolov3-tiny')
    flags.DEFINE_string('weights', './checkpointsTemp/yolov3_train_180.tf',
                        'path to weights file')
    flags.DEFINE_string('checkpoint_dir_path', './results/checkpoints', 'path to checkpoint directory')
    flags.DEFINE_string('log_dir_path', './results/', 'path to log directory')
    flags.DEFINE_string('tflite_model_dir_path', './results/TfliteModel', 'path to tflite model directory')
    flags.DEFINE_string('classes', './data/polyp.names', 'path to classes file')
    flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_fit', 'eager_tf'],
                      'fit: model.fit, '
                      'eager_fit: model.fit(run_eagerly=True), '
                      'eager_tf: custom GradientTape')
    flags.DEFINE_enum('transfer', 'none',
                      ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                      'none: Training from scratch, '
                      'darknet: Transfer darknet, '
                      'no_output: Transfer all but output, '
                      'frozen: Transfer and freeze all, '
                      'fine_tune: Transfer all and freeze darknet only')
    flags.DEFINE_integer('epochs', 50, 'number of epochs')
    flags.DEFINE_integer('save_points', 50, 'save weights every')
    flags.DEFINE_integer('batch_size', 1, 'batch size')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
