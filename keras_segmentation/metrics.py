import numpy as np
from keras.metrics import MeanIoU
import tensorflow as tf
EPS = 1e-12


def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise
def mean_iou(y_true, y_pred):
    # Convert predictions to integer class values
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate Mean IoU
    mean_iou = MeanIoU(num_classes=n_classes)
    mean_iou.update_state(y_true, y_pred)
    return mean_iou.result().numpy()

def fw_iou(y_true, y_pred):
    # Convert predictions to integer class values
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate frequency weights
    class_counts = tf.reduce_sum(y_true, axis=(0, 1))
    freq_weights = 1.0 / (class_counts + 1)

    # Calculate FW IoU
    intersection = tf.reduce_sum(y_true * y_pred, axis=(0, 1))
    union = tf.reduce_sum(y_true + y_pred, axis=(0, 1)) - intersection
    fw_iou = tf.reduce_sum(freq_weights * intersection) / tf.reduce_sum(freq_weights * union)
    return fw_iou
