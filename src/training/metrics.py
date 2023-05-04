import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5):
    y_true, y_pred = y_true.numpy(), y_pred.numpy()

    y_pred[y_pred < threshold] = 0
    y_pred[y_pred >= threshold] = 1

    intersection = np.logical_and(y_pred, y_true)
    union = np.logical_or(y_pred, y_true)
    if np.nansum(union) == 0:
        return 0
    iou = np.nansum(intersection) / np.nansum(union)

    return iou


if __name__ == '__main__':
    pass
