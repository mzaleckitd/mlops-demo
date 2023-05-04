import pathlib

import tensorflow as tf
from typing import Union


def load_model(model_path: Union[str, pathlib.Path], verbose: bool = True) -> tf.keras.models.Model:
    model = tf.keras.models.load_model(model_path)
    if verbose:
        model.summary()

    return model


if __name__ == '__main__':
    pass
