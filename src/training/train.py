import os
import pathlib
from typing import Union

import mlflow
import tensorflow as tf
import yaml

from datasets import create_datagen
from metrics import iou_score
from model import load_model
from tools import register_experiment, read_yaml_config


def main(config_filepath: Union[str, pathlib.Path]):

    # config
    config = read_yaml_config(config_filepath)

    # set mlflow
    cwd = pathlib.Path(os.getcwd())
    mlflow.set_tracking_uri(f"file://{cwd / config['mlflow']['dirpath']}")
    mlflow.autolog()
    experiment = register_experiment(config['mlflow']['experiments']['train']['name'])

    # save run to proper mlflow
    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # read training config
        config_train = read_yaml_config(cwd / config['mlflow']['experiments']['train']['config_file'])

        # create dataset generator
        data_generators = dict()
        for dataset_type in ['train', 'valid', 'test']:
            datagen = create_datagen(
                cwd / config['datasets'][dataset_type]['images'],
                cwd / config['datasets'][dataset_type]['masks'],
                batch_size=int(config_train['training']['batch_size']),
            )
            data_generators[dataset_type] = datagen

        # load model
        model_filepath = cwd / config_train['model']['model_path']
        model = load_model(model_filepath)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=config_train['training']['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy", iou_score]
        )

        # training
        model.fit(
            data_generators['train'],
            validation_data=data_generators['valid'],
            epochs=config_train['training']['epochs'],
            steps_per_epoch=100,
            validation_steps=10,
        )

        # testing
        _, acc, iou = model.evaluate(data_generators['test'], steps=10)
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('test_iou_score', iou)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_filepath', help='Filepath to configuration YAML', default='config/mlflow.yaml')
    args = parser.parse_args()

    config_file = pathlib.Path(args.config_filepath)
    print(config_file.absolute())
    main(config_file)

