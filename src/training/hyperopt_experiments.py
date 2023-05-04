import datetime
import os
import pathlib

import mlflow
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from datasets import ImageSegmentationDataGen
from metrics import iou_score
from model import load_model
from tools import register_experiment, read_yaml_config

import numpy as np

tf.config.run_functions_eagerly(True)


def train(params):
    with mlflow.start_run(
        run_name=params.get('run_name', None),
        experiment_id=params['experiment_id'],
        description="child",
        nested=True,
    ):
        data_generators = dict()
        for dataset_type in ['train', 'valid', 'test']:
            datagen = ImageSegmentationDataGen(
                config['datasets'][dataset_type]['images'],
                config['datasets'][dataset_type]['masks'],
                batch_size=int(params['batch_size']),
            )
            data_generators[dataset_type] = datagen

        model_filepath = '/Users/marek.zalecki/Projects/ml-serco/models/edge/base/model_edge_224x224'
        model = load_model(model_filepath)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy", iou_score]
        )

        model.fit(
            data_generators['train'],
            validation_data=data_generators['valid'],
            epochs=int(params['epochs']),
            steps_per_epoch=10,
            validation_steps=5,
        )

        # testing
        _, acc, iou = model.evaluate(data_generators['test'], steps=5)
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('test_iou_score', iou)

    return {'loss': -iou, 'status': STATUS_OK}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_filepath', help='Filepath to configuration YAML', default='config/mlflow.yaml')
    args = parser.parse_args()

    config_filepath = pathlib.Path(args.config_filepath)
    config = read_yaml_config(config_filepath)

    cwd = pathlib.Path(os.getcwd())
    mlflow.set_tracking_uri(f"file://{cwd / config['mlflow']['dirpath']}")
    experiment = register_experiment(config['mlflow']['experiments']['hyperopt']['name'])

    mlflow.autolog(log_models=False)
    with mlflow.start_run(
            run_name=datetime.datetime.now().isoformat(),
            experiment_id=experiment.experiment_id,
            tags={"version": "v1", "priority": "P1"},
            description="parent",
    ):
        space = {
            'batch_size': hp.uniform('batch_size', 1, 128),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1)),
            'epochs': hp.uniform('epochs', 1, 32),
            'experiment_id': experiment.experiment_id,
        }

        trials = Trials()

        best = fmin(train, space, algo=tpe.suggest, max_evals=10, trials=trials)
        print('best: ', best)

        mlflow.autolog(log_models=True)
        train(best | {'experiment_id': experiment.experiment_id, 'run_name': 'best-run-model-saved'})
