import pathlib

import mlflow
import mlflow.entities
from typing import Optional, Union
import yaml


def read_yaml_config(config_filepath: Union[str, pathlib.Path]):
    with open(config_filepath) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def register_experiment(experiment_name: str) -> Optional[mlflow.entities.Experiment]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        _ = mlflow.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)

    return experiment


if __name__ == '__main__':
    pass
