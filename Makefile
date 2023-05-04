prepare-data:
	python3 src/preprocessing/prepare_data.py

final-training:
	python3 src/training/train.py

hyperopt-experiments:
	python3 src/training/hyperopt_experiments.py
