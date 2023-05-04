import pathlib
import yaml
import numpy as np

from utils import copy_split_data
from augmentation import flip_images, augment_images


if __name__ == '__main__':
    np.random.seed(42)

    config_filepath = pathlib.Path('config/prepare_data.yaml')
    print(config_filepath.absolute())

    with open(config_filepath) as yaml_file:
        config = yaml.safe_load(yaml_file)

    input_dirpath = pathlib.Path(config['copying']['input_dirpath']).absolute()
    output_dirpath = pathlib.Path(config['copying']['output_dirpath']).absolute()

    for data_type in ['edge', 'line']:
        print('='*64)
        print(f'Preparing data for {data_type} detection')
        input_dir = input_dirpath / data_type
        output_dir = output_dirpath / data_type

        print('Input dir:', input_dir)
        print('Output dir:', output_dir)
        output_dirpaths = copy_split_data(input_dirpath=input_dir, output_dirpath=output_dir)

        print(output_dirpaths)

        flip_images(
            images_dirpath=output_dirpaths['images'].get('train', None),
            masks_dirpath=output_dirpaths['masks'].get('train', None),
            config=config['flipping']
        )

        augment_images(
            images_dirpath=output_dirpaths['images'].get('train', None),
            masks_dirpath=output_dirpaths['masks'].get('train', None),
            config=config['augmenting']
        )
        print()
