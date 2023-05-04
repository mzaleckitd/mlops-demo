import pathlib
import shutil

from typing import Union
import numpy as np

from tqdm import tqdm

from mlutils.tools.labels.loader import load_mask_labels
from mlutils.tools.labels.converter import convert_labels_to_mask


def train_valid_test_split(labels, train_frac: float = 0.7, valid_frac: float = 0.85, seed: int = 42):
    keys = np.array(list(labels.keys()))
    np.random.seed(seed)
    np.random.shuffle(keys)
    train_idx, valid_idx = int(train_frac * keys.shape[0]), int(valid_frac * keys.shape[0])
    train, valid, test = keys[:train_idx], keys[train_idx:valid_idx], keys[valid_idx:]

    return {
        'train': {kk: labels[kk] for kk in train},
        'valid': {kk: labels[kk] for kk in valid},
        'test':  {kk: labels[kk] for kk in test},
    }


def copy_split_data(input_dirpath: Union[str, pathlib.Path], output_dirpath: Union[str, pathlib.Path]):
    input_dirpath = pathlib.Path(input_dirpath)
    output_dirpath = pathlib.Path(output_dirpath)
    if output_dirpath.exists():
        shutil.rmtree(output_dirpath)

    lbl_filepaths = sorted(input_dirpath.glob('**/*.json'))
    labels = dict()

    for lbl_filepath in lbl_filepaths:
        lbl = load_mask_labels(lbl_filepath)
        labels |= lbl

    output_paths = {
        'images': dict(),
        'masks': dict(),
    }
    pbar = tqdm(train_valid_test_split(labels).items(), desc='Coping and splitting data')
    for lbl_type, lbl in pbar:
        for ii, (image_filepath, mask) in enumerate(lbl.items()):
            out_image = output_dirpath / 'images' / lbl_type / 'img'
            if not out_image.exists():
                out_image.mkdir(parents=True)
            image_filepath_new = out_image / f'{str(ii).zfill(6)}.jpg'
            shutil.copy(image_filepath, image_filepath_new)
            output_paths['images'][lbl_type] = out_image

            out_mask = output_dirpath / 'masks' / lbl_type / 'img'
            if not out_mask.exists():
                out_mask.mkdir(parents=True)
            mask_pil = convert_labels_to_mask(mask, line_width=6)
            mask_pil.save(out_mask / f'{str(ii).zfill(6)}.jpg')
            output_paths['masks'][lbl_type] = out_mask

    return output_paths


if __name__ == '__main__':
    pass
