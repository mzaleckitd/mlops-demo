import pathlib
from typing import Union

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def load_images_masks(images_dirpath: Union[str, pathlib.Path], masks_dirpath: Union[str, pathlib.Path],
                      file_extension: str = 'jpg') -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    images = sorted(images_dirpath.glob(f'**/*.{file_extension}'))
    masks = sorted(masks_dirpath.glob(f'**/*.{file_extension}'))

    assert len(images) == len(masks), "There is different number of images and mask, please check input folders or " \
                                      "previous step in data processing"

    return images, masks


def perform_augmentation(image: np.array, mask: np.array, augmentation_function, image_dirpath, mask_dirpath, iterator):
    transformed = augmentation_function(image=image, mask=mask)
    cv2.imwrite(str(image_dirpath / f'{str(iterator).zfill(6)}.jpg'),
                cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_dirpath / f'{str(iterator).zfill(6)}.jpg'),
                cv2.cvtColor(transformed['mask'], cv2.COLOR_RGB2BGR))


def flip_images(images_dirpath: Union[str, pathlib.Path], masks_dirpath: Union[str, pathlib.Path],
                config: dict, file_extension: str = 'jpg'):

    if images_dirpath is None or masks_dirpath is None:
        print('Skipping flipping')
        return

    images_dirpath, masks_dirpath = pathlib.Path(images_dirpath), pathlib.Path(masks_dirpath)
    images, masks = load_images_masks(images_dirpath, masks_dirpath, file_extension)

    flip_pipeline = list()
    if config['flip_left_right']['apply']:
        flip_pipeline.append(A.from_dict(config['flip_left_right']))

    # flip top bottom
    if config['flip_top_bottom']['apply']:
        flip_pipeline.append(A.from_dict(config['flip_top_bottom']))

    image_iterator = len(images)
    pbat = tqdm(images, desc='Flipping images')
    for image_filepath in pbat:
        mask_filepath = masks_dirpath / image_filepath.name

        image = cv2.cvtColor(cv2.imread(str(image_filepath)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_filepath))

        for flip_func in flip_pipeline:
            perform_augmentation(image, mask, flip_func, image_filepath.parent, mask_filepath.parent, image_iterator)
            image_iterator += 1


def augment_images(images_dirpath: Union[str, pathlib.Path], masks_dirpath: Union[str, pathlib.Path],
                   config: dict, file_extension: str = 'jpg'):
    if images_dirpath is None or masks_dirpath is None:
        print('Skipping augmenting')
        return

    images_dirpath, masks_dirpath = pathlib.Path(images_dirpath), pathlib.Path(masks_dirpath)
    images, masks = load_images_masks(images_dirpath, masks_dirpath, file_extension)

    augmentations = [A.from_dict(x) for x in config['augmentations']]

    image_iterator = len(images)
    select_aug_functions = config['augmentations_per_image']
    pbat = tqdm(images, desc='Augmenting data')
    for image_filepath in pbat:
        mask_filepath = masks_dirpath / image_filepath.name

        image = cv2.cvtColor(cv2.imread(str(image_filepath)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_filepath))

        for func_idx in np.random.choice(np.arange(len(augmentations)), select_aug_functions):
            func = augmentations[func_idx]
            perform_augmentation(image, mask, func, image_filepath.parent, mask_filepath.parent, image_iterator)
            image_iterator += 1


if __name__ == '__main__':
    pass
