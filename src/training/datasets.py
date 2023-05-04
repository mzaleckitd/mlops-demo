import pathlib

import numpy as np
import tensorflow as tf
from typing import Union

import pandas as pd


def create_datagen(images_dirpath, masks_dirpath, target_size: tuple = (224, 224), batch_size: int = 32,
                   color_mode: dict = None, rescale: bool = True) -> ...:

    if color_mode is None:
        color_mode = dict(image='rgb', mask='grayscale')

    generator_images = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255. if rescale else None
    ).flow_from_directory(
        directory=images_dirpath,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode['image'],
        class_mode=None,
        seed=42,
    )

    generator_masks = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255. if rescale else None
    ).flow_from_directory(
        directory=masks_dirpath,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode['mask'],
        class_mode=None,
        seed=42,
    )

    return zip(generator_images, generator_masks)


class ImageSegmentationDataGen(tf.keras.utils.Sequence):

    def __init__(self, images_dirpath: Union[str, pathlib.Path], masks_dirpath: Union[str, pathlib.Path],
                 batch_size: int = 32, image_shape=(224, 224, 3), masks_shape=(224, 224, 1), normalize: bool = True,
                 shuffle: bool = True, file_extensions: Union[tuple, list] = ('jpg', 'png')):

        print('Creating a new ImageSegmentationDataGen object.')
        # -------- paths and files --------
        self.images_dirpath = pathlib.Path(images_dirpath)
        self.masks_dirpath = pathlib.Path(masks_dirpath)
        self.file_extensions = file_extensions
        print('Image dir:', self.images_dirpath)
        print('Masks dir:', self.masks_dirpath)
        print(f'Files extensions: {self.file_extensions}')

        # -------- main dataframe ---------
        self.df = self.__prepare_files_df()
        print(f'Find {self.df.shape[0]} pairs of images and masks.')

        # -------- data processing ---------
        self.batch_size = batch_size
        print(f'Set batch_size == {self.batch_size}. Establish {self.__len__()} steps per epoch.')

        self.shuffle = shuffle
        if self.shuffle:
            print('Turn on shuffling a data.')
            self.on_epoch_end()

        # -------- image processing ---------
        self.shapes = {'image': image_shape, 'mask': masks_shape}
        self.normalize = normalize
        print()

    def __prepare_files_df(self) -> pd.DataFrame:
        images = sum([sorted(self.images_dirpath.glob(f'**/*.{fe}')) for fe in self.file_extensions], [])
        masks = sum([sorted(self.masks_dirpath.glob(f'**/*.{fe}')) for fe in self.file_extensions], [])

        df = pd.DataFrame({
            'image_filepath': images, 'mask_filepath': masks
        })

        df['image_filename'] = df['image_filepath'].apply(lambda x: x.name)
        df['mask_filename'] = df['mask_filepath'].apply(lambda x: x.name)

        df = df[df['image_filename'] == df['mask_filename']].reset_index(drop=True)

        return df

    def __get_picture_from_path(self, path: Union[str, pathlib.Path], data_type: str = 'image'):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (self.shapes[data_type][0], self.shapes[data_type][1])).numpy()
        if self.shapes[data_type][2] == 1:
            image_arr = tf.image.rgb_to_grayscale(image_arr)
        if self.normalize:
            image_arr = image_arr / 255.

        return image_arr

    def __get_data(self, batches: pd.DataFrame) -> tuple[np.array, np.array]:
        images_batch = batches['image_filepath']
        masks_batch = batches['mask_filepath']

        images = np.asarray([self.__get_picture_from_path(xx, 'image') for xx in images_batch])
        masks = np.asarray([self.__get_picture_from_path(xx, 'mask') for xx in masks_batch])

        return images, masks

    def __getitem__(self, index: int):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self.__get_data(batches)
        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            self.df.sample(frac=1).reset_index(drop=True, inplace=True)

    def __len__(self):
        return self.df.shape[0] // self.batch_sizepip


if __name__ == '__main__':
    pass
