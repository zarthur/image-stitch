"""combine multiple images into one using a sliding window of vertical strips"""
import glob
import os
import sys

from typing import List, Tuple

import numpy as np

from PIL import Image, ImageFilter


def get_files(directory: str, name_filter: str = "*.jpg") -> List:
    """
    List files in specified directory matching filter

    :param directory: directory in which to search
    :param name_filter: wildcard filter for filenames
    :return: list of strings corresponding to file paths
    """
    path_str = os.path.join(directory, name_filter)
    return sorted(glob.glob(path_str))


def read_strip(image_path: str, start: int = None, end: int = None) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read vertical strip from an image

    :param image_path: path to image file
    :param start: column from which to begin reading
    :param end: column at which reading will end (exclusive)
    :return: three Numpy arrays corresponding to RGB data from the image
    """
    with Image.open(str(image_path)) as image_file:
        start = start if start is not None else 0
        end = end if end is not None else image_file.width
        array = np.array(image_file)
        data_r, data_g, data_b = [array[:, :, x]
                                  for x in range(array.shape[-1])]
        return (data_r[:, start:end],
                data_g[:, start:end],
                data_b[:, start:end])


def create_strips(file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
    """
    Aggregate vertical-strip data from source images

    :param file_paths: list of paths to source images, read in order
    :return: three Numpy arrays corresponding to RGB data for combined image
    """
    first_file, *_ = file_paths
    first_image_data, *_ = read_strip(first_file)
    _, image_width = first_image_data.shape
    final_r = np.zeros_like(first_image_data)
    final_g = np.zeros_like(first_image_data)
    final_b = np.zeros_like(first_image_data)

    number_of_files = len(file_paths)
    strip_width = image_width // number_of_files

    for index, file_path in enumerate(file_paths):
        _, filename = os.path.split(file_path)
        start = index * strip_width
        end = (image_width if index == (number_of_files - 1)
               else (index + 1) * strip_width)
        new_r, new_g, new_b = read_strip(file_path, start, end)
        final_r[:, start:end] = new_r
        final_g[:, start:end] = new_g
        final_b[:, start:end] = new_b

        mean_r = int(new_r.mean())
        mean_g = int(new_g.mean())
        mean_b = int(new_b.mean())
        message = ("Processed image {0} of {1}: {2}, "
                   "{3}px to {4}px, mean RGB: ({5}, {6}, {7})"
                   .format(index + 1, number_of_files, filename,
                           start, end - 1, mean_r, mean_g, mean_b))
        print(message)

    return final_r, final_g, final_b


def write_image(source_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                file_path: str) -> None:
    """
    Write image data to a file

    :param source_data: Numpy arrays containing RGB data
    :param file_path: path to which output will be written
    """
    image = Image.fromarray(np.dstack(source_data))
    image.save(file_path, optimize=True)


def main(directory: str, output_path: str) -> None:
    """
    Create an image consisting of components of other images

    :param directory: directory of source images
    :param output_path: path to which new file will be written
    """
    files = get_files(directory)
    combined = create_strips(files)
    write_image(combined, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage {0} INPUT_DIRECTORY OUTPUT_PATH".format(sys.argv[0]))
    else:
        directory, filename = sys.argv[1:]
        main(directory, filename)
