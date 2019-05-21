import os
import argparse
import sys


def read_text(path):
    with open(path) as f:
        text = f.read().lower()
    text = text.strip()
    return text


def generate_full_path(dset_path, lst, dir):
    res_lst = []
    for l in lst:
        res_lst.append(os.path.join(dset_path, dir, "images", l))
    return res_lst


def prepareWithLabels(dset_path, recursive_dirs=False):
    """ создание листа с путями до изображений и соответствующими метками
    Args:
        dset_path: путь до датасета: изображения должны находиться в папке "images"
                   метки должны находиться в папке "labels"
        recursive_dirs: True, если датасет разбит на несколько папок (в таком случае в каждой из папок должны лежать папки
        "images" и "labels"
    Returns:
        Лист с путями до изображений и меток
        Example:
            /home/root/workspase/dsets/data/images/0.jpg hello
    """

    if recursive_dirs:
        imageList = []
        for dir in os.listdir(dset_path):
            filenames = list(filter(lambda x: x[-4:] == '.jpg',
                                    generate_full_path(dset_path, os.listdir(os.path.join(dset_path, dir, "images")), dir)))
            imageList += filenames
    else:
        dset_path = os.path.join(dset_path, "images")
        imageList = list(filter(lambda x: x[-4:] == '.jpg',
                                os.listdir(dset_path)))
        imageList = [os.path.join(dset_path, x) for x in imageList]
    data_set = []
    for p in imageList:
        if not os.path.exists(p.replace('.jpg', '.txt').replace('images', 'labels')):
            print('{} does not exist!'.format(p.replace('.jpg', '.txt').replace('images', 'labels')))
            continue
        data_set.append((p, read_text(p.replace('.jpg', '.txt').replace('images', 'labels'))))

    return data_set


def prepareWithoutLabels(dset_path, recursive_dirs):
    """ создание листа с путями до изображений без меток
    Args:
        dset_path: путь до датасета: изображения должны находиться в этой же папке
        recursive_dirs: True, если датасет разбит на несколько папок (в таком случае в каждой из папок должны лежать изображения
    Returns:
        Лист с путями до изображений
        Example:
            /home/root/workspase/dsets/data/images/0.jpg
    """

    if recursive_dirs:
        imageList = []
        for dir in os.listdir(dset_path):
            filenames = list(filter(lambda x: x[-4:] == '.jpg',
                                    generate_full_path(dset_path, os.listdir(os.path.join(dset_path, dir)), dir)))
            imageList += filenames
    else:
        dset_path = os.path.join(dset_path)
        imageList = list(filter(lambda x: x[-4:] == '.jpg',
                                os.listdir(dset_path)))
        imageList = [os.path.join(dset_path, x) for x in imageList]

    data_set = []
    for p in imageList:
        data_set.append((p))

    return data_set


def writeList(data_set, path_to_save, with_labels, filename="sample.txt"):
    if with_labels:
        with open(os.path.join(path_to_save, filename), "w") as out:
            for path, label in data_set:
                out.writelines(path + " " + label + '\n')
    else:
        with open(os.path.join(path_to_save, filename), "w") as out:
            for path in data_set:
                out.writelines(path + '\n')

def data_file(data_dir, with_labels, recursive_dirs):
    if with_labels:
        d_list = prepareWithLabels(data_dir, recursive_dirs)
    else:
        d_list = prepareWithoutLabels(data_dir, recursive_dirs)
    writeList(d_list, data_dir, with_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default="../data/Recognition", help="Директория, в которой лежат вырезанные слова")
    parser.add_argument('--with-labels', default=False, help="True, если нужно добавить метки")
    parser.add_argument('--recursive-dirs', default=False, help="True, если данные хранятся в нескольких папках")
    parameters = parser.parse_args(sys.argv[1:])
    data_file(parameters.data_dir, parameters.with_labels, parameters.recursive_dirs)