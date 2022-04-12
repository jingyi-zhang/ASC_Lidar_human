import os
import shutil


def get_paths_by_suffix(dirname, suffix):
    filenames = list(filter(lambda x: x.endswith(suffix), os.listdir(dirname)))
    assert len(filenames) > 0
    return [os.path.join(dirname, filename) for filename in filenames]


def get_one_path_by_suffix(dirname, suffix):
    return get_paths_by_suffix(dirname, suffix)[0]


def get_index(filename):
    basename = os.path.basename(filename)
    return int(os.path.splitext(basename)[0])


def is_index_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0].isdigit()


def clear_folder(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)


def is_index_continuous(dirname):
    filenames = get_sorted_filenames_by_index(dirname)
    n_filenames = len(filenames)
    first = get_index(filenames[0])
    last = get_index(filenames[-1])
    return last - first + 1 == n_filenames


def get_sorted_filenames_by_index(dirname, isabs=True):
    filenames = sorted(
        list(filter(is_index_filename, os.listdir(dirname))), key=lambda x: get_index(x))
    if isabs:
        filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames


def remove_unnecessary_frames(dirname, indexes):
    filenames = os.listdir(dirname)
    assert len(filenames) > 0
    for filename in filenames:
        if is_index_filename(filename) and get_index(filename) not in indexes:
            os.remove(os.path.join(dirname, filename))
