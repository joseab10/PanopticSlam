import errno
from os import path, makedirs


def parse_path(in_path):
    """
    Expands user and variable in path. Used for argument parsing.

    :param in_path: (str) String representing a path in the operating system.

    :return: (path) Path with user and variables expanded.
    """

    return path.expandvars(path.expanduser(in_path))


def mkdir(in_path):
    """
    Make directories in a way similar to "mkdir -p".

    :param in_path: (str) Path of director(y/ies) to be created.

    :return: None
    """
    try:
        makedirs(in_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.isdir(in_path):
            pass
        else:
            raise
