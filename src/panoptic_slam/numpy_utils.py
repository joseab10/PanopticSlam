import numpy as np


def join_arrays(arrays):
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)

    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)

    dtype = sum((a.dtype.descr for a in arrays), [])

    return joint.ravel().view(dtype)
