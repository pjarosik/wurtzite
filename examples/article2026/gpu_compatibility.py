import os
import numpy as np
xp = np

USE_GPU = False

if os.environ.get("WURTZITE_USE_GPU"):
    print("USING GPU")
    USE_GPU = True
    import cupy as cp
    xp = cp
else:
    print("USING CPU")


## --- GPU COMPATIBILTIY FUNCTIONS
def _d2h_gpu(array):
    """
    Transfers the array from GPU to host.
    If the array is already located on the host machine, returns the array
    unmodified.
    """
    if isinstance(array, cp.ndarray):
        return array.get()


def _d2h_cpu(array):
    """
    This method assumes that that array is already located on GPU.
    Simply returns the numpy array unmodified.
    """
    return np.asarray(array)


def _h2d_gpu(array):
    """Moves the array from the host PC memory to GPU device memory."""
    return cp.asarray(array)


def _h2d_cpu(array):
    """Returns np ndarray for the given np.ndarray/list."""
    return np.asarray(array)


# By default use CPU functions
d2h = _d2h_cpu
h2d = _h2d_cpu

if USE_GPU:
    d2h = _d2h_gpu
    h2d = _h2d_gpu