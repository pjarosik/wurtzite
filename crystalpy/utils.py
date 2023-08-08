import numpy as np


def is_vector(array: np.ndarray):
    """
    Returns true if the given array is a vector.
    """
    return len(array.squeeze().shape) == 1


def point_plane_distance(a, b, c, d, points):
    """
    Plane: ax + by + cz + d = 0

    :param points: points for which the distance should be calculated,
        (n_points, 3)
    """
    denominator = np.sqrt(a**2 + b**2 + c**2)
    n = np.asarray([a, b, c]).reshape(-1, 1)
    numerator = np.abs(np.dot(points, n) + d).squeeze()
    return numerator/denominator
