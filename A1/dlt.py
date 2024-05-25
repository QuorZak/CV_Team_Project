import math
import numpy as np


def project_point(point: tuple, H: np.ndarray) -> tuple:
    homo_point = np.asarray([*point, 1])
    projected = H @ homo_point
    cartesian = projected / projected[2]
    cart_point = np.round(cartesian[:2], 2)
    return tuple(cart_point)


def interpolate(mat: np.ndarray, projection: tuple) -> int:
    excess_x = projection[0] % 1
    excess_y = projection[1] % 1
    value = excess_x * excess_y * mat[tuple(map(math.ceil, projection))]
    value += (
        excess_x * (1 - excess_y) * mat[(math.ceil(projection[0]), int(projection[1]))]
    )
    value += (
        excess_y * (1 - excess_x) * mat[(int(projection[0]), math.ceil(projection[1]))]
    )
    value += (1 - excess_x) * (1 - excess_y) * mat[tuple(map(int, projection))]
    return round(value)


def blend(mat1: np.ndarray, mat2: np.ndarray, point1: tuple, projection: tuple) -> int:
    value = interpolate(mat2, projection)
    return round((mat1[point1] + value) / 2)


def contains(shape: tuple, point: tuple) -> bool:
    return (0 <= point[0] < shape[0]) and (0 <= point[1] < shape[1])


def dlt(matches: list[list[tuple]]) -> np.ndarray:
    """
    DLT algorithm to compute homography between two images based on a set of matches, and return the stitched image

    :param im1: an image you want to stitch
    :param im2: another image you want to stitch
    :param matches: a list of lists, where each sublist contains two tuples representing a point in im1 and a point in im2 which match
    :return: homography to project im1 onto im2
    """
    assert len(matches) >= 4
    "There needs to be at least 4 matches!!!"

    left = [match[0] for match in matches]
    right = [match[1] for match in matches]

    A = np.zeros((2 * len(right), 9))
    for i in range(len(right)):
        idx = i * 2
        A[idx, 3:6] = [*left[i], 1]  # indices 3:6 = x,y,1
        A[idx + 1, 0:3] = [*left[i], 1]  # indices 0:3 = x,y,1
        A[idx, 6:9] = [
            right[i][1] * left[i][0] * -1,
            right[i][1] * left[i][1] * -1,
            right[i][1] * -1,
        ]
        A[idx + 1, 6:9] = [
            right[i][0] * left[i][0] * -1,
            right[i][0] * left[i][1] * -1,
            right[i][0] * -1,
        ]

    U, D, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H


def stitch(im1: np.ndarray, im2: np.ndarray, H: np.ndarray):
    """
    Image stitching method using two images, and a homography H which projects im1 onto im2.

    :param im1: an image you want to stitch
    :param im2: another image you want to stitch
    :return: Stitched image
    """
    canvas = np.zeros((im1.shape[0], im2.shape[1] + im1.shape[1]))
    canvas_points = zip(
        np.repeat(list(range(canvas.shape[0])), canvas.shape[1]),
        list(range(canvas.shape[1])) * canvas.shape[0],
    )
    for point in canvas_points:
        projection = project_point(point, H)
        if contains(im1.shape, point) and not contains(im2.shape, projection):
            canvas[point] = im1[point]
        elif contains(im1.shape, point) and contains(im2.shape, projection):
            canvas[point] = blend(im1, im2, point, tuple(map(int, projection)))
        elif not contains(im1.shape, point) and contains(im2.shape, projection):
            canvas[point] = im2[tuple(map(int, projection))]

    return canvas
