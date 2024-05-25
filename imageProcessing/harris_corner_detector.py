import matplotlib.pyplot
import numpy as np
from filters import sobel_filter, gaussian_filter
from feature_functions import (
    non_max_suppression,
    simple_thresholding,
    gradual_cornerness_thresholding,
)

import matplotlib

CORNERNESS_THRESHOLD = 1e9  # set it here like this so we can easily change it
NMS_WINDOW_SIZE = 5


def get_harris_corners(
    img,
    alpha=0.04,
    blur_kernal_size=3,
    nms_size=NMS_WINDOW_SIZE,
    num_of_corners=1000,
    threshold=CORNERNESS_THRESHOLD,
    adaptive_threshold=False,
):
    """Takes a blurred 8-bit grey scale image as an input and returns a list of corner points in the form of tuples (x, y)"""
    img = np.array(img).astype(np.float32)

    # Calculate Ix and Iy (sobel filter over I'm) - Stephen
    Ix, Iy = sobel_filter(img)

    # Calculate Ix^2, Iy^2 and IxIy (these are element wise multiplications not mat mult) - Stephen
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = np.multiply(Ix, Iy)

    # Gaussian blur on Ix^2, Iy^2 and IxIy - Kelham
    Ix2 = gaussian_filter(Ix2, blur_kernal_size)
    Iy2 = gaussian_filter(Iy2, blur_kernal_size)
    IxIy = gaussian_filter(IxIy, blur_kernal_size)

    # Computer corner score of every pixel - Kelham
    cornerness_score = (
        np.multiply(Ix2, Iy2) - np.square(IxIy) - alpha * np.square(Ix2 + Iy2)
    )

    # Threshold cornerness score - Zak
    if adaptive_threshold:
        thresholded_array = gradual_cornerness_thresholding(cornerness_score)
    else:
        thresholded_array = simple_thresholding(cornerness_score, threshold)

    # Perform non-maximal supression - Zak
    suppressed_array = non_max_suppression(thresholded_array, nms_size)

    # Put remaining corner points into a list in the form (row, col) - Zak
    score_feature_list = []  # optional list with the score included
    score_feature_list = [
        (score, coords[1], coords[0])
        for coords, score in list(np.ndenumerate(suppressed_array))
        if score > 0
    ]
    score_feature_list.sort(
        reverse=True, key=lambda x: x[0]
    )  # Lambda function explicitly tells it to sort on the first element even though I think it does this automatically

    xy_feature_list = [(row, col) for (_, row, col) in score_feature_list]

    return xy_feature_list[:num_of_corners]
