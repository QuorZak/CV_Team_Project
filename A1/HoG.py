import numpy as np
from imageProcessing.feature_functions import get_search_area, l1_distance
from filters import gaussian_filter, sobel_filter

from matplotlib import pyplot
from matplotlib.patches import Circle

from filters import gaussian_filter
from timeit import default_timer as timer


def compute_hog_cell(n_orientations, magnitudes, orientations):
    bin_width = int(180 / n_orientations)
    hist = np.zeros(n_orientations)
    rav_orientations = orientations.ravel()
    rav_magnitudes = magnitudes.ravel()
    assert len(rav_orientations) == len(rav_magnitudes)
    "Somehow the patches arent the same size"
    for i in range(len(rav_orientations)):
        orientation = rav_orientations[i]
        magnitude = rav_magnitudes[i]
        lower_bin_idx = min(int(orientation / bin_width), n_orientations - 1)
        upper_bin_idx = (lower_bin_idx + 1) % n_orientations
        proportion_upper = (orientation % bin_width) / bin_width
        hist[upper_bin_idx] = magnitude * proportion_upper
        hist[lower_bin_idx] = magnitude * (1 - proportion_upper)

    return hist


def get_hog_descriptor(magnitudes, orientations, col, row, patch_size=36, cell_size=4):
    mag_patch = get_search_area(magnitudes, col, row, patch_size)
    ori_patch = get_search_area(orientations, col, row, patch_size)
    cells = []
    for y in range(0, patch_size, cell_size):
        for x in range(0, patch_size, cell_size):
            mag_cell = mag_patch[
                y : y + cell_size, x : x + cell_size
            ]  # Extract cell from patch
            ori_cell = ori_patch[y : y + cell_size, x : x + cell_size]
            hog_cell = compute_hog_cell(9, mag_cell, ori_cell)
            hog_cell[:] /= np.sum(np.abs(hog_cell + 1e-10))
            cells.append(hog_cell)
    hog_descriptor = np.stack(cells)
    return np.sum(hog_descriptor, axis=0)


def get_hog_histograms(px_array, corners, window=9, cell_size=4, n_orientations=9):
    image = np.float32(px_array) / 255
    gx, gy = sobel_filter(image)
    magnitudes = np.hypot(gx, gy)
    orientations = np.rad2deg(np.arctan2(gy, gx)) % 180

    histograms = np.zeros((image.shape[0], image.shape[1], n_orientations))

    for col, row in corners:
        histograms[row, col, :] = get_hog_descriptor(
            magnitudes,
            orientations,
            col,
            row,
            patch_size=round(window * cell_size),
            cell_size=cell_size,
        )
    return histograms


def hog_matching(
    left_array,
    right_array,
    left_im_corners,
    right_im_corners,
    window=9,
    cell_size=4,
    n_orientations=9,
    ratio=0.9,
):
    left_histograms = get_hog_histograms(
        left_array,
        left_im_corners,
        window=window,
        cell_size=cell_size,
        n_orientations=n_orientations,
        type="hog",
    )
    right_histograms = get_hog_histograms(
        right_array,
        right_im_corners,
        window=window,
        cell_size=cell_size,
        n_orientations=n_orientations,
        type="hog",
    )

    corner_pairs = []
    dup_right_corners = list(right_im_corners)
    for col, row in left_im_corners:
        target_hist = left_histograms[row, col, :]
        closest_match = None
        closest_dist = 1e10
        second_closest_dist = 1e10
        closest_i = None
        for i, (col_r, row_r) in enumerate(dup_right_corners):
            right_hist = right_histograms[row_r, col_r, :]
            dist = l1_distance(target_hist, right_hist)
            if dist < closest_dist:
                second_closest_dist = closest_dist
                closest_dist = dist
                closest_match = (col_r, row_r)
                closest_i = i
            elif dist < second_closest_dist:
                second_closest_dist = dist
        if closest_dist / second_closest_dist < ratio:
            corner_pairs.append([(col, row), closest_match])
            dup_right_corners.pop(closest_i)

    return corner_pairs
