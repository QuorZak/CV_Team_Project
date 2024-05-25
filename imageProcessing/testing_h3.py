import numpy as np
import cv2
from A2_utilities import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

R = np.array(
    [[0.62102505, -0.78378701, -0.0024104],
    [-0.29478764, -0.2406425, 0.92476561],
    [-0.7254033, -0.5735952, -0.38049783]]
)


T = np.array([4.56695734, -8.82056786, 60.20706773])
s_x = 1.0009169904257296
f = 0.2730571111216076

WRF = np.array(
    [[6.21025047e-01, -7.83787012e-01, -2.41040463e-03, 4.56695734e00],
    [-2.94787637e-01, -2.40642502e-01, 9.24765611e-01, -8.82056786e00],
    [-7.25403302e-01, -5.73595198e-01, -3.80497830e-01, 6.02070677e01],
    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]]
)

projection = np.array(
    [[0.27305711, 0.0, 0.0, 0.0],[0.0, 0.27305711, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0]]
)
CRF = np.array(
    [[6.45752897e03, 0.00000000e00, 1.50000000e03],
    [0.00000000e00, -6.45161290e03, 1.12500000e03],
    [0.00000000e00, 0.00000000e00, 1.00000000e00]]
)

img = cv2.imread("A2_data/undistorted_H3.jpg")

# read the pixel size from json settings
settings = read_settings()
name = "H3"
dx = settings[f"{name}_pixel_size_x"]
dy = settings[f"{name}_pixel_size_y"]

# draw_world_axis(img, WRF, projection, CRF)
# cv2.imshow("draw_world_axis", img)
# cv2.waitKey(0)

points = extract_points(img)
pixels = points[:,3:]
real_points = points[:, :3]

predicted_pixels = np.apply_along_axis(estimate_2D_from_3D, 1, real_points, WRF, projection, CRF)

rmse = image_calibration_error(pixels, predicted_pixels)

plot_pixel_projections(img, pixels, predicted_pixels)