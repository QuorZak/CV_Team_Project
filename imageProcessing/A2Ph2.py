import numpy as np
from matplotlib import pyplot as plt
import glob
import matplotlib.pyplot as plt
import cv2
from Get3d2dCoordinates import *
from A2_utilities import *


def main():

    # Given H3 and W3 images, the main goal is to calibrate the left and right cameras using Tsai method, computing 
    #     Calibration error in the image (for both left and right datasets, for both W3 and H3 cameras)
    #     Calibration error in the cube (for both left and right datasets, for both W3 and H3 cameras)
    #     Stereo calibration error for both cameras

    # All the input images must be undistorted before doing calibration
    #     You can use your own distortion removal model to undistort the distorted left and right images
    #     For those still having problems with the distortion removal, they can use the provided undistorted left and right images

    # We have also provided the coordinate sheets with each row specifying a corner: 
    #     X, Y, Z are the corresponding 3D coordinates 
    #     u and v are the distorted corner coordinates
    #     u' and v' are the undistorted corner coordinates (you need these coordinates and their corresponding X, Y, Z values for calibration

    # For those who do not want to use the provided sheets:
    #     There is a py file that provides a pipeline to detect all the 3D and 2D coordinates. 
    #     from Get3d2dCoordinates import *
    #     Use the following format to get all the points
    #     pattern_size = pattern_size = (10,7)
    #     points3d2d = Get3d2dpoints(img, square_size=3.5, rows=pattern_size[0], columns=pattern_size[1], left_offset=3.4,  right_offset=3.4, debug=False)
    #     In points3d2d, the first three values are for 3D coordinates, and the rest is for corresponding 2D corner coordinate
    #     It is clear that if img is the distorted image (or undistorted image), you will have distorted (or undistorted) corner coordinates.

    # Please plot everything, projection to the image and backprojection to the cube for all both left and right datasets
    #     Let us know if you find it difficult to plot them. 


    # If you want to do any extension, please provide a separate py file or at least make it clear that this part of code is for the extension.

    # PLEASE DEFINE ALL YOUR FUNCTIONS IN UTILITIES AND IMPORT THEM HERE


    print("Perfoming single image calibration:")
    for im_name in ["A2_data/H3.jpg", "A2_data/W3.png"]:
        single_image_calibrate(im_name)

    print("\nPerfoming stereo image calibration:")
    stereo_image_calibrate("H3")
    stereo_image_calibrate("W3")


if __name__ == "__main__":
    main()
