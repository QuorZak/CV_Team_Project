
import numpy as np
import cv2
from utilities import *
from A2_utilities import *

# from PIL import Image


def main():

    # PHASE-2-ASSIGNMENT-1: distortion removal

    # Given a chessboard image, you need to undistort the image by the assumption that straight lines in the real world are expected to be straight in the undistorted image.
    # The first step is to estimate kappa (k1) using an iterative algorithm based on RANSAC line fitting.
    # Refer to the lecture and tutorial notes for more details.
    # Once k1 is estimated you can use "NearestNeighbourInterpolation" function to undistort the original image
    # The provided function "NearestNeighbourInterpolation" needs the distorted image and the estimated kappa as inputs, and it returns the undistorted image.

    W3_img = cv2.imread("A2_data/W3.png")
    H3_img = cv2.imread("A2_data/H3.jpg")
    # cv2.imshow("image", img)
    # cv2.waitKey(0)

    # 1. Import your own Harris corner detection to find all the corners in the chessboard. You will mark down if you use opencv "findChessboardCorners."

    # 2. You need to find a way to group the detected corner coordinates on horizontal lines in the chessboard image

    # 3. Define the RANSAC line-fitting algorithm to get the best slope and intercept of each horizontal line of points

    # 4. Define a function to compute the average error for the fitted lines

    # 5. You need an iterative approach to brute force the kappa values

    # 6. After estimating kappa, use "NearestNeighbourInterpolation" to undistort the original distorted image.

    # PLEASE DEFINE ALL YOUR FUNCTION IN UTILITIES AND IMPORT THEM HERE.
    print("\nRemoving distortion from W3 Image:")
    print("First order distortion removal:")
    print()
    undistorted_image = remove_distortion(W3_img, kappa2=False, harris=False, debug=False)
    show(undistorted_image)
    print("-"*80)
    print("Second order distortion removal:")
    print()
    undistorted_image = remove_distortion(W3_img, kappa2=True, harris=False, debug=False)
    show(undistorted_image)
    print("="*80)
    print("\nRemoving distortion from H3 Image:")
    print("First order distortion removal:")
    print()
    undistorted_image = remove_distortion(H3_img, kappa2=False, harris=False, debug=False)
    show(undistorted_image)
    print("-"*80)
    print("Second order distortion removal:")
    print()
    undistorted_image = remove_distortion(H3_img, kappa2=True, harris=False, debug=False)
    show(undistorted_image)


if __name__ == "__main__":
    main()
