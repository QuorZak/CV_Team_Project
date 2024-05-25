import cv2
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth

from imageProcessing.harris_corner_detector import get_harris_corners

from DoG import DoG
from DoG import show_DoG

from imageProcessing.feature_functions import improved_ncc_matching
from HoG import hog_matching

from image_warping import ransac, warp_image


# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
def prepareRGBImageFromIndividualArrays(
    r_pixel_array, g_pixel_array, b_pixel_array, image_width, image_height
):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(
    left_pixel_array, right_pixel_array, image_width, image_height
):

    matchingImage = IPUtils.createInitializedGreyscalePixelArray(
        image_width * 2, image_height
    )
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage


# This is our code skeleton that performs the stitching
def main():

    # ===========================================================
    # Settings
    # ===========================================================
    adapt = False  # adaptive thresholding for Harris
    ncc_window = 15  # window size for NCC matching

    harris_pipe = True
    DoG_pipe = False
    extra_image_pipe = False

    HoG_matching = False

    # ===========================================================
    # Load Images
    # ===========================================================
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    # filename_left_image =  "./images/panoramaStitching/bryce_left_02.png"
    # filename_right_image = "./images/panoramaStitching/bryce_right_02.png"

    start = timer()
    (image_width, image_height, px_array_left_original) = (
        IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    )
    px_array_left = IPSmooth.computeGaussianAveraging3x3(
        px_array_left_original, image_width, image_height
    )
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(
        px_array_left, image_width, image_height
    )
    (image_width, image_height, px_array_right_original) = (
        IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)
    )
    px_array_right = IPSmooth.computeGaussianAveraging3x3(
        px_array_right_original, image_width, image_height
    )
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(
        px_array_right, image_width, image_height
    )
    end = timer()
    print("elapsed time image formatting: ", end - start)

    # ===========================================================
    # Phase 1: Feature extraction
    # ===========================================================

    # Compute harris corners
    if harris_pipe:
        start = timer()
        left_im_corners = get_harris_corners(
            px_array_left, blur_kernal_size=5, adaptive_threshold=adapt
        )
        right_im_corners = get_harris_corners(
            px_array_right, blur_kernal_size=5, adaptive_threshold=adapt
        )
        end = timer()
        print("elapsed time of corner detection for both images: ", end - start)

    # Compute DoG blobs
    if DoG_pipe:
        # blob_filename =  "./images/blob_butterfly.png"
        # (image_width, image_height, blob_image_original)  = IORW.readRGBImageAndConvertToGreyscalePixelArray(blob_filename)
        # blob_image = IPSmooth.computeGaussianAveraging3x3(blob_image_original, image_width, image_height)
        # blob_image = IPPixelOps.scaleTo0And255AndQuantize(blob_image, image_width, image_height)
        # show_DoG(blob_image, blob_image, sigma=1.5, nms_window=5, max_features=250, single_image=True)
        start = timer()
        sig = 2
        max_detections = 750
        n = 8
        left_blobs = DoG(px_array_left, sigma=sig, max_features=max_detections, n=n)
        right_blobs = DoG(px_array_right, sigma=sig, max_features=max_detections, n=n)
        end = timer()
        print("elapsed time of blob detection for both images: ", end - start)

    # ===========================================================
    # Phase 2: Feature matching
    # ===========================================================

    # Compute NCC matching, returns a list of tuples (left corner, right corner match)

    # start = timer()
    # ncc_matches_list = ncc_matching(px_array_left, px_array_right, left_im_corners, right_im_corners, window_size=15)
    # end = timer()
    # print("elapsed time of feature matching: ", end - start) - Tested by Zak, 130 seconds vs 11 seconds for improved. Use improved!

    if harris_pipe:
        start = timer()
        improved_matches_list = improved_ncc_matching(
            px_array_left,
            px_array_right,
            left_im_corners,
            right_im_corners,
            window_size=ncc_window,
        )
        end = timer()
        print("elapsed time of improved feature matching: ", end - start)

    if DoG_pipe:
        start = timer()
        dog_matches = improved_ncc_matching(
            px_array_left,
            px_array_right,
            left_blobs,
            right_blobs,
            window_size=15,
            DoG_features=True,
        )
        end = timer()
        print("elapsed time of DoG detection and matching: ", end - start)

    if HoG_matching:
        filename = "./images/panoramaStitching/tongariro_right_01.png"
        (hog_image_width, hog_image_height, px_array_original) = (
            IORW.readRGBImageAndConvertToGreyscalePixelArray(filename)
        )
        px_array = IPSmooth.computeGaussianAveraging3x3(
            px_array_original, hog_image_width, hog_image_height
        )
        px_array = IPPixelOps.scaleTo0And255AndQuantize(
            px_array, hog_image_width, hog_image_height
        )

        hog_image_width = 750
        left_array = np.float32(px_array)[:, 0:hog_image_width]
        right_array = np.float32(px_array)[:, 250 : (250 + hog_image_width)]
        left_im_corners_hog = get_harris_corners(
            left_array, blur_kernal_size=5, adaptive_threshold=False
        )
        right_im_corners_hog = get_harris_corners(
            right_array, blur_kernal_size=5, adaptive_threshold=False
        )

        hog_corner_pairs = hog_matching(
            left_array,
            right_array,
            left_im_corners_hog,
            right_im_corners_hog,
            window=9,
            cell_size=4,
            n_orientations=9,
            ratio=0.8,
        )
        bad_pairs = hog_matching(
            px_array_left,
            px_array_right,
            left_im_corners,
            right_im_corners,
            window=9,
            cell_size=4,
            n_orientations=9,
            ratio=0.8,
        )

        matchingImage = prepareMatchingImage(
            px_array_left, px_array_right, image_width, image_height
        )
        pyplot.imshow(matchingImage, cmap="gray")
        ax = pyplot.gca()
        ax.set_title("HoG Feature Matching on Images With Rotation")

        # Plot matches of different images
        for p1, p2 in bad_pairs:
            connection = ConnectionPatch(
                p1, (p2[0] + image_width, p2[1]), "data", edgecolor="r", linewidth=1
            )
            ax.add_artist(connection)
        pyplot.show()

        # Plot matches of cropped image
        matchingImage = prepareMatchingImage(
            left_array, right_array, hog_image_width, hog_image_height
        )
        pyplot.imshow(matchingImage, cmap="gray")
        ax = pyplot.gca()
        ax.set_title("HoG Feature Matching on Images Without Rotation")

        for p1, p2 in hog_corner_pairs:
            connection = ConnectionPatch(
                p1, (p2[0] + hog_image_width, p2[1]), "data", edgecolor="r", linewidth=1
            )
            ax.add_artist(connection)
        pyplot.show()

    # ===========================================================
    # Phase 3: Image transformation
    # ===========================================================

    if harris_pipe:
        start = timer()
        H = ransac(improved_matches_list, 1000, 3)
        np.set_printoptions(suppress=True)
        print(np.round(H / H[2][2], 4))
        warped_im = warp_image(filename_left_image, filename_right_image, H, crop=False)
        end = timer()
        print("elapsed time of image transformation: ", end - start)

    if DoG_pipe:
        start = timer()
        H = ransac(dog_matches, 1000, 3, blob_mode=True)
        np.set_printoptions(suppress=True)
        print(np.round(H / H[2][2], 4))
        DoG_warped_im = warp_image(
            filename_left_image, filename_right_image, H, crop=False
        )
        end = timer()
        print("elapsed time of image transformation: ", end - start)

    if extra_image_pipe:
        left_image = cv2.imread("images/WestAuckland/WestAuckland1.png")
        right_image = cv2.imread("images/WestAuckland/WestAuckland2.png")
        px_array_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        px_array_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        left_height, left_width = px_array_left.shape
        right_height, right_width = px_array_right.shape

        px_array_left = IPSmooth.computeGaussianAveraging3x3(
            px_array_left, left_width, left_height
        )
        px_array_left = IPPixelOps.scaleTo0And255AndQuantize(
            px_array_left, left_width, left_height
        )

        px_array_right = IPSmooth.computeGaussianAveraging3x3(
            px_array_right, right_width, right_height
        )
        px_array_right = IPPixelOps.scaleTo0And255AndQuantize(
            px_array_right, right_width, right_height
        )

        left_im_corners = get_harris_corners(
            px_array_left, blur_kernal_size=5, adaptive_threshold=False
        )
        right_im_corners = get_harris_corners(
            px_array_right, blur_kernal_size=5, adaptive_threshold=False
        )

        improved_matches_list = improved_ncc_matching(
            px_array_left,
            px_array_right,
            left_im_corners,
            right_im_corners,
            window_size=15,
        )
        H = ransac(improved_matches_list, 1000, 3)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
        warped_1_2 = warp_image(
            "images/WestAuckland/WestAuckland1.png",
            "images/WestAuckland/WestAuckland2.png",
            H,
        )

        left_image = cv2.imread("images/WestAuckland/WestAuckland2.png")
        right_image = cv2.imread("images/WestAuckland/WestAuckland3.png")
        px_array_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        px_array_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        left_height, left_width = px_array_left.shape
        right_height, right_width = px_array_right.shape

        px_array_left = IPSmooth.computeGaussianAveraging3x3(
            px_array_left, left_width, left_height
        )
        px_array_left = IPPixelOps.scaleTo0And255AndQuantize(
            px_array_left, left_width, left_height
        )

        px_array_right = IPSmooth.computeGaussianAveraging3x3(
            px_array_right, right_width, right_height
        )
        px_array_right = IPPixelOps.scaleTo0And255AndQuantize(
            px_array_right, right_width, right_height
        )

        left_im_corners = get_harris_corners(
            px_array_left, blur_kernal_size=5, adaptive_threshold=False
        )
        right_im_corners = get_harris_corners(
            px_array_right, blur_kernal_size=5, adaptive_threshold=False
        )

        improved_matches_list = improved_ncc_matching(
            px_array_left,
            px_array_right,
            left_im_corners,
            right_im_corners,
            window_size=15,
        )
        H = ransac(improved_matches_list, 1000, 3)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
        warped_2_3 = warp_image(
            "images/WestAuckland/WestAuckland2.png",
            "images/WestAuckland/WestAuckland3.png",
            H,
        )

    # ===========================================================
    # Output visualisations
    # ===========================================================

    # Phase 1 visulation ----------------------------------------

    if harris_pipe:
        fig1, axs1 = pyplot.subplots(1, 2)
        axs1[0].set_title("Harris response left overlaid on orig image")
        axs1[1].set_title("Harris response right overlaid on orig image")
        axs1[0].imshow(px_array_left, cmap="gray")
        axs1[1].imshow(px_array_right, cmap="gray")

        # Plot corners points on image
        for corner_point in left_im_corners:
            circle = Circle(corner_point, 2.5, color="r")
            axs1[0].add_patch(circle)

        for corner_point in right_im_corners:
            circle = Circle(corner_point, 2.5, color="r")
            axs1[1].add_patch(circle)

        pyplot.show()

    if DoG_pipe:
        fig1, axs1 = pyplot.subplots(1, 2)
        axs1[0].set_title("DoG response left overlaid on orig image")
        axs1[1].set_title("DoG response right overlaid on orig image")
        axs1[0].imshow(px_array_left, cmap="gray")
        axs1[1].imshow(px_array_right, cmap="gray")

        # Plot blobs points on image
        for blob in left_blobs:
            circle = Circle(blob, 2.5, color="r")
            axs1[0].add_patch(circle)
            circle1 = Circle(
                blob[:2], blob[2] * 2**0.5, color="blue", linewidth=1, fill=False
            )
            axs1[0].add_patch(circle1)

        for blob in right_blobs:
            circle = Circle(blob, 2.5, color="r")
            axs1[1].add_patch(circle)
            circle1 = Circle(
                blob[:2], blob[2] * 2**0.5, color="blue", linewidth=1, fill=False
            )
            axs1[1].add_patch(circle1)

        pyplot.show()

    # Phase 2 visulation ------------------------------------------
    if harris_pipe or DoG_pipe:
        matchingImage = prepareMatchingImage(
            px_array_left, px_array_right, image_width, image_height
        )

    if harris_pipe:
        pyplot.imshow(matchingImage, cmap="gray")
        ax = pyplot.gca()
        ax.set_title("Matching image NCC")

        # Plot matches
        matches = improved_matches_list
        for p1, p2 in matches:
            connection = ConnectionPatch(
                p1, (p2[0] + image_width, p2[1]), "data", edgecolor="r", linewidth=1
            )
            ax.add_artist(connection)

        pyplot.show()

    # Plot DoG matches
    if DoG_pipe:
        pyplot.imshow(matchingImage, cmap="gray")
        ax = pyplot.gca()
        ax.set_title("Matching image DoG")

        matches = dog_matches
        for p1, p2 in matches:
            circle1 = Circle(
                p1[:2], p1[2] * 2**0.5, color="blue", linewidth=1, fill=False
            )
            ax.add_patch(circle1)
            circle2 = Circle(
                (p2[0] + image_width, p2[1]),
                p2[2] * 2**0.5,
                color="blue",
                linewidth=1,
                fill=False,
            )
            ax.add_patch(circle2)
            connection = ConnectionPatch(
                p1[:2], (p2[0] + image_width, p2[1]), "data", edgecolor="r", linewidth=1
            )
            ax.add_artist(connection)

        pyplot.show()

    # Phase 3 visulation ------------------------------------------

    if harris_pipe:
        pyplot.imshow(warped_im)
        bx = pyplot.gca()
        bx.set_title("Stitched and blended image from Harris")
        pyplot.show()

    if DoG_pipe:
        pyplot.imshow(DoG_warped_im)
        bx = pyplot.gca()
        bx.set_title("Stitched and blended image from DoG")
        pyplot.show()

    if extra_image_pipe:
        fig1, axs1 = pyplot.subplots(1, 2)
        axs1[0].set_title("West Auckland Image 1 & 2 Stitched and blended")
        axs1[1].set_title("West Auckland Image 2 & 3 Stitched and blended")
        axs1[0].imshow(warped_1_2)
        axs1[1].imshow(warped_2_3)
        pyplot.show()


if __name__ == "__main__":
    main()
