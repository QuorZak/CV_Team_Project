import numpy as np
import cv2
import math
import random

from dlt import dlt

def apply_homography(H, point):
    result = np.matmul(H, np.array([point[0], point[1], 1]))
    return (result[0]/result[2], result[1]/result[2])


def check_colinear(p1, p2, p3):
    triangle_area = 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    return abs(triangle_area) < 1e-5


def point_difference(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def ransac(pairs, num_rounds=1000, inlay_treshold=3, blob_mode=False):
    best_transform = None
    best_inliers = 0

    if blob_mode: # turns blobs into points by only keeping the x and y values
        pairs = [( (p1[0],p1[1]) , (p2[0],p2[1]) ) for p1, p2 in pairs]

    for iteration in range(num_rounds):
        chosen_points = random.sample(pairs, 4)
        colinear_points_picked = False
        for i in range(4):
            i_1 = i % 4
            i_2 = (i + 1) % 4
            i_3 = (i + 2) % 4
            if check_colinear(chosen_points[i_1][0], chosen_points[i_2][0], chosen_points[i_3][0]):
                colinear_points_picked = True
                break

        if colinear_points_picked: continue
                
        H = dlt(chosen_points)

        inlier_count = 0
        for pair in pairs:
            p1, p2 = pair
            transformed_p1 = apply_homography(H, p1)
            if point_difference(transformed_p1, p2) <= inlay_treshold: inlier_count += 1

        if best_inliers < inlier_count:
            best_transform = H
            best_inliers = inlier_count


    if best_transform is None:
        print("No transformation found")
        return None

    inlier_pairs = []
    for pair in pairs:
        p1, p2 = pair
        transformed_p1 = apply_homography(best_transform, p1)
        if point_difference(transformed_p1, p2) <= inlay_treshold: inlier_pairs.append(pair)
    return dlt(inlier_pairs)
    

def in_image(im, point):
    h, w, _ = im.shape
    return point[0] >= 0 and point[0] < w and point[1] >= 0 and point[1] < h


def interpolate_colour(im, p):
    x1 = math.floor(p[0])
    x2 = x1 + 1
    y1 = math.floor(p[1])
    y2 = y1 + 1
    
    a = p[0] - x1
    b = p[1] - y1

    top_left_col = im[y1][x1] if in_image(im, (x1, y1)) else 0
    top_right_col = im[y1][x2] if in_image(im, (x2, y1)) else 0
    bottom_left_col = im[y2][x1] if in_image(im, (x1, y2)) else 0
    bottom_right_col = im[y2][x2] if in_image(im, (x2, y2)) else 0

    colour = (1 - a) * (1 - b) * top_left_col + a * (1 - b) * top_right_col + (1 - a) * b * bottom_left_col + a * b * bottom_right_col
    return colour


def blend(c1, c2):
    return (c1 + c2)/2


def crop_canvas(im):
    h, w, _ = im.shape
    top = 0
    right = w - 1
    bottom = h - 1
    left = 0

    while not np.all(np.sum(im[top, :, :], 1)):
        top += 1
        if top >= h:
            top = 0
            break

    while not np.all(np.sum(im[bottom, :, :], 1)):
        bottom -= 1
        if bottom < 0:
            bottom = h - 1
            break

    while not np.all(np.sum(im[:, left, :], 1)):
        left += 1
        if left >= w:
            left = 0
            break

    while not np.all(np.sum(im[:, right, :], 1)):
        right -= 1
        if right < 0:
            right = w - 1
            break

    return im[top:bottom + 1, left:right + 1]


def warp_image(left_image, right_image, H, crop = False, preloaded=False):
    if not preloaded:
        left_image = cv2.imread(left_image)
        right_image = cv2.imread(right_image)

        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)

    h, w, c = left_image.shape
    canvas = np.zeros((h, 2 * w, c))

    for y in range(0, h):
        for x in range(0, 2 * w):
            point = (x, y)
            transformed_point = apply_homography(H, point)
            p_in_left_im = in_image(left_image, point)
            transformed_p_in_right_im = in_image(right_image, transformed_point)

            if p_in_left_im and not transformed_p_in_right_im:
                #take colour from left im
                colour = left_image[y][x]
            elif p_in_left_im and transformed_p_in_right_im:
                #take colour from both and blend
                left_colour = left_image[y][x]
                right_colour = interpolate_colour(right_image, transformed_point)
                colour = blend(left_colour, right_colour)
            elif not p_in_left_im and transformed_p_in_right_im:
                #take colour from right image
                colour = interpolate_colour(right_image, transformed_point)
            else:
                colour = np.array([0, 0, 0])

            canvas[y][x] = colour

    canvas = (np.rint(canvas)).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2RGB)
    if crop: canvas = crop_canvas(canvas)

    return canvas
