import numpy as np
import cv2
import math
from harris_corner_detector import get_harris_corners
from filters import gaussian_filter
from Get3d2dCoordinates import Get3d2dpoints
import json
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def NearestNeighbourInterpolation(img, kappa, kappa2=None):
    def undistPixel(xd, yd, kappa, kappa2=None):
        cx = img.shape[1] // 2
        cy = img.shape[0] // 2
        p = ((xd - cx) ** 2 + (yd - cy) ** 2) ** 0.5
        if kappa2 is None:
            xu = (xd - cx) * (1 + kappa * p**2) + cx
            yu = (yd - cy) * (1 + kappa * p**2) + cy
        else:
            xu = (xd - cx) * (1 + kappa * p**2 + kappa2 * p**4) + cx
            yu = (yd - cy) * (1 + kappa * p**2 + kappa2 * p**4) + cy
        return xu, yu

    # Find known points
    known = dict()
    out = -1 * np.ones(img.shape)
    for y in range(0, img.shape[0] * 2):
        for x in range(0, img.shape[1] * 2):
            x_real = x // 2
            y_real = y // 2
            xu, yu = undistPixel(x / 2, y / 2, kappa, kappa2)
            xu_int = int(round(xu))
            yu_int = int(round(yu))

            # If value is within the image bounds
            if (
                0 <= int(round(xu)) < img.shape[1]
                and 0 <= int(round(yu)) < img.shape[0]
            ):
                # Find closest matching point
                if (xu_int, yu_int) not in known:
                    known[(xu_int, yu_int)] = (xu_int - xu) ** 2 + (yu_int - yu) ** 2
                    out[yu_int, xu_int] = img[y_real, x_real]
                else:
                    # If a new point is found closer to the original
                    if (
                        known[(xu_int, yu_int)]
                        > (xu_int - xu) ** 2 + (yu_int - yu) ** 2
                    ):
                        known[(xu_int, yu_int)] = (xu_int - xu) ** 2 + (
                            yu_int - yu
                        ) ** 2
                        out[yu_int, xu_int] = img[y_real, x_real]

    out = out.astype(np.uint8)
    return out


def show(img, scale_percent=30, waitKey=-1):
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imshow("image", img)
    k = cv2.waitKey(waitKey) & 0xFF
    if k == ord("s"):
        cv2.imwrite("image.png", img)
        cv2.destroyAllWindows()
    if k == ord("q"):
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()


def undistorted_point(distorted_point, kappa, c_x, c_y, kappa2=None):
    """Applies the distortion model to point to compute its undisroted coordinate"""
    xd, yd = distorted_point
    p = ((xd - c_x) ** 2 + (yd - c_y) ** 2) ** 0.5
    if kappa2 is None:
        xu = (xd - c_x) * (1 + kappa * p**2) + c_x
        yu = (yd - c_y) * (1 + kappa * p**2) + c_y
    else:
        xu = (xd - c_x) * (1 + kappa * p**2 + kappa2 * p**4) + c_x
        yu = (yd - c_y) * (1 + kappa * p**2 + kappa2 * p**4) + c_y
        # print("x:", xu_test, xu)
        # print("y:", yu_test, yu)
    return (xu, yu)


def ransac_lines(points, iterations=100, threshold=3):
    # find the best line using RANSAC
    # the points input parameter is a list of points in the form (x, y)
    # the p parameter is a tuple containing the x and y coordinates of the point
    # the k parameter is the kappa value
    # the c_x and c_y parameters are the center of the image
    best_line = None
    best_inliers = []
    best_inlier_count = 0

    for i in range(iterations):
        # select two points
        i1, i2 = np.random.choice(len(points), size=2, replace=False)
        p1, p2 = points[i1], points[i2]
        x1, y1 = p1
        x2, y2 = p2

        # compute the line
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        # find all inliers on the line within the threshold
        inliers = []
        for p in points:
            x, y = p
            # find the distance of the point to the line
            dist = abs((m * x) - y + c) / (m**2 + 1) ** 0.5
            if dist <= threshold:
                inliers.append(p)

            # if the current line has more inliers than the best line, update the best line
            if len(inliers) > best_inlier_count:
                best_line = (m, c)
                best_inliers = inliers
                best_inlier_count = len(inliers)

    return best_line, best_inliers


def least_square_error(line, points):
    """
    Computes the least square error between a line and a set of points

    :param line: tuple containing gradient and inteercept (m, c)
    :param points: list of tuples containing (x, y) coordinates
    :returns: float mean squared error
    """
    line_array = np.reshape(np.array(line).T, (2, 1))
    points_array = np.array(points)
    points_x, points_y = np.hsplit(points_array, 2)
    predictors = np.concatenate([points_x, np.ones(points_x.shape)], axis=1)
    fit_points = predictors @ line_array
    errors = fit_points - points_y
    sq_errors = errors**2
    return np.sum(sq_errors)


def draw_line(im, line):
    """Takes a line in the form (m, c) and draws it on the supplied image"""
    m, c = line
    x1 = -10
    y1 = round(m * x1 + c)
    x2 = im.shape[1] + 10
    y2 = round(m * x2 + c)

    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_points(im, points, col=(0, 0, 255)):
    "Draws a set of corner points onto an image"
    for p in points:
        p = (round(p[0]), round(p[1]))
        cv2.circle(im, p, 8, col, -1)


def draw_groups(im, groups):
    for g in groups:
        draw_points(im, g)
        minX, minY = g[0]
        maxX, maxY = g[0]
        for x, y in g:
            if x < minX:
                minX = x
            if y < minY:
                minY = y
            if x > maxX:
                maxX = x
            if y > maxY:
                maxY = y

        cv2.rectangle(
            im, (minX - 20, minY - 20), (maxX + 20, maxY + 20), (255, 0, 0), 3
        )


def group_points(
    im, points, border_percent=0.08, min_row_length=5, max_row_length=8, debug=False
):
    x_cutoff = im.shape[1] * border_percent
    y_cutoff = im.shape[0] * border_percent
    points = [
        p
        for p in points
        if p[0] >= x_cutoff
        and p[0] <= im.shape[1] - x_cutoff
        and p[1] >= y_cutoff
        and p[1] <= im.shape[0] - y_cutoff
    ]

    top = 90
    right = 200
    left = 0
    bottom = 35
    points.sort()
    groups = []

    while len(points):
        new_group = []
        current_point = points[0]
        points.pop(0)
        new_group.append(current_point)
        top_left = (current_point[0] - left, current_point[1] - top)
        bottom_right = (current_point[0] + right, current_point[1] + bottom)

        for other_p in points:
            if (
                other_p[0] >= top_left[0]
                and other_p[0] <= bottom_right[0]
                and other_p[1] >= top_left[1]
                and other_p[1] <= bottom_right[1]
            ):
                current_point = other_p
                new_group.append(current_point)
                points.remove(current_point)
                top_left = (current_point[0] - left, current_point[1] - top)
                bottom_right = (current_point[0] + right, current_point[1] + bottom)

            if len(new_group) >= max_row_length:
                break

        if len(new_group) >= min_row_length:
            groups.append(new_group)

    if debug:
        for g in groups:
            print(g)
            cop = np.copy(im)
            draw_points(cop, g)

            for p in g:
                top_left = (p[0] - left, p[1] - top)
                bottom_right = (p[0] + right, p[1] + bottom)
                cv2.rectangle(cop, top_left, bottom_right, (255, 0, 0), 3)

            show(cop)

    return groups


def read_settings():
    f = open("A2_data/settings.json")
    settings = json.load(f)
    f.close()

    return settings


def extract_points(im):
    """Uses the provided function to get the corner points"""
    settings = read_settings()

    rows = settings["rows"]
    columns = settings["columns"]

    points = Get3d2dpoints(
        np.copy(im),
        settings["square_size"],
        rows,
        columns,
        settings["left_offset"],
        settings["right_offset"],
        debug=False,
    )

    return points


def get_points(im):
    """Uses the provided function to get the corner points and then groups them by row"""
    points = extract_points(im)
    settings = read_settings()
    rows, columns = settings["rows"], settings["columns"]

    left_groups = [[] for i in range(rows)]
    right_groups = [[] for i in range(rows)]

    for i in range(len(left_groups)):
        for j in range(columns):
            target_point = points[i + j * rows]
            left_groups[i].append((round(target_point[3]), round(target_point[4])))

    for i in range(len(right_groups)):
        for j in range(columns):
            target_point = points[rows * columns + i + j * rows]
            right_groups[i].append((round(target_point[3]), round(target_point[4])))

    return left_groups + right_groups


def calculate_error(kappas, cx, cy, point_groups, harris=False):
    kappa, kappa2 = kappas
    undistorted_groups = [
        [undistorted_point(p, kappa, cx, cy, kappa2) for p in group]
        for group in point_groups
    ]

    # calculate the best lines for each group of undistorted points
    filtered_lines = []
    for group in undistorted_groups:
        lines, _ = ransac_lines(group)
        if len(lines) > 0:
            filtered_lines.append(lines)
    lines = filtered_lines

    # compute the average least square error for each line and its corrosponding set of distorted points
    error_values = [
        least_square_error(l, points) for l, points in zip(lines, undistorted_groups)
    ]
    if not harris:
        err = sum(error_values) / len(lines)
    else:
        err = np.median(np.array(error_values))

    return err


def remove_distortion(
    distored_image,
    err_threshold=5,
    k_nudge=1e-8,
    kappa2=False,
    harris=False,
    debug=False,
):
    """Removes barrel distortion from an image"""

    # get points and group them by row
    if harris:
        grey_image = cv2.cvtColor(distored_image, cv2.COLOR_BGR2GRAY)
        grey_image = gaussian_filter(grey_image)
        points = get_harris_corners(grey_image, threshold=1.5e7, nms_size=51)
        print("Finished detecting corners")
        point_groups = group_points(distored_image, points, debug=False)
        draw_groups(distored_image, point_groups)
        show(distored_image)
    else:
        point_groups = get_points(distored_image)

    print(f"{len(point_groups)} groups detected")

    # Remove outliers using RANSAC
    filtered_groups = []
    for group in point_groups:
        _, inliers = ransac_lines(group)
        if len(inliers) > 0:
            filtered_groups.append(inliers)
    print(f"{len(filtered_groups)} groups detected after outlier removal")
    point_groups = filtered_groups

    c_x = distored_image.shape[1] // 2
    c_y = distored_image.shape[0] // 2
    k = 0
    err = err_threshold

    if not kappa2:
        # determine the optimal k
        while err >= err_threshold:
            # compute the undistorted equation
            undistorted_groups = [
                [undistorted_point(p, k, c_x, c_y, 0) for p in group]
                for group in point_groups
            ]

            # calculate the best lines for each group of undistorted points
            filtered_lines = []
            for group in undistorted_groups:
                lines, _ = ransac_lines(group)
                if len(lines) > 0:
                    filtered_lines.append(lines)
            lines = filtered_lines

            # compute the average least square error for each line and its corrosponding set of distorted points
            error_values = [
                least_square_error(l, points)
                for l, points in zip(lines, undistorted_groups)
            ]
            if not harris:
                err = sum(error_values) / max(len(lines),1)
            else:
                err = np.median(np.array(error_values))

            if debug:
                print(f"current k = {k}")
                print(f"avg err = {err}")
                im_copy = np.copy(distored_image)
                draw_points(im_copy, [p for g in point_groups for p in g], (255, 0, 0))
                draw_points(im_copy, [p for g in undistorted_groups for p in g])
                for l in lines:
                    draw_line(im_copy, l)
                show(im_copy)

            # if the error is less than the threshold increase k
            if err >= err_threshold:
                k += k_nudge
                if k >= 0.5:
                    print("ERROR: max number of iterations reached, no k value found")
                    return None

        print(f"Estimated k = {k}")
    else:
        error_score = partial(
            calculate_error, cx=c_x, cy=c_y, point_groups=point_groups
        )
        kappas = [0, 0]
        res = minimize(
            error_score,
            kappas,
            method="nelder-mead",
            options={"disp": True},
        )
        k, k2 = res.x
        print(f"Estimated k = {k}")
        print(f"Estimated k2 = {k2}")

        error = error_score([k, k2])
        print(f"Err = {error}")

    # Warp the image using the calculate k value
    if not kappa2:
        undistorted_image = NearestNeighbourInterpolation(distored_image, k)
        return undistorted_image
    else:
        undistorted_image = NearestNeighbourInterpolation(distored_image, k, k2)
        return undistorted_image


# A2 -----------------------------------------------------------------


def estimate_2D_from_3D(point, WRF_to_CRF, projection_matrix, CRF_to_image):
    point = CRF_to_image @ projection_matrix @ WRF_to_CRF @ np.array([*point, 1])
    point /= point[2]
    point = (round(point[0]), round(point[1]))
    return point


def draw_world_axis(img, WRF_to_CRF, projection_matrix, CRF_to_image):
    orig_est = estimate_2D_from_3D(
        (0, 0, 0), WRF_to_CRF, projection_matrix, CRF_to_image
    )
    x = estimate_2D_from_3D((10, 0, 0), WRF_to_CRF, projection_matrix, CRF_to_image)
    cv2.circle(img, x, 15, (0, 0, 255), -1)
    cv2.line(img, x, orig_est, (0, 0, 255), 5)
    y = estimate_2D_from_3D((0, 10, 0), WRF_to_CRF, projection_matrix, CRF_to_image)
    cv2.circle(img, y, 15, (0, 255, 0), -1)
    cv2.line(img, y, orig_est, (0, 255, 0), 5)
    z = estimate_2D_from_3D((0, 0, 10), WRF_to_CRF, projection_matrix, CRF_to_image)
    cv2.circle(img, z, 15, (255, 0, 0), -1)
    cv2.line(img, z, orig_est, (255, 0, 0), 5)
    cv2.circle(img, orig_est, 15, (5, 165, 245), -1)


def tsai_calibration(im, dx, dy):
    """Arguments:
    im: the undistorted image
    dx: pixel size in the x direction (in cm)
    dy: pixel size in the y direction (in, cm)"""
    # each row of points in an np array in the form [X, Y, Z, u, v]
    # X, Y, Z are in cm, u and v are the undistorted pixel coordinates
    points = extract_points(im)

    # image center point in px
    c_x, c_y = im.shape[1] // 2, im.shape[0] // 2

    s_x = 1

    # Step 1 - Compute image coordinates in cm from image center
    xu_vector = s_x * dx * (points[:, 3] - c_x)
    yu_vector = dy * (c_y - points[:, 4])

    # Step 2 - Compute L
    M = np.zeros((points.shape[0], 7))
    for i, (world_point, xu, yu) in enumerate(zip(points[:, :3], xu_vector, yu_vector)):
        X, Y, Z = world_point
        m_i = np.array([yu * X, yu * Y, yu * Z, yu, -xu * X, -xu * Y, -xu * Z])
        M[i] = m_i

    L = np.linalg.inv((M.T @ M)) @ M.T @ xu_vector

    # Step 3 - Find the magnitude of t_y
    t_y_mag = 1 / np.linalg.norm(L[4:])

    # Step 4 - Find s_x
    s_x = t_y_mag * np.linalg.norm(L[:3])

    # Step 5 - Find the sign of t_y
    # Find the 3D point whoses image point is most distant from the center
    farthest_point = None
    farthest_dist = 0
    index = 0
    for i in range(points.shape[0]):
        dist = np.linalg.norm([xu_vector[i], yu_vector[i]])
        if dist > farthest_dist:
            farthest_dist = dist
            farthest_point = points[i, :3]
            index = i

    sign_x = (
        np.sum(np.multiply((L[:3] * t_y_mag), farthest_point)) + L[3] * t_y_mag
    ) > 0
    sign_y = (np.sum(np.multiply((L[4:] * t_y_mag), farthest_point)) + t_y_mag) > 0

    if sign_x != (xu_vector[index] > 0) or sign_y != (yu_vector[index] > 0):
        t_y = -t_y_mag
    else:
        t_y = t_y_mag

    # Step 6 - recalculate components
    R = np.zeros((3, 3))
    R[0] = np.multiply(L[:3], (t_y / s_x))
    R[1] = np.multiply(L[4:], t_y)
    t_x = L[3] * t_y / s_x

    # Step 7 - calculate remaining rotation components
    R[2] = np.cross(R[0], R[1])
    R[2] /= np.linalg.norm(R[2])

    # Step 8 - find f and t_z
    M_t = np.zeros((2, points.shape[0]))
    m_t = np.zeros((1, points.shape[0]))
    for i in range(M_t.shape[1]):
        X, Y, Z = points[i, :3]
        M_t[0, i] = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t_y
        M_t[1, i] = -yu_vector[i]

        m_t[0, i] = (R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z) * yu_vector[i]

    f, t_z = (np.linalg.inv((M_t @ M_t.T)) @ M_t) @ m_t.T
    f = f[0]
    t_z = t_z[0]
    T = np.array([t_x, t_y, t_z])

    # Print results
    print(f"R = {R}")
    print(f"T = {T}")
    print(f"s_x = {s_x}")
    print(f"f = {f}")

    # Create the 3 calibration matrices
    WRF_to_CRF = np.zeros((4, 4))
    WRF_to_CRF[:3, :3] = R
    WRF_to_CRF[:3, 3] = T.T
    WRF_to_CRF[3, 3] = 1

    projection_matrix = np.zeros((3, 4))
    projection_matrix[0, 0] = f
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = 1

    CRF_to_image = np.zeros((3, 3))
    CRF_to_image[0, 0] = s_x / dx
    CRF_to_image[1, 1] = -1 / dy
    CRF_to_image[0, 2] = c_x
    CRF_to_image[1, 2] = c_y
    CRF_to_image[-1, -1] = 1

    return WRF_to_CRF, projection_matrix, CRF_to_image


def compute_error(points1, points2):
    """Takes two sets of either 3D or 2D points and computers the average
    error and standard deviation error"""
    diff = points1-points2
    errors = np.apply_along_axis(np.linalg.norm, 1, diff)
    return (np.mean(errors), np.std(errors))


def root_mean_square_error(points1, points2):
    """Projects each world point to the image and determines the error between
    the projected image point and true_image points"""
    num_points = len(points1)
    diff = points1-points2
    sq_diff = np.power(diff, 2)
    return np.sqrt(np.sum(sq_diff)/num_points)


def apply_homogenous_mat(mat, point):
    result = mat @ np.array([*point, 1])
    result /= result[-1]
    return result[:-1]


def pixels_to_WRF(pixel_point, dx, dy, s_x, c_x, c_y, f, WRF_to_CRF):
    xu = dx * s_x * (pixel_point[0] - c_x)
    yu = dy * (c_y - pixel_point[1])
    CRF_to_WRF = np.linalg.inv(WRF_to_CRF)
    world_point = CRF_to_WRF @ np.array([xu, yu, f, 1])
    world_point /= world_point[3]
    return world_point[:3]


def cube_calibration_error(image_points, true_world_points, focal_length, WRF_to_CRF,
                           dx, dy, s_x, c_x, c_y):
    """Uses the calibration parameters to back project an image point to the cube
    and determines the error between the back projected point and true world point"""
    estimated_world_points = np.zeros((image_points.shape[0], 3))
    camera_origin = apply_homogenous_mat(np.linalg.inv(WRF_to_CRF), (0, 0, 0))
    for i, image_point in enumerate(image_points):
        image_point_wrf = pixels_to_WRF(image_point, dx, dy, s_x, c_x, c_y, focal_length, WRF_to_CRF)
        ray_direction = image_point_wrf - camera_origin

        if i < image_points.shape[0]//2: #if the point is in the left board
            t = -camera_origin[0]/ray_direction[0]
        else: # if the point is in the right board
            t = -camera_origin[1]/ray_direction[1]

        estimated_world_points[i] = camera_origin + t * ray_direction

    return estimated_world_points, *compute_error(true_world_points, estimated_world_points), root_mean_square_error(true_world_points, estimated_world_points)


def plot_pixel_projections(img: np.ndarray, real_pixels: np.ndarray, projected_pixels: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.imshow(img)

    pixels_list = np.int32(np.round(real_pixels))
    predicted_pixels_list = np.int32(np.round(projected_pixels))

    for i in range(len(pixels_list)):
        circle_true = patches.Circle(tuple(pixels_list[i]), 4, edgecolor='blue', facecolor='blue', linewidth=2)
        circle_pred = patches.Circle(tuple(predicted_pixels_list[i]), 4, edgecolor='red', facecolor='red', linewidth=2)
        
        ax.add_patch(circle_true)
        ax.add_patch(circle_pred)

    legend_true = plt.Line2D([], [], color="Blue", marker="o", linewidth=0, label="Real Pixels")
    legend_pred = plt.Line2D([], [], color="Red", marker="o", linewidth=0, label="Projected Pixels")

    # Add the legend to the plot
    ax.legend(handles=[legend_true, legend_pred])

    ax.set_title("Results")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (py)")

    plt.show()


def plot_cube_projections(real_world_points, projected_world_points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(real_world_points[:, 0], real_world_points[:, 1], real_world_points[:, 2], color="b")
    ax.scatter(projected_world_points[:, 0], projected_world_points[:, 1], projected_world_points[:, 2], color="r")
    
    legend_true = plt.Line2D([], [], color="Blue", marker="o", linewidth=0, label="Real cube points")
    legend_pred = plt.Line2D([], [], color="Red", marker="o", linewidth=0, label="Estimated cube points")
    ax.legend(handles=[legend_true, legend_pred])
    ax.set_title("Results")
    plt.show()


def ray_intersection(p1, p2, p3, p4):
    d21 = p2 - p1
    d34 = p3 - p4
    d13 = p1 - p3

    mat1 = np.array([[d21.T @ d21, d21.T @ d34],
                        [d21.T @ d34, d34.T @ d34]])
    
    mat2 = np.array([[-d13.T @ d21],
                        [-d13.T @ d34]])

    t1, t2 = np.linalg.inv(mat1) @ mat2
    t1, t2 = t1[0], t2[0]

    intersection_point_left = p1 + t1 * (p2 - p1)
    intersection_point_right = p3 + t2 * (p4 - p3)
    intersection_point = (intersection_point_left + intersection_point_right)/2

    return intersection_point


def single_image_calibrate(im_name):
    img = cv2.imread(im_name)

    print("Distortion removal:")
    undistorted_image = remove_distortion(img, kappa2=False, harris=False, debug=False)
    show(undistorted_image)
 
    # load in the undistorted_H3.jpg image - for debugging
    #undistorted_image = cv2.imread("A2_data/undistorted_H3.jpg")
 
    # read the pixel size from json settings
    settings = read_settings()
    name = im_name.split(".")[0][-2:]
    dx = settings[f"{name}_pixel_size_x"]
    dy = settings[f"{name}_pixel_size_y"]

    print("Calibration:")
    WRF_to_CRF, projection_matrix, CRF_to_image = tsai_calibration(
        undistorted_image, dx, dy
    )
    c_x, c_y = img.shape[1]//2, img.shape[0]//2
    s_x = CRF_to_image[0, 0] * dx

    calibration_matrix = CRF_to_image @ projection_matrix @ WRF_to_CRF
    print("Calibration Matrix:")
    print(calibration_matrix)
    print("-"*80)

    points = extract_points(undistorted_image)
    pixels = points[:,3:]
    real_points = points[:, :3]

    predicted_pixels = np.apply_along_axis(estimate_2D_from_3D, 1, real_points, WRF_to_CRF, projection_matrix, CRF_to_image)
    mean_pixel_error, std_pixel_error = compute_error(pixels, predicted_pixels)
    rmse = root_mean_square_error(pixels, predicted_pixels)
    print(name, "Pixel Error Mean:", round(mean_pixel_error,2))
    print(name, "Pixel Error Standard Deviation:", round(std_pixel_error,2))
    print(name, "Pixel Error RMSE:", round(rmse,2))
    plot_pixel_projections(undistorted_image, pixels, predicted_pixels)

    # Calculate the error between the projected points and the true world points    
    backproj_world_points, mean_cube_error, std_cube_error, rmse_cube_error = cube_calibration_error(predicted_pixels, real_points, projection_matrix[0, 0], WRF_to_CRF,
                                                             dx, dy, s_x, c_x, c_y)

    # Print the cube error results
    print(name, "Cube Error Mean:", round(mean_cube_error,2))
    print(name, "Cube Error Standard Deviation:", round(std_cube_error,2))
    print(name, "Cube Error RMSE:", round(rmse_cube_error,2))
    print("="*80)
    
    # Plot the pixel projections
    plot_cube_projections(real_points, backproj_world_points)
    
    return calibration_matrix


def stereo_image_calibrate(im_name):
    left_im = cv2.imread(f"Assignment-2-Phase-2-DATA/{im_name}/Left/{im_name}_Cube_Left.png")
    right_im = cv2.imread(f"Assignment-2-Phase-2-DATA/{im_name}/Right/{im_name}_Cube_Right.png")

    print("Distortion removal (left image):")
    undistorted_left_image = remove_distortion(left_im, kappa2=False, harris=False, debug=False)
    show(undistorted_left_image)
    print()

    print("Distortion removal (right image):")
    undistorted_right_image = remove_distortion(right_im, kappa2=False, harris=False, debug=False)
    show(right_im)
    print()

    # read the pixel size from json settings
    settings = read_settings()
    dx = settings[f"{im_name}_pixel_size_x"]
    dy = settings[f"{im_name}_pixel_size_y"]

    print("Calibration (left image):")
    WRF_to_CRF_l, projection_matrix_l, CRF_to_image_l = tsai_calibration(undistorted_left_image, dx, dy)
    calibration_matrix_l = CRF_to_image_l @ projection_matrix_l @ WRF_to_CRF_l
    print("Calibration Matrix:")
    print(calibration_matrix_l)
    print()

    print("Calibration (right image):")
    WRF_to_CRF_r, projection_matrix_r, CRF_to_image_r = tsai_calibration(undistorted_right_image, dx, dy)
    calibration_matrix_r = CRF_to_image_r @ projection_matrix_r @ WRF_to_CRF_r
    print("Calibration Matrix:")
    print(calibration_matrix_r)
    print()

    print("-"*80)
    f_l = projection_matrix_l[0, 0]
    f_r = projection_matrix_r[0, 0]
    c_x, c_y = left_im.shape[1]//2, left_im.shape[0]//2
    s_x_l = CRF_to_image_l[0, 0] * dx
    s_x_r = CRF_to_image_r[0, 0] * dx
    cam_center_left_wrf = apply_homogenous_mat(np.linalg.inv(WRF_to_CRF_l), (0, 0, 0))
    cam_center_right_wrf = apply_homogenous_mat(np.linalg.inv(WRF_to_CRF_r), (0, 0, 0))

    points_l = extract_points(undistorted_left_image)
    points_r = extract_points(undistorted_right_image)

    estimated_points = np.zeros((points_l.shape[0], 3))
    for i, (row_l, row_r) in enumerate(zip(points_l, points_r)):
        image_point_left_wrf = pixels_to_WRF(row_l[3:], dx, dy, s_x_l, c_x, c_y, f_l, WRF_to_CRF_l)
        image_point_right_wrf = pixels_to_WRF(row_r[3:], dx, dy, s_x_r, c_x, c_y, f_r, WRF_to_CRF_r)

        estimated_points[i, :] = ray_intersection(cam_center_left_wrf, image_point_left_wrf, cam_center_right_wrf, image_point_right_wrf)

    mean_stereo_err, std_stereo_err = compute_error(points_l[:, :3], estimated_points)
    rmse = root_mean_square_error(points_l[:, :3], estimated_points)
    print(im_name, "Stereo Error Mean:", round(mean_stereo_err,2))
    print(im_name, "Stereo Error Standard Deviation:", round(std_stereo_err,2))
    print(im_name, "Stereo Error RMSE:", round(rmse,2))
    print("="*80)

    plot_cube_projections(points_l[:, :3], estimated_points)