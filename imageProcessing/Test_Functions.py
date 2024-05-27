import numpy as np
from matplotlib import pyplot as plt
import glob
import matplotlib.pyplot as plt
import cv2

from utilities import *
from A2_utilities import *

def main():
    x = 0
    y = 0
    z = 0


    question = 3

    if (question == 1): # cross product
        v1 = [1, 3, 5]
        v2 = [5, 4, 8]

        x = v1[1] * v2[2] - v1[2] * v2[1]
        y = v1[2] * v2[0] - v1[0] * v2[2]
        z = v1[0] * v2[1] - v1[1] * v2[0]

    elif (question == 2): # M-1 2D affine transform Q3
        theta = 30
        translation = [8, 2]
        x = affine_transformation_inverse(theta, translation)

    elif (question == 3): # Z min and Z max from baseline Quiz1
        baseline = 35
        focal_length = 1.5
        num_pixels = 1 # 1 for Zmax, max pixels for Zmin
        pixel_width = 0.002
        x = (baseline * focal_length) / (num_pixels * pixel_width)
        if (baseline == None):
            y = baseline = (Zmin * num_pixels * pixel_width) / focal_length

    elif (question == 4): # baseline from transforms
        x1,y1,z1,x2,y2,z2 = -5,2,54,5,9,10
        x = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    
    elif (question == 6): # Z = 0, image pixel to 3d scene location, with P array Q5
        P = np.array([
        [0.7547, 0, 0.6561, 13.0],
        [0, 1, 0, 12.0],
        [-0.6561, 1, 0.7547, 76],
        [0, 0, 0, 1]])
        u, v = 337, 134
        f = 7
        cx, cy = 113, 195
        dx, dy = 0.001, 0.001
        sx = 1
        x,y,z = compute_3d_location(P, u, v, f, cx, cy, dx, dy, sx) 
    
    elif (question == 7): # intersection of a point on plane
        ray_origin = [0, 1, -1]
        ray_direction = [1, 1, 1]
        plane_equation = [1, 1, 1, 2] # x y z = _
        x,y,z = intersection_point(ray_origin, ray_direction, plane_equation)

    elif (question == 8): # SSD then SAD
        template = [[11, 3, 6],
                    [9, 11, 4],
                    [11, 8, 6]]

        window = [[2, 6, 6],
                    [9, 12, 11],
                    [8, 4, 2]]

        template = np.array(template)
        window = np.array(window)
        ssd = np.sum((template - window) ** 2)
        sad = np.sum(np.abs(template - window))

        x,y = ssd, sad
    
    elif (question == 9): # max kappa for delta Q7
        x = calculate_kappa1_max(8,6,4000,3000,1,'mm') # sensor_width_mm, sensor_height_mm, image_x, image_y, delta, units='p'or'mm'
    
    elif (question == 10): # disparity (Z delta) from baseline Q15
        disparity = 50
        baseline = 75
        focal_length = 3
        num_pixels = 1000
        pixel_width = 0.001
        x = calculate_delta_Z_minus(disparity, baseline, focal_length, num_pixels, pixel_width)
    
    elif (question == 11): # 3D scene to image with 3D point P Q8
        P = [31, -4, 447]
        f = 3
        cx = 178
        cy = 195
        dx = 0.001
        dy = 0.001
        x,y = project_3d_to_image(P, f, cx, cy, dx, dy)
    
    elif (question == 12): # computer undistorted pixel coordinates with given k Q9/12
        x_d = 0
        y_d = 1875
        k_m = -1.2 * 10**-3
        sensor_width = 5.0
        sensor_height = 3.75
        width_pixels = 5000
        height_pixels = 3750
        x,y = calculate_undistorted_pixel_coordinates(x_d, y_d, k_m, sensor_width, sensor_height, width_pixels, height_pixels)
    
    elif (question == 13):
        pass
    
    elif (question == 13):
        pass
    
    elif (question == 14):
        pass
    
    elif (question == 15):
        pass
    
    
    print(x)
    print(y)
    print(z)



def calculate_stereo_parameter_Zmin(baseline=None, focal_length=None, num_pixels=None, pixel_width=None, Z=None, is_max_depth=False):
    if is_max_depth:
        disparity = 1
    else:
        disparity = num_pixels

    if Z is None:
        Z = (focal_length * baseline) / (disparity * pixel_width)
        return round(Z, 2)
    
    if baseline is None:
        baseline = (Z * disparity * pixel_width) / focal_length
        return round(baseline, 2)
    
    if focal_length is None:
        focal_length = (Z * disparity * pixel_width) / baseline
        return round(focal_length, 2)
    
    if num_pixels is None:
        num_pixels = (focal_length * baseline) / (Z * pixel_width)
        return int(round(num_pixels))
    
    if pixel_width is None:
        pixel_width = (focal_length * baseline) / (Z * disparity)
        return round(pixel_width, 6)

def compute_3d_location(P, u, v, f, cx, cy, dx, dy, sx):
    u_c = sx * dx * (u - cx)
    v_c = dy * (cy - v)
    normalized_coords = np.array([u_c, v_c, f, 1])
    R = P[:3, :3]
    T = P[:3, 3]
    R_inv = R.T
    T_inv = -R_inv @ T
    P_inv = np.eye(4)
    P_inv[:3, :3] = R_inv
    P_inv[:3, 3] = T_inv
    O_cw = P_inv @ np.array([0, 0, 0, 1])
    X_cw_O, Y_cw_O, Z_cw_O, _ = O_cw
    MI_W = P_inv @ normalized_coords
    X_cw, Y_cw, Z_cw, _ = MI_W
    t = -Z_cw_O / (Z_cw - Z_cw_O)
    X_wI = X_cw_O + t * (X_cw - X_cw_O)
    Y_wI = Y_cw_O + t * (Y_cw - Y_cw_O)
    Z_wI = 0

    return np.round([X_wI, Y_wI, Z_wI], 2)

def intersection_point(ray_origin, ray_direction, plane_equation):
    A, B, C, D = plane_equation

    denominator = A * ray_direction[0] + B * ray_direction[1] + C * ray_direction[2]

    if denominator == 0:
        return (-1,-1,-1)  # No intersection, ray is parallel to the plane

    t0 = (D - A * ray_origin[0] - B * ray_origin[1] - C * ray_origin[2]) / denominator

    intersection_point = [ray_origin[0] + t0 * ray_direction[0],
                          ray_origin[1] + t0 * ray_direction[1],
                          ray_origin[2] + t0 * ray_direction[2]]

    return intersection_point

def calculate_kappa1_max(sensor_width_mm, sensor_height_mm, image_x, image_y, delta, units='p'):
    cx = image_x / 2
    cy = image_y / 2
    
    pixel_size_x = sensor_width_mm / image_x
    pixel_size_y = sensor_height_mm / image_y
    pixel_size = (pixel_size_x + pixel_size_y) / 2
    
    rd = ((image_x - cx)**2 + (image_y - cy)**2)**0.5
    
    kappa1_p = delta / (rd**3)
    
    if units == 'p':
        return kappa1_p
    elif units == 'mm':
        kappa1_m = kappa1_p * (pixel_size**2)
        return kappa1_m

def calculate_delta_Z_minus(disparity, baseline_length, focal_length, num_pixels_per_line, pixel_width):
    
    Z_d = (baseline_length * focal_length) / (pixel_width * disparity)
    Z_d_plus_1 = (baseline_length * focal_length) / (pixel_width * (disparity + 1))
    
    delta_Z_minus = Z_d - Z_d_plus_1
    
    return round(delta_Z_minus, 2)

def affine_transformation_inverse(theta_degrees, translation):
    theta_radians = np.radians(theta_degrees)
    cos_theta = np.cos(theta_radians)
    sin_theta = np.sin(theta_radians)

    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])

    T = np.array(translation).reshape(2, 1)

    R_inv = R.T
    T_inv = -R_inv @ T

    M_inv = np.eye(3)
    M_inv[:2, :2] = R_inv
    M_inv[:2, 2] = T_inv.flatten()

    return np.round(M_inv, 1)

def project_3d_to_image(P, f, cx, cy, dx, dy):
    X, Y, Z = P

    u_t = (f * X) / Z
    v_t = (f * Y) / Z

    u = (u_t / dx) + cx
    v = (-v_t / dy) + cy # remove negative sign for pointing down
    
    return round(u, 2), round(v, 2)

def calculate_undistorted_pixel_coordinates(x_d, y_d, k_m, sensor_width, sensor_height, width_pixels, height_pixels):
    cx = width_pixels / 2
    cy = height_pixels / 2
    
    pixel_size_x = sensor_width / width_pixels
    pixel_size_y = sensor_height / height_pixels
    pixel_size = (pixel_size_x + pixel_size_y) / 2
    
    k_p = k_m * (pixel_size ** 2)
    dist = (1 + (k_p * (x_d - cx)**2 + (y_d - cy)**2))

    x_u = ((x_d - cx) * dist) + cx
    y_u = ((y_d - cy) * dist) + cy
    
    return round(x_u), round(y_u)

if __name__ == "__main__":
    main()