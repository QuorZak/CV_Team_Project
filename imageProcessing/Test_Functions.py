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


    question = 9


    if (question == 1): # Z min and Z max from baseline
        x = calculate_stereo_parameter_Zmin(50,2,1000,0.001, 200 ,False) # baseline, focal_length, num_pixels, pixel_width, Zmin, is_Zmax

    elif (question == 2): # baseline from transforms
        x1,y1,z1,x2,y2,z2 = -5,2,54,5,9,10
        x = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) #p1_vector, p2_vector
    
    elif (question == 3): # camera pixel to scene, scene to cam pixel
        x,y,z = calculate_scene_coordinates(1,(0,0),(131,142),1,0.001,None) # (f, optical_center, pixel, Z, pixel_size, scene_pixel):
    
    elif (question == 4): # SSD then SAD
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
    
    elif (question == 5): # cross product
        v1 = [-3, -1, -9]
        v2 = [3, -4, -4]

        x = v1[1] * v2[2] - v1[2] * v2[1]
        y = v1[2] * v2[0] - v1[0] * v2[2]
        z = v1[0] * v2[1] - v1[1] * v2[0]
    
    elif (question == 6): # placeholder
        pass
    
    elif (question == 7): # intersection of a point on plane
        ray_origin = [1, 1, 1]
        ray_direction = [2, 1, -1]
        plane_equation = [1, 2, 4, 7]
        x,y,z = intersection_point(ray_origin, ray_direction, plane_equation)
    
    elif (question == 8): # max kappa and rd
        x,y = compute_kappa1_max(4000,3000,8,6,(8/4000),False) # image_width, image_height, sensor_width, sensor_height, pixel_width (sensorW/imageW), pixel mode
    
    elif (question == 9): # disparity (Z delta) from baseline
        x = calculate_delta_Z_minus(200,50,2,1000,0.001) # disparity, baseline, focal_length, num_pixels, pixel_width
    
    elif (question == 10):
        pass
    
    elif (question == 11):
        pass
    
    elif (question == 12):
        pass
    
    elif (question == 13):
        pass
    
    elif (question == 14):
        pass
    
    elif (question == 15):
        pass
    
    elif (question == 16):
        pass
    
    elif (question == 17):
        pass
    
    elif (question == 18):
        pass
    
    elif (question == 19):
        pass
    
    elif (question == 20):
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

def calculate_scene_coordinates(f=None, optical_center=None, pixel=None, Z=None, pixel_size=None, scene_pixel=None):
    if optical_center is None:
        u, v = pixel
        u0 = u - (pixel[0] * f) / Z
        v0 = v - (pixel[1] * f) / Z
        return [u0, v0, 0]

    if pixel is None:
        u0, v0 = optical_center
        u = (f * pixel_size * pixel[0] + u0 * Z) / f
        v = (f * pixel_size * pixel[1] + v0 * Z) / f
        return [u, v, 0]

    if Z is None:
        u0, v0 = optical_center
        u, v = pixel
        x = (u - u0) * pixel_size
        y = (v - v0) * pixel_size
        Z = (f * max(abs(x), abs(y))) / pixel_size
        return 0,0,Z

    if pixel_size is None:
        u0, v0 = optical_center
        u, v = pixel
        x = (u - u0) * pixel_size
        y = (v - v0) * pixel_size
        pixel_size = (f * max(abs(x), abs(y))) / Z
        return pixel_size,0,0

    if scene_pixel is None:
        u0, v0 = optical_center
        u, v = pixel
        x = (u - u0) * pixel_size
        y = (v - v0) * pixel_size
        X = (x * Z) / f
        Y = (y * Z) / f
        return [X, -Y, Z]

    if scene_pixel is not None:
        u0, v0 = optical_center
        X, Y, Z = scene_pixel
        u = (f * X / Z) / pixel_size + u0
        v = (f * (-Y) / Z) / pixel_size + v0 
        return [u, v, 0]

    return 0,0,0

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

def compute_kappa1_max(image_width, image_height, sensor_width, sensor_height, pixel_width, pixel_k = False):
    center_x = sensor_width / 2
    center_y = sensor_height / 2
    distance = np.sqrt((center_x ** 2) + (center_y ** 2))
    
    kappa1_max = (pixel_width / (distance ** 3))

    if (pixel_k): kappa1_max = kappa1_max * (pixel_width ** 2)
    
    return kappa1_max, (pixel_width / (distance ** 3))

def calculate_delta_Z_minus(disparity, baseline_length, focal_length, num_pixels_per_line, pixel_width):
    
    Z_d = (baseline_length * focal_length) / (pixel_width * disparity)
    Z_d_plus_1 = (baseline_length * focal_length) / (pixel_width * (disparity + 1))
    
    delta_Z_minus = Z_d - Z_d_plus_1
    
    return round(delta_Z_minus, 2)


if __name__ == "__main__":
    main()