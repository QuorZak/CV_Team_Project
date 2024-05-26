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


    question = 4


    if (question == 1): # cameras with baseline
        x = calculate_stereo_parameter_Zmin(35,1.5,1000,0.002, None ,True) # baseline, focal_length, num_pixels, pixel_width, Zmin, is_Zmax

    elif (question == 2): # baseline from transforms
        x1,y1,z1,x2,y2,z2 = -5,2,54,5,9,10
        x = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) #p1_vector, p2_vector
    
    elif (question == 3): # camera pixel to scene, scene to cam pixel
        x,y,z = calculate_scene_coordinates(6,(132,163),(181,95),378,0.001,(3.09,4.28,378)) # (f, optical_center, pixel, Z, pixel_size, scene_pixel):
    
    elif (question == 4): # SSD then SAD
        template = [[11, 3, 6],
                    [9, 11, 4],
                    [11, 8, 6]]

        window = [[2, 6, 6],
                  [9, 12, 11],
                  [8, 4, 2]]

        template = np.array(template)
        window = np.array(window)
        # Calculate SSD
        ssd = np.sum((template - window) ** 2)
        # Calculate SAD
        sad = np.sum(np.abs(template - window))

        x,y = ssd, sad
    
    elif (question == 5):
        pass
    
    elif (question == 6):
        pass
    
    elif (question == 7):
        pass
    
    elif (question == 8):
        pass
    
    elif (question == 9):
        pass
    
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
        return round(pixel_width, 6)  # rounded to 6 decimal places

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
        Z = (f * max(abs(x), abs(y))) / pixel_size  # Use maximum value of x and y
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


if __name__ == "__main__":
    main()