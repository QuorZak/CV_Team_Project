import numpy as np
import math
import cv2

def filter(im, kernel):
    return cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def sobel_filter(im):
    Gx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    Gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1,-2,-1]
    ])

    ix = filter(im, Gx)
    iy = filter(im, Gy)
    
    return ix, iy


def gaussian_filter(im, width=3, sigma=1):
    if width is None: #compute the width based on sigma
        width = int(2 * np.ceil(3 * sigma) + 1)
        width = width + 1 if width % 2 == 0 else width

    gaussian_kernal = np.zeros((width, width))
    k_radius = (width - 1)//2

    for x in range(-k_radius, k_radius + 1):
        for y in range(-k_radius, k_radius + 1): 
            gaussian_kernal[k_radius + x, k_radius + y] = (1/(2 * math.pi * sigma ** 2)) * math.e ** -((x**2 + y**2) / (2 * sigma**2))

    gaussian_kernal = gaussian_kernal / np.sum(gaussian_kernal)

    return filter(im, gaussian_kernal)


def gaussian_filter_seperable(im, width=3, sigma=1):
    Gx = np.zeros((1, width))
    k_radius = (width - 1)//2

    for x in range(-k_radius, k_radius + 1):
        Gx[0, k_radius + x] = (1/(sigma * math.sqrt(2 * math.pi))) * math.pow(math.e, -((x*x)/(2 * sigma * sigma)))

    Gx = Gx / np.sum(Gx)
    Gy = np.transpose(Gx)

    return filter(filter(im, Gx), Gy)