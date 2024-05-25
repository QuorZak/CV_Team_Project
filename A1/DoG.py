import numpy as np
from filters import gaussian_filter

from matplotlib import pyplot
from matplotlib.patches import Circle

from filters import gaussian_filter
from timeit import default_timer as timer

def DoG_nms(im, window_size, height=3):
    x,y,z = im.shape
    im = np.abs(im)
    half_win = window_size//2
    half_height = height//2
    for z_layer in range(half_height, z-half_height):
        for i in range(half_win, x - half_win):
            for j in range(half_win, y - half_win):
                cube = im[(i-half_win):(i+half_win+1), (j-half_win):(j+half_win+1), (z_layer-half_height):(z_layer+half_height+1)]
                im[(i-half_win):(i+half_win+1), (j-half_win):(j+half_win+1), (z_layer-half_height):(z_layer+half_height+1)] = cube * (cube == np.max(cube))
    
    return im

def DoG(im, sigma=1, k=1.6,n=6, nms_window=13, nms_height=3, max_features=1000):
    im = np.float64(im)
    sigma_list = [k*i*sigma for i in range(1,n)]
    dog_pyramid = np.empty(im.shape + (len(sigma_list)-1,), dtype="float64")

    previous = gaussian_filter(im, None, sigma_list[0])
    for i, sig in enumerate(sigma_list[1:]):
        current = gaussian_filter(im, None, sig)
        dog_pyramid[..., i] = previous - current
        previous = current

    normalisation_factor = 1 / (k - 1)
    dog_pyramid *= normalisation_factor

    maxima = DoG_nms(dog_pyramid, nms_window, nms_height)

    score_feature_list = [(score, coords[0], coords[1], coords[2]) for coords, score in list(np.ndenumerate(maxima)) if score != 0]
    score_feature_list.sort(reverse=True, key=lambda x: x[0]) #Lambda function explicitly tells it to sort on the first element even though I think it does this automatically

    score_feature_list = score_feature_list[:max_features]
    #print("There are", len(score_feature_list), "blobs")
    score_feature_list = [(row, col, sigma_list[sig+1]) for (_,col, row, sig) in score_feature_list]

    return score_feature_list

def show_DoG(px_array_left, px_array_right, sigma=1, k=1.6,n=6, nms_window=13, nms_height=3, max_features=1000, single_image=False):
    start = timer()
    left_blobs = DoG(px_array_left, sigma=sigma, k=k,n=n, nms_window=nms_window, nms_height=nms_height, max_features=max_features)
    right_blobs = DoG(px_array_right, sigma=sigma, k=k,n=n, nms_window=nms_window, nms_height=nms_height, max_features=max_features)
    end = timer()
    print("elapsed time of blob detection for both images: ", end - start)

    if not single_image:
        fig1, axs1 = pyplot.subplots(1, 2)
        axs1[0].set_title('DoG response left overlaid on orig image')
        axs1[1].set_title('DoG response right overlaid on orig image')
        axs1[0].imshow(px_array_left, cmap='gray')
        axs1[1].imshow(px_array_right, cmap='gray')

        for corner_point in left_blobs:
            circle = Circle(corner_point[:2], corner_point[2] * 2**0.5, color='red', linewidth=1, fill=False)
            axs1[0].add_patch(circle)

        for corner_point in right_blobs:
            circle = Circle(corner_point[:2], corner_point[2] * 2**0.5, color='red', linewidth=1, fill=False)
            axs1[1].add_patch(circle)

    else:
        fig1, axs1 = pyplot.subplots()
        fig1.suptitle(f'Sigma={sigma}, k={k}, n={n}, nms_window={nms_window}x{nms_window}x{nms_height}, max_blobs={max_features}')

        axs1.set_title('DoG response overlaid on orig image')
        axs1.imshow(px_array_left, cmap='gray')
        for corner_point in left_blobs:
            circle = Circle(corner_point[:2], corner_point[2] * 2**0.5, color='red', linewidth=1, fill=False)
            axs1.add_patch(circle)

    pyplot.show()