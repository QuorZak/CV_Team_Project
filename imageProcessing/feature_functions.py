import numpy as np
import math
import cv2
from filters import sobel_filter

def get_search_area(array, x, y, window_size, border_extend=False):
    """
    Takes an array, a point and total window size (full width) and returns a snapshot of the area around the point. Doesn't go out of bounds
    """
    if not isinstance(array, np.ndarray):
        print("Error: array is not a numpy array")
        return np.zeros(window_size, window_size)

    search_size = window_size//2 # get the size of the halves of the window area
    slice = array[max(y - search_size, 0) : min(y + search_size, array.shape[0]) + 1,
                  max(x - search_size, 0) : min(x + search_size, array.shape[1]) + 1]

    if border_extend:
        top_padding = max(0, search_size - y)
        bottom_padding = max(0, search_size - (array.shape[0] - 1 - y))
        left_padding = max(0, search_size - x)
        right_padding = max(0, search_size - (array.shape[1] - 1 - x))

        return np.pad(slice, ((top_padding, bottom_padding), (left_padding, right_padding)), "edge")
    else:
        return slice
    

def get_DoG_search_area(array, x, y, scale, window_size):
    blob_radius = math.ceil(scale * 2 ** 0.5) + 2
    #returns a blob_raidus x blob_radius window
    slice = get_search_area(array, x, y, blob_radius, True)
    return cv2.resize(slice.astype(float), (window_size, window_size), interpolation=cv2.INTER_LINEAR)


def gaussian_filter_seperable(im, width=3, sigma=1):
    Gx = np.zeros((1, width))
    k_radius = (width - 1)//2

    for x in range(-k_radius, k_radius + 1):
        Gx[0, k_radius + x] = (1/(sigma * math.sqrt(2 * math.pi))) * math.pow(math.e, -((x*x)/(2 * sigma * sigma)))

    Gx = Gx / np.sum(Gx)
    Gy = np.transpose(Gx)

    return filter(filter(im, Gx), Gy)


def non_max_suppression(input_array, search_size):
    suppressed_array = np.zeros_like(input_array)
    for index, value in np.ndenumerate(input_array):
        x, y = index
        search_area = input_array[max(x-search_size, 0) : min(x+search_size, input_array.shape[0]), # create a snapshot of the search area around the point
                                       max(y-search_size, 0) : min(y+search_size, input_array.shape[1])] # x and y values cut off to not go out of bounds

        if input_array[x, y] == np.max(search_area):
            suppressed_array[x, y] = input_array[x, y]

    return suppressed_array


def simple_thresholding(input_array, threshold):
    thresholded_array = np.zeros_like(input_array)
    for index, value in np.ndenumerate(input_array):
        if value >= threshold:
            thresholded_array[index[0], index[1]] = value
    return thresholded_array


def gradual_cornerness_thresholding(input_array):  # from paper Vino & Sappa
    LOW_THRESHOLD = 3E6
    HIGH_THRESHOLD = 12E6
    
    thresholded_array = np.zeros_like(input_array)
    for (x,y), score in np.ndenumerate(input_array):
        if score >= HIGH_THRESHOLD: # get the obvious strong corners
            thresholded_array[x,y] = score
        elif score >= LOW_THRESHOLD: # find and deal with potential corners
            weight_sum = 0.0        
            corner_sum = 0.0
            adaptive_threshold = 0.0
            weight_sum += score
            if (x-1 != -1): # left of x
                weight_sum += input_array[x-1,y]
            if (x+1 < input_array.shape[0]): # right of x
                weight_sum += input_array[x+1,y]
            if (y-1 != -1): # above x
                weight_sum += input_array[x,y-1]
            if (y+1 < input_array.shape[1]): # below x
                weight_sum += input_array[x][y+1]

            # 4 corners is less easy, all could be edge of matrix, don't want to round inwards or go off edge
            corner_sum = 0.0
            if (x-1 != -1 and y-1 != -1): # top left (-,-)
                corner_sum += input_array[x-1,y-1]           
            elif (x+1 < input_array.shape[0] and y-1 != -1): # top right (+,-)
                corner_sum += input_array[x+1,y-1]            
            elif (x-1 != -1 and y+1 < input_array.shape[1]): # bottom left (-,+)
                corner_sum += input_array[x-1,y+1]  
            elif (x+1 < input_array.shape[0] and y+1 < input_array.shape[1]): # bottom right (+,+)
                corner_sum += input_array[x+1,y+1]
            weight_sum += corner_sum/math.sqrt(2)
            
            normalised_score = weight_sum/score
            
            if (score < 4E6):
                adaptive_threshold = 2.99
            else:
                adaptive_threshold = 1.99 + ((1/(49E12)) * ((score-11E6)**2)) # emperical equation from Vino & Sappa: 1.99 + 1/(49x10^12)[score - 11x10^6]^2
            
            if (normalised_score > adaptive_threshold):
                thresholded_array[x,y] = score
                
    return thresholded_array


def ncc_matching(left_array, right_array, left_corners, right_corners, window_size=15):
    left_array = np.array(left_array)
    right_array = np.array(right_array)
    ncc_matches_list = []
    
    unmatched_right_points_indices = list(range(len(right_corners)))
    for left_point in left_corners:
        best_ncc = -1
        second_best_ncc = -1
        best_match_index = None

        x, y = left_point
        left_window = get_search_area(left_array, x, y, window_size, border_extend=True)
        left_minus_mean = left_window-np.mean(left_window)
        left_ncc = left_minus_mean/np.linalg.norm(left_minus_mean)

        #if there are no right points left to match with
        if not unmatched_right_points_indices: break
        
        for i in unmatched_right_points_indices:
            x, y = right_corners[i]
            right_window = get_search_area(right_array, x, y, window_size, border_extend=True)
            right_minus_mean = right_window-np.mean(right_window)
            right_ncc = right_minus_mean/np.linalg.norm(right_minus_mean)

            ncc_value = np.sum(left_ncc * right_ncc)
            
            if ncc_value > best_ncc:
                second_best_ncc = best_ncc
                best_ncc = ncc_value
                best_match_index = i
            elif ncc_value > second_best_ncc:
                second_best_ncc = ncc_value
        
        if second_best_ncc/best_ncc >= 0.9: continue
        ncc_matches_list.append((left_point, right_corners[best_match_index])) # left corner, best right corner
        unmatched_right_points_indices.remove(best_match_index)
        
    return ncc_matches_list


def improved_ncc_matching(left_array, right_array, left_corners, right_corners, window_size=15, DoG_features=False):
    # do some pre-processing
    left_array = np.array(left_array)
    right_array = np.array(right_array)

    ncc_left_list = []
    ncc_right_list = []
    ncc_matches_list = []

    #compute ncc component for left corners
    for left_corner in left_corners:
        if DoG_features:
            x, y, s = left_corner
            left_window = get_DoG_search_area(left_array, x, y, s, window_size)
        else:
            x, y = left_corner
            left_window = get_search_area(left_array, x, y, window_size, border_extend=True)

        left_minus_mean = left_window-np.mean(left_window)
        left_ncc = left_minus_mean/np.linalg.norm(left_minus_mean)
        ncc_left_list.append(left_ncc)

    #compute ncc component for right corners   
    for right_corner in right_corners:
        if DoG_features:
            x, y, s = right_corner
            right_window = get_DoG_search_area(right_array, x, y, s, window_size)
        else:
            x, y = right_corner
            right_window = get_search_area(right_array, x, y, window_size, border_extend=True)

        right_window = get_search_area(right_array, x, y, window_size, border_extend=True)
        right_minus_mean = right_window-np.mean(right_window)
        right_ncc = right_minus_mean/np.linalg.norm(right_minus_mean)
        ncc_right_list.append(right_ncc)
     
    # now find the matches
    unmatched_right_points_indices = list(range(len(right_corners)))
    for left_point, left_ncc in zip(left_corners, ncc_left_list):
        best_ncc = -1
        second_best_ncc = -1
        best_match_index = None

        #if there are no right points left to match with
        if not unmatched_right_points_indices: break
        
        for i in unmatched_right_points_indices:
            right_ncc = ncc_right_list[i]

            ncc_value = np.sum(left_ncc * right_ncc)
            
            if ncc_value > best_ncc:
                second_best_ncc = best_ncc
                best_ncc = ncc_value
                best_match_index = i
            elif ncc_value > second_best_ncc:
                second_best_ncc = ncc_value
        
        if second_best_ncc/best_ncc >= 0.9: continue
        if best_ncc < 0.7:  continue
        ncc_matches_list.append((left_point, right_corners[best_match_index])) # left corner, best right corner
        unmatched_right_points_indices.remove(best_match_index)
        
    return ncc_matches_list

def l1_distance(hist1, hist2):
    dist = np.linalg.norm(hist1-hist2)
    return dist