"""
Description:
This program implements the structured light pipeline with graycode and phase shift pattern.
"""
import os
import cv2
import time
import numpy as np
import numba
from numba import prange
from stereo_rectify import StereoRectify

### parameters
phase_decoding_unvalid_thres = 5  # if the diff of pixel in an inversed pattern(has pi phase shift) is smaller than this, consider it's unvalid;
                                  # this value is a balance between valid pts rates and error points rates
                                  # e.g., 1, 2, 5 for low-expo real captured images; 20, 30, 40 for normal expo rendered images.
remove_possibly_outliers_when_matching = True
depth_cutoff_near, depth_cutoff_far = 0.1, 2.0  # depth cutoff
depth_filter_max_distance = 0.005 # about 5-7 times of resolution per pxiel
depth_filter_minmum_points_in_checking_range = 2  # including the point itsself, will also add a ratio of width // 400
use_depth_avg_filter = True
depth_avg_filter_max_length = 2   # 4, from 0 - 6
depth_avg_filter_unvalid_thres = 0.001  # 0.002

roughly_projector_area_in_image = 0.8  # the roughly prjector area in image / image width, e.g., 0.75, 1.0, 1.25
phsift_pattern_period_per_pixel = 10.0  # normalize the index. porjected pattern res width is 1280; 7 graycode pattern = 2^7 = 128 phase shift periods; 1290/128=10 
default_image_seq_start_index = 24  # in some datasets, (0, 24) are for pure gray code solutions 

use_parallel_computing = False
save_mid_res_for_visulize = False
visulize_res = True

@numba.jit  ((numba.uint8[:,:,:], numba.uint8[:,:], numba.uint8[:,:], numba.int64, numba.int64, numba.int64, numba.int16[:,:], numba.float32),nopython=True, parallel=use_parallel_computing, cache=True)
def gray_decode(src, imgs_thresh, valid_map, image_num, height, width, img_index, unvalid_thres):
    for h in prange(height):
        for w in range(width):
            if valid_map[h,w] == 0:
                img_index[h,w] = -1
                continue
            bin_code = 0
            current_bin_code_bit = 0
            for i in range(0, image_num):
                if src[i][h,w]>=imgs_thresh[h,w]+unvalid_thres:
                    current_bin_code_bit = current_bin_code_bit ^ 1
                elif src[i][h,w]<=imgs_thresh[h,w]-unvalid_thres:
                    current_bin_code_bit = current_bin_code_bit ^ 0
                else:
                    bin_code = -1
                    break
                bin_code += (current_bin_code_bit <<  (image_num-1-i))
            img_index[h,w] = bin_code

"""
For 4 step phaseshift, phi = np.arctan2(I4-I2, I3-I1), from -pi to pi
"""
@numba.jit  ((numba.uint8[:,:,:], numba.int64, numba.int64, numba.float32[:,:], numba.int16[:,:], numba.float32),nopython=True, parallel=use_parallel_computing, cache=True)
def phase_shift_decode(src, height, width, img_phase, img_index, unvalid_thres):
    pi = 3.14159265358979
    unvalid_thres_diff = unvalid_thres
    outliers_checking_thres_diff = 4 * (1+unvalid_thres_diff) # above this, will skip outlier checking
    for h in prange(height):
        for w in range(width):
            if img_index[h,w] == -1:
                img_phase[h,w] = np.nan
                continue
            i1, i2, i3, i4 = 1.0 * src[0][h,w], 1.0 * src[1][h,w], 1.0 * src[2][h,w], 1.0 * src[3][h,w]  # force numba use float
            unvalid_flag = (abs(i4 - i2) <= unvalid_thres_diff and abs(i3 - i1) <= unvalid_thres_diff)
            need_outliers_checking_flag = (abs(i4 - i2) <= outliers_checking_thres_diff and abs(i3 - i1) <= outliers_checking_thres_diff)
            if unvalid_flag:
                img_phase[h,w] = np.nan
                continue
            phase = - np.arctan2(i4-i2, i3-i1) + pi
            # phase_2 = - np.arctan2(i1-i3, i4-i2) + pi
            phase_main_index = img_index[h,w] // 2
            phase_sub_index = img_index[h,w] % 2
            if phase_sub_index == 0 and phase > pi*1.5:
                phase = phase - 2*pi  # 0
            if phase_sub_index == 1 and phase < pi*0.5:
                phase = phase + 2*pi  # 2*pi
            img_phase[h,w] = phase_main_index * phsift_pattern_period_per_pixel + (phase * phsift_pattern_period_per_pixel / (2*pi))
            img_index[h,w] = not need_outliers_checking_flag # reuse img_index as belief map


@numba.jit  ((numba.float32[:,:], numba.float32[:,:], numba.float32[:,:], numba.int64, numba.int64, numba.float32[:,:], numba.float32[:,:]),nopython=True, parallel=use_parallel_computing, cache=True)
def rectify_phase(img_phase, rectify_map_x, rectify_map_y, height, width, rectified_img_phase, sub_pixel_map_x):
    # rectify_map is, for each pixel (u,v) in the destination (corrected and rectified) image, the corresponding coordinates in the source image (that is, in the original image from camera)
    for h in prange(height):
        for w in range(width):
            src_x, src_y = rectify_map_x[h,w], rectify_map_y[h,w]
            ### if use interpolation, only apply to y axis. The precision will be a little bit higher
            # src_x_round, src_y_int = round(src_x), int(src_y)
            # upper = img_phase[src_y_int, src_x_round]
            # lower = img_phase[src_y_int+1, src_x_round]
            # diff = lower - upper
            # if diff >= 1.0 or np.isnan(diff):
            #     rectified_img_phase[h,w] = img_phase[round(src_y), src_x_round]
            #     # if np.isnan(upper): rectified_img_phase[h,w] = lower
            #     # else: rectified_img_phase[h,w] = upper
            # else:
            #     inter_value = upper + diff * (src_y-src_y_int)
            #     rectified_img_phase[h,w] = inter_value
            rectified_img_phase[h,w] = img_phase[round(src_y), round(src_x)]
            sub_pixel_map_x[h,w] = w + (round(src_x) - src_x)

@numba.jit  ((numba.int16[:,:], numba.float32[:,:], numba.float32[:,:], numba.int64, numba.int64, numba.int16[:,:]),nopython=True, parallel=use_parallel_computing, cache=True)
def rectify_belief_map(img, rectify_map_x, rectify_map_y, height, width, rectified_img):
    for h in prange(height):
        for w in range(width):
            src_x, src_y = rectify_map_x[h,w], rectify_map_y[h,w]
            rectified_img[h,w] = img[round(src_y), round(src_x)]

@numba.jit  ((numba.float32[:,:], numba.int64,numba.int64, numba.float32[:,:],numba.float32[:,:], numba.float32,numba.float32,numba.float32, numba.float32[:,:],numba.float32[:,:],numba.int16[:,:], numba.int16[:,:] ), nopython=True, parallel=use_parallel_computing, nogil=True, cache=True)
def gen_depth_from_index_matching(depth_map, height,width, img_index_left,img_index_right, baseline,dmap_base,fx, img_index_left_sub_px,img_index_right_sub_px, belief_map_l, belief_map_r):
    area_scale = 1.333 * roughly_projector_area_in_image
    max_allow_pixel_per_index = 1.25 + area_scale * width / 1280.0
    max_index_offset_when_matching = 1.3 * (1280.0 / width)  # typical condition: a lttle larger than 2.0 for 640, 1.0 for 1280, 0.5 for 2560
    max_index_offset_when_matching_ex = max_index_offset_when_matching * 1.5
    right_corres_point_offset_range = (width // 128) * area_scale
    check_outliers = remove_possibly_outliers_when_matching
    for h in prange(height):
        line_r = img_index_right[h,:]
        line_l = img_index_left[h,:]
        last_right_corres_point = -1
        for w in range(width):
            if np.isnan(line_l[w]):   # unvalid
                last_right_corres_point = -1
                continue
            ## find the possible corresponding points in right image
            cnt_l, cnt_r = 0, 0
            most_corres_pts_l, most_corres_pts_r = -1, -1
            if last_right_corres_point > 0:
                checking_left_edge = last_right_corres_point - right_corres_point_offset_range
                checking_right_edge = last_right_corres_point + right_corres_point_offset_range
                if checking_left_edge <=0: checking_left_edge=0
                if checking_right_edge >=width: checking_right_edge=width
            else:
                checking_left_edge, checking_right_edge = 0, width
            for i in range(checking_left_edge, checking_right_edge):
                if np.isnan(line_r[i]): continue
                if line_l[w]-max_index_offset_when_matching <= line_r[i] <= line_l[w]:
                    if most_corres_pts_l == -1: most_corres_pts_l = i
                    elif line_l[w] - line_r[i] <= line_l[w] - line_r[most_corres_pts_l]: most_corres_pts_l = i
                    cnt_l += 1
                if line_l[w] <= line_r[i] <= line_l[w]+max_index_offset_when_matching:
                    if most_corres_pts_r == -1: most_corres_pts_r = i
                    elif line_r[i] - line_l[w] <= line_r[most_corres_pts_r] - line_l[w]: most_corres_pts_r = i
                    cnt_r += 1
            if cnt_l == 0 and cnt_r == 0:  # expand the searching range and try again
                for i in range(width):
                    if np.isnan(line_r[i]): continue
                    if line_l[w]-max_index_offset_when_matching_ex <= line_r[i] <= line_l[w]:
                        if most_corres_pts_l == -1: most_corres_pts_l = i
                        elif line_l[w] - line_r[i] <= line_l[w] - line_r[most_corres_pts_l]: most_corres_pts_l = i
                        cnt_l += 1
                    if line_l[w] <= line_r[i] <= line_l[w]+max_index_offset_when_matching_ex:
                        if most_corres_pts_r == -1: most_corres_pts_r = i
                        elif line_r[i] - line_l[w] <= line_r[most_corres_pts_r] - line_l[w]: most_corres_pts_r = i
                        cnt_r += 1
            if cnt_l == 0 and cnt_r == 0: continue
            if most_corres_pts_l == -1: most_corres_pts_l = most_corres_pts_r
            elif most_corres_pts_r == -1: most_corres_pts_r = most_corres_pts_l
            left_pos, right_pos = line_r[most_corres_pts_l], line_r[most_corres_pts_r]
            left_value, right_value = img_index_right_sub_px[h, most_corres_pts_l], img_index_right_sub_px[h, most_corres_pts_r]
            if cnt_l != 0 and cnt_r != 0:
                # interpo for corresponding right point
                if right_pos-left_pos != 0: inter_value = left_value + (right_value-left_value) * (line_l[w]-left_pos)/(right_pos-left_pos)
                else: inter_value = left_value
                w_r = inter_value
            elif cnt_l != 0:
                w_r = left_value
            else:
                w_r = right_value
            # check possiblely outliers using max_allow_pixel_per_index and belief_map
            outliers_flag = False
            if check_outliers and belief_map_r[h,round(w_r)]==0:
                if abs(most_corres_pts_l-w_r) >= max_allow_pixel_per_index: outliers_flag = True
                if abs(most_corres_pts_r-w_r) >= max_allow_pixel_per_index: outliers_flag = True
            if outliers_flag: continue
            last_right_corres_point = round(w_r)
            # get left index
            w_l = img_index_left_sub_px[h, w]
            # check possiblely left outliers
            if check_outliers and belief_map_l[h,w]==0:
                for i in range(width):
                    if line_l[w]-max_index_offset_when_matching_ex <= line_l[i] <= line_l[w]+max_index_offset_when_matching_ex:
                        if abs(w-i) > max_allow_pixel_per_index: outliers_flag = True
            if outliers_flag: continue
            ## stereo diff and depth
            stereo_diff = dmap_base + w_l - w_r
            if stereo_diff > 0.000001:
                depth = fx * baseline / stereo_diff
                if depth_cutoff_near < depth < depth_cutoff_far:
                    depth_map[h, w] = depth

@numba.jit  ((numba.float32[:,:], numba.float32[:,:], numba.int64,numba.int64, numba.float32[:,:]), nopython=True, parallel=use_parallel_computing, nogil=True, cache=True)
def optimize_dmap_using_sub_pixel_map(depth_map, optimized_depth_map, height,width, img_index_left_sub_px):
    # interpo for depth map using sub_pixel
    # this does not improve a lot on rendered data because no distortion and less stereo rectify for left camera, but useful for real captures
    for h in prange(height):
        for w in range(width):
            left_value, right_value = 0, 0
            real_pos_for_current_depth = img_index_left_sub_px[h,w]
            if depth_map[h,w] <= 0.00001:
                if depth_map[h,w-1] >= 0.00001 and depth_map[h,w+1] >= 0.00001:
                    right_pos = img_index_left_sub_px[h,w+1]
                    right_value = depth_map[h,w+1]
                    left_pos = img_index_left_sub_px[h,w-1]
                    left_value = depth_map[h,w-1]
            elif real_pos_for_current_depth >= w:
                right_pos = real_pos_for_current_depth
                right_value = depth_map[h,w]
                left_pos = img_index_left_sub_px[h,w-1]
                left_value = depth_map[h,w-1]
            else:
                right_pos = img_index_left_sub_px[h,w+1]
                right_value = depth_map[h,w+1]
                left_pos = real_pos_for_current_depth
                left_value = depth_map[h,w]
            if left_value >= 0.00001 and right_value >= 0.00001:
                inter_value = left_value + (right_value-left_value) * (w-left_pos)/(right_pos-left_pos)
                optimized_depth_map[h,w] = inter_value

@numba.jit  ((numba.float32[:,:], numba.float32[:,:], numba.int64, numba.int64, numba.float32[:,:]), nopython=True, parallel=use_parallel_computing, nogil=True, cache=True)
def depth_filter(depth_map, depth_map_raw, height, width, camera_kp):
    # a point could be considered as not flying when: points in checking range below max_distance > minmum num 
    max_distance = depth_filter_max_distance
    minmum_point_num_in_range = depth_filter_minmum_points_in_checking_range + (width // 400) * (width // 400)
    checking_range_in_meter = max_distance * 1.2
    checking_range_limit = width // 50
    fx, cx, fy, cy = camera_kp[0][0], camera_kp[0][2], camera_kp[1][1], camera_kp[1][2]
    for h in prange(height):
        w = 0
        while w < width:
            if depth_map_raw[h,w] != 0:
                point_x = depth_map_raw[h,w] * (w - cx) / fx
                point_y = depth_map_raw[h,w] * (h - cy) / fy
                checking_range_in_pix_x = (int)(checking_range_in_meter * fx / depth_map_raw[h,w])
                checking_range_in_pix_y = (int)(checking_range_in_meter * fy / depth_map_raw[h,w])
                checking_range_in_pix_x = min(checking_range_in_pix_x, checking_range_limit)
                checking_range_in_pix_y = min(checking_range_in_pix_y, checking_range_limit)
                is_not_flying_point_flag = 0
                for i in range(h-checking_range_in_pix_y, min(height, h+checking_range_in_pix_y+1)):
                    for j in range(w-checking_range_in_pix_x, min(width, w+checking_range_in_pix_x+1)):
                        curr_x = depth_map_raw[i,j] * (j - cx) / fx
                        curr_y = depth_map_raw[i,j] * (i - cy) / fy
                        distance = np.square(curr_x - point_x) + np.square(curr_y - point_y) + np.square(depth_map_raw[h,w] - depth_map_raw[i,j])
                        if distance < np.square(max_distance): is_not_flying_point_flag += 1
                if is_not_flying_point_flag <= minmum_point_num_in_range: # unvalid the point
                    depth_map[h,w] = 0
                elif is_not_flying_point_flag >= checking_range_in_pix_x * checking_range_in_pix_y - 1:
                    w += checking_range_in_pix_x
                    continue
            w += 1

@numba.jit  ((numba.float32[:,:], numba.int64, numba.int64), nopython=True, parallel=use_parallel_computing, nogil=True, cache=True)
def depth_avg_filter(depth_map, height, width):
    filter_max_length = depth_avg_filter_max_length
    filter_weights = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1])
    filter_thres = depth_avg_filter_unvalid_thres
    # horizontal
    for h in prange(height): 
        for w in range(width):
            if depth_map[h,w] != 0:
                left_weight, right_weight, depth_sum = 0.0, 0.0, depth_map[h,w]*filter_weights[0]
                for i in range(1, filter_max_length+1):
                    l_idx, r_idx = w-i, w+i
                    stop_flag = False
                    if(depth_map[h,l_idx] != 0 and depth_map[h,r_idx] != 0 and l_idx > 0 and r_idx < width and \
                        abs(depth_map[h,l_idx] - depth_map[h,w]) < filter_thres and abs(depth_map[h,r_idx] - depth_map[h,w]) < filter_thres):
                        left_weight += filter_weights[i]
                        right_weight += filter_weights[i]
                        depth_sum += (depth_map[h,r_idx] + depth_map[h,l_idx]) * filter_weights[i]
                    else:
                        stop_flag = True
                        break
                if not stop_flag: depth_map[h,w] = depth_sum / (filter_weights[0] + left_weight + right_weight)
    # vertical
    for w in prange(width):
        for h in range(height): 
            if depth_map[h,w] != 0:
                left_weight, right_weight, depth_sum = 0.0, 0.0, depth_map[h,w]*filter_weights[0]
                for i in range(1, filter_max_length+1):
                    l_idx, r_idx = h-i, h+i
                    stop_flag = False
                    if(depth_map[l_idx,w] != 0 and depth_map[r_idx,w] != 0 and l_idx > 0 and r_idx < height and \
                        abs(depth_map[l_idx,w] - depth_map[h,w]) < filter_thres and abs(depth_map[r_idx,w] - depth_map[h,w]) < filter_thres):
                        left_weight += filter_weights[i]
                        right_weight += filter_weights[i]
                        depth_sum += (depth_map[r_idx,w] + depth_map[l_idx,w]) * filter_weights[i]
                    else:
                        stop_flag = True
                        break
                if not stop_flag: depth_map[h,w] = depth_sum / (filter_weights[0] + left_weight + right_weight)


### the index decoding part
global_reading_img_time = 0
img_phase = None # will be faster as global variable(will not free mem every call)
img_index = None
def index_decoding_from_images(image_path, appendix, rectifier, res_path=None, images=None):
    global global_reading_img_time, img_phase, img_index
    unvalid_thres = 0
    save_mid_res = save_mid_res_for_visulize
    image_seq_start_index = default_image_seq_start_index
    start_time = time.time()
    if images is None:
        fname = image_path + str(image_seq_start_index) + appendix
        if not os.path.exists(fname): image_seq_start_index = 0
        ### read projector fully open and fully close images
        prj_area_posi = cv2.imread(image_path + str(image_seq_start_index) + appendix, cv2.IMREAD_UNCHANGED)
        prj_area_nega = cv2.imread(image_path + str(image_seq_start_index+1) + appendix, cv2.IMREAD_UNCHANGED)
        ### read gray code and phase shift images
        images_graycode = []
        for i in range(image_seq_start_index+2, image_seq_start_index+10):  # (0, 24) for pure gray code solutions in dataset
            fname = image_path + str(i) + appendix
            if not os.path.exists(fname): break
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            images_graycode.append(img)
        images_phsft = []
        for i in range(image_seq_start_index+10, image_seq_start_index+14):  # phase shift patern 
            fname = image_path + str(i) + appendix
            if not os.path.exists(fname): break
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            images_phsft.append(img)
    else:
        prj_area_posi, prj_area_nega = images[image_seq_start_index], images[image_seq_start_index+1]
        images_graycode = images[image_seq_start_index+2:image_seq_start_index+10] # gray code posi images
        images_phsft = images[image_seq_start_index+10:image_seq_start_index+14] # phase shift images
    prj_valid_map = prj_area_posi - prj_area_nega
    if rectifier.remap_x_left_scaled is None: _ = rectifier.rectify_image(prj_area_posi, interpolation=cv2.INTER_NEAREST)  # to build the internal LUT map
    thres, prj_valid_map_bin = cv2.threshold(prj_valid_map, unvalid_thres, 255, cv2.THRESH_BINARY)
    if img_phase is None:
        img_phase = np.empty_like(prj_valid_map, dtype=np.float32)
        img_index = np.empty_like(img_phase, dtype=np.int16)
    # print("read images and rectfy map: %.3f s" % (time.time() - start_time))
    global_reading_img_time += (time.time() - start_time)
    ### decoding
    start_time = time.time()
    src_imgs = np.array(images_graycode)
    images_phsft_src = np.array(images_phsft)
    rectified_img_phase = np.empty_like(img_phase, dtype=np.float32)
    rectified_belief_map = np.empty_like(img_phase, dtype=np.int16)
    sub_pixel_map = np.empty_like(img_phase, dtype=np.float32)
    height, width = images_graycode[0].shape[:2]
    print("build ndarrays for decoding: %.3f s" % (time.time() - start_time))
    
    start_time = time.time()
    gray_decode(src_imgs, prj_area_posi//2 + prj_area_nega//2, prj_valid_map_bin, len(images_graycode), height,width, img_index, unvalid_thres)
    print("gray code decoding: %.3f s" % (time.time() - start_time))
    if save_mid_res and res_path is not None:
        mid_res_corse_gray_index_raw = img_index // 2
        mid_res_corse_gray_index = np.clip(mid_res_corse_gray_index_raw * 80 % 255, 0, 255).astype(np.uint8)
        cv2.imwrite(res_path + "/mid_res_corse_gray_index" + appendix, mid_res_corse_gray_index)
  
    start_time = time.time()
    phase_shift_decode(images_phsft_src, height,width, img_phase, img_index, phase_decoding_unvalid_thres)
    belief_map = img_index # img_index reused as belief_map when phase_shift_decoding
    print("phase decoding: %.3f s" % (time.time() - start_time))
    
    start_time = time.time()
    ### rectify the decoding res, accroding to left or right
    if appendix == '_l.bmp': rectify_map_x, rectify_map_y, camera_kd = rectifier.remap_x_left_scaled, rectifier.remap_y_left_scaled, rectifier.rectified_camera_kd_l
    else: rectify_map_x, rectify_map_y, camera_kd = rectifier.remap_x_right_scaled, rectifier.remap_y_right_scaled, rectifier.rectified_camera_kd_r
    rectify_belief_map(belief_map, rectify_map_x, rectify_map_y, height,width, rectified_belief_map)
    rectify_phase(img_phase, rectify_map_x, rectify_map_y, height,width, rectified_img_phase, sub_pixel_map)
    print("rectify: %.3f s" % (time.time() - start_time))

    if save_mid_res:
        mid_res_wrapped_phase = (img_phase - mid_res_corse_gray_index_raw * phsift_pattern_period_per_pixel) / phsift_pattern_period_per_pixel
        mid_res_wrapped_phase = (mid_res_wrapped_phase * 254.0)
        cv2.imwrite(res_path + "/mid_res_wrapped_phase"+appendix, mid_res_wrapped_phase.astype(np.uint8))

    return rectified_belief_map, rectified_img_phase, camera_kd, sub_pixel_map


def run_stru_li_pipe(pattern_path, res_path, rectifier=None, images=None):
    global global_reading_img_time
    if rectifier is None: rectifier = StereoRectify(scale=1.0, cali_file=pattern_path+'calib.yml')
    if images is not None: images_left, images_right = images[0], images[1]
    else: images_left, images_right = None, None
    ### Rectify and Decode 
    pipe_start_time = start_time = time.time()
    belief_map_left, img_index_left, camera_kd_l, img_index_left_sub_px = index_decoding_from_images(pattern_path, '_l.bmp', rectifier=rectifier, res_path=res_path, images=images_left)
    belief_map_right, img_index_right, camera_kd_r, img_index_right_sub_px = index_decoding_from_images(pattern_path, '_r.bmp', rectifier=rectifier, res_path=res_path, images=images_right)
    print("- Read image and decoding in total: %.3f s" % (time.time() - start_time))
    # get camera parameters
    fx = camera_kd_l[0][0]
    cx, cx_r = camera_kd_l[0][2], camera_kd_r[0][2]
    dmap_base = cx_r - cx
    cam_transform = np.array(rectifier.T)[:,0]
    height, width = img_index_left.shape[:2]
    baseline = np.linalg.norm(cam_transform)  # * ( 0.8/(0.8+0.05*0.001) )  # = 0.9999375039060059
    ### Infer DepthMap from Index Matching
    unoptimized_depth_map = np.empty_like(img_index_left, dtype=np.float32)
    depth_map = np.empty_like(img_index_left, dtype=np.float32)
    start_time = time.time()
    gen_depth_from_index_matching(unoptimized_depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right)
    print("index matching and depth map generating: %.3f s" % (time.time() - start_time))
    start_time = time.time()
    optimize_dmap_using_sub_pixel_map(unoptimized_depth_map, depth_map, height,width, img_index_left_sub_px)
    print("subpix optimize: %.3f s" % (time.time() - start_time))
    ### Run Depth Map Filter
    depth_map_raw = depth_map.copy()  # save raw depth map
    start_time = time.time()
    depth_filter(depth_map, depth_map_raw, height, width, camera_kd_l.astype(np.float32))
    print("flying point filter: %.3f s" % (time.time() - start_time))
    if use_depth_avg_filter:
        start_time = time.time()
        depth_avg_filter(depth_map, height, width)
        print("depth avg filter: %.3f s" % (time.time() - start_time))
    print("- Total time: %.3f s" % (time.time() - pipe_start_time))
    print("- Total time except reading imgs: %.3f s" % (time.time() - pipe_start_time - global_reading_img_time))
    global_reading_img_time = 0
    ### Save Mid Results for visualizing
    if save_mid_res_for_visulize:   
        depth_map_uint16 = depth_map * 30000
        cv2.imwrite(res_path + '/depth_alg2.png', depth_map_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        depth_map_raw_uint16 = depth_map_raw * 30000
        cv2.imwrite(res_path + '/depth_alg2_raw.png', depth_map_raw_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        belief_map_left = np.clip(belief_map_left * 255, 0, 255).astype(np.uint8)
        belief_map_right = np.clip(belief_map_right * 255, 0, 255).astype(np.uint8)
        images_phsft_left_v = np.clip(img_index_left/4.0, 0, 255).astype(np.uint8)
        images_phsft_right_v = np.clip(img_index_right/4.0, 0, 255).astype(np.uint8)
        cv2.imwrite(res_path + "/belief_map_left.png", belief_map_left)
        cv2.imwrite(res_path + "/belief_map_right.png", belief_map_right)
        cv2.imwrite(res_path + "/ph_correspondence_l.png", images_phsft_left_v)
        cv2.imwrite(res_path + "/ph_correspondence_r.png", images_phsft_right_v)
    ### Prepare results
    depth_map_mm = depth_map * 1000
    global default_image_seq_start_index
    if images is None:
        fname = pattern_path + str(default_image_seq_start_index) + "_l.bmp"
        if not os.path.exists(fname): default_image_seq_start_index = 0
        gray_img = cv2.imread(pattern_path + str(default_image_seq_start_index) + "_l.bmp", cv2.IMREAD_UNCHANGED)
    else:
        gray_img = images_left[default_image_seq_start_index]
    gray_img = rectifier.rectify_image(gray_img)
    return gray_img, depth_map_mm, camera_kd_l


# test with existing pattern example: 
#   win: python structured_light.py pattern_examples\struli_test1\
#   linux: python structured_light.py pattern_examples/struli_test1/
if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("run with args 'pattern_path'")
    image_path = sys.argv[1]

    ### build up runing parameters and run
    res_path = image_path + r'\res' if sys.platform == 'win32' else image_path + '/res'
    if not os.path.exists(res_path): os.system("mkdir " + res_path)
    gray, depth_map_mm, camera_kp = run_stru_li_pipe(image_path, res_path)
    
    def report_depth_error(depth_img, depth_gt):
        gray_img = cv2.imread(image_path + str(default_image_seq_start_index) + "_l.bmp", cv2.IMREAD_UNCHANGED).astype(np.int16)
        gray_img_dark = cv2.imread(image_path + str(default_image_seq_start_index+1) + "_l.bmp", cv2.IMREAD_UNCHANGED).astype(np.int16)
        projector_area_diff = gray_img - gray_img_dark
        projector_area = np.where(projector_area_diff > phase_decoding_unvalid_thres)
        valid_points = np.where(depth_img>=1.0)
        error_img = (depth_img - depth_gt)[valid_points]

        pxiel_num = depth_img.shape[0]*depth_img.shape[1]
        valid_points_num, valid_points_gt_num = len(valid_points[0]), len(projector_area[0])
        # # error below 10.0mm
        error_img = error_img[np.where((error_img<10.0)&(error_img>-10.0))]
        print("valid points rate below 10mm: " + str(error_img.shape[0]) + "/" + str(valid_points_gt_num) + ", " + str(100*error_img.shape[0]/valid_points_gt_num)+"%")
        print("average_error(mm):" + str(np.average(abs(error_img))))
        # error below 1.0mm
        error_img = error_img[np.where((error_img<1.0)&(error_img>-1.0))]
        print("valid points rate below 1mm: " + str(error_img.shape[0]) + "/" + str(valid_points_gt_num) + ", " + str(100*error_img.shape[0]/valid_points_gt_num)+"%")
        print("average_error(mm):" + str(np.average(abs(error_img))))
        # error below 0.25mm
        error_img = error_img[np.where((error_img<0.25)&(error_img>-0.25))]
        print("valid points rate below 0.25mm: " + str(error_img.shape[0]) + "/" + str(valid_points_gt_num) + ", " + str(100*error_img.shape[0]/valid_points_gt_num)+"%")
        print("average_error(mm):" + str(np.average(abs(error_img))))

        # write error map
        error_map_thres = 0.25
        unvalid_points = np.where(depth_img<=1.0)
        diff = depth_img - depth_gt
        depth_img_show_error = (depth_img * 255.0 / 2000.0).astype(np.uint8)
        error_part = depth_img_show_error.copy()
        error_part[np.where((diff>error_map_thres)|(diff<-error_map_thres))] = 255
        error_part[unvalid_points] = 0
        depth_img_show_error = cv2.cvtColor(depth_img_show_error, cv2.COLOR_GRAY2RGB)
        depth_img_show_error[:,:,2] = error_part
        cv2.imwrite(res_path + "/error_map.png", depth_img_show_error)

    if os.path.exists(image_path + "depth_gt.exr"):
        gt_depth = cv2.imread(image_path + "depth_gt.exr", cv2.IMREAD_UNCHANGED)[:,:,0]
        gt_depth = gt_depth * 1000.0  # scale to mili-meter
        rectifier = StereoRectify(scale=1.0, cali_file=image_path+'calib.yml')
        gt_depth_rectified = rectifier.rectify_image(gt_depth) #, interpolation=cv2.INTER_NEAREST
        report_depth_error(depth_map_mm, gt_depth_rectified)
    
    ### build point cloud and visualize
    if visulize_res:
        import open3d as o3d
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(gray.astype(np.uint8)),
            o3d.geometry.Image(depth_map_mm.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=6000.0)
        h, w = gray.shape[:2]
        fx, fy, cx, cy = camera_kp[0][0], camera_kp[1][1], camera_kp[0][2], camera_kp[1][2]
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy))
        # save point cloud
        o3d.io.write_point_cloud(res_path + "/points.ply", pcd, write_ascii=False, compressed=False)
        print("ply res saved to:" + res_path)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.translate(np.zeros(3), relative=False)
        o3d.visualization.draw(geometry=pcd, width=1600, height=900, point_size=1)
