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
gray_decoding_unvalid_thres = 0
phase_decoding_unvalid_thres = 2   # 1,2, 5 for low-expo real captured images; 20, 30 for normal expo rendered images. this value is a balance between valid pts rates and error points rates
phsift_pattern_period_per_pixel = 10.0  # normalize the index. porjected pattern res width is 1280; 7 graycode pattern = 2^7 = 128 phase shift periods; 1290/128=10 
numba.config.NUMBA_DEFAULT_NUM_THREADS=8

@numba.jit  ((numba.uint8[:,:,:], numba.uint8[:,:,:], numba.uint8[:,:], numba.int64, numba.int64, numba.int64, numba.int16[:,:], numba.float64),nopython=True, parallel=False, cache=True)
def gray_decode(src, imgs_thresh, valid_map, image_num, height, width, img_index, unvalid_thres):
    for h in prange(height):
        for w in range(width):
            if valid_map[h,w] == 0:
                img_index[h,w] = -1
                continue
            bin_code = 0
            current_bin_code_bit = 0
            for i in range(0, image_num):
                if src[i][h,w]>=imgs_thresh[i][h,w]+unvalid_thres:
                    current_bin_code_bit = current_bin_code_bit ^ 1
                elif src[i][h,w]<=imgs_thresh[i][h,w]-unvalid_thres:
                    current_bin_code_bit = current_bin_code_bit ^ 0
                else:
                    bin_code = -1
                    break
                bin_code += (current_bin_code_bit <<  (image_num-1-i))
            img_index[h,w] = bin_code

"""
For 4 step phaseshift, phi = np.arctan2(I4-I2, I3-I1), from -pi to pi
"""
@numba.jit  ((numba.uint8[:,:,:], numba.uint8[:,:], numba.int64, numba.int64, numba.int64, numba.float32[:,:], numba.int16[:,:], numba.float64),nopython=True, parallel=False, cache=True)
def phase_shift_decode(src, valid_map, image_num, height, width, img_phase, img_index, unvalid_thres):
    pi = 3.14159265358979
    unvalid_thres_diff = phase_decoding_unvalid_thres
    for h in prange(height):
        for w in range(width):
            if img_index[h,w] == -1:
                img_phase[h,w] = np.nan
                continue
            i1, i2, i3, i4 = 1.0 * src[0][h,w], 1.0 * src[1][h,w], 1.0 * src[2][h,w], 1.0 * src[3][h,w]  # force numba use float
            unvalid_flag = (abs(i4 - i2) <= unvalid_thres_diff and abs(i3 - i1) <= unvalid_thres_diff)
            if unvalid_flag:
                img_phase[h,w] = np.nan
                continue
            phase = - np.arctan2(i4-i2, i3-i1) + pi
            phase_main_index = img_index[h,w] // 2
            phase_sub_index = img_index[h,w] % 2  # for the edges
            if phase_sub_index == 0 and phase > pi*1.5:
                phase = 0
            if phase_sub_index == 1 and phase < pi*0.5:
                phase = 2 * pi
            img_phase[h,w] = phase_main_index * phsift_pattern_period_per_pixel + (phase * phsift_pattern_period_per_pixel / (2*pi))

@numba.jit  ((numba.float32[:,:], numba.float32[:,:], numba.float32[:,:], numba.int64, numba.int64, numba.float32[:,:]),nopython=True, parallel=False, cache=True)
def rectify_phase(img_phase, rectify_map_x, rectify_map_y, height, width, rectified_img_phase):
    # rectify_map is, for each pixel (u,v) in the destination (corrected and rectified) image, the corresponding coordinates in the source image (that is, in the original image from camera)
    for h in prange(height):
        for w in range(width):
            src_x, src_y = rectify_map_x[h,w], rectify_map_y[h,w]
            if src_x <= 0.0: src_x = 0.0
            if src_x >= width-1: src_x = width-1
            if src_y <= 0.0: src_y = 0.0
            if src_y >= height-1: src_y = height-1
            rectified_img_phase[h,w] = img_phase[round(src_y), round(src_x)]

@numba.jit  ((numba.float64[:,:], numba.int64,numba.int64, numba.float32[:,:],numba.float32[:,:], numba.float64,numba.float64,numba.float64), nopython=True, parallel=False, nogil=True, cache=True)
def gen_depth_from_index_matching(depth_map, height,width, img_index_left,img_index_right, baseline,dmap_base,fx):
    max_index_offset_when_matching = 1.3 * (1280.0 / width)  # typical condition: a lttle larger than 2.0 for 640, 1.0 for 1280, 0.5 for 2560
    right_corres_point_offset_range = (width // 128)
    for h in prange(height):
        line_r = img_index_right[h,:]
        line_l = img_index_left[h,:]
        possible_points = np.zeros(width, dtype=np.int64)
        last_right_corres_point = -1
        for w in range(width):
            if np.isnan(line_l[w]):   # unvalid
                last_right_corres_point = -1
                continue
            ## find the possible corresponding points in right image
            cnt = 0
            if last_right_corres_point > 0:
                checking_left_edge = last_right_corres_point - right_corres_point_offset_range
                checking_right_edge = last_right_corres_point + right_corres_point_offset_range
                if checking_left_edge <=0: checking_left_edge=0
                if checking_right_edge >=width: checking_left_edge=width
            else:
                checking_left_edge, checking_right_edge = 0, width
            for i in range(checking_left_edge, checking_right_edge):
                if line_l[w]-max_index_offset_when_matching <= line_r[i] <= line_l[w]+max_index_offset_when_matching:
                    possible_points[cnt] = i
                    cnt += 1
            if cnt == 0:
                last_right_corres_point = -1
                continue
            ## find the nearest right index 'w_r' in 'possible_points'
            most_corres_pts = possible_points[0]
            for i in range(cnt): 
                p = possible_points[i]
                if abs(line_r[p] - line_l[w]) <= abs(line_r[most_corres_pts] - line_l[w]):
                    most_corres_pts = p
            last_right_corres_point = most_corres_pts
            ## get stereo diff and depth
            w_l, w_r = w, most_corres_pts
            stereo_diff = dmap_base + w_l - w_r
            if stereo_diff > 0.000001:
                depth_map[h, w] = fx * baseline / stereo_diff

### the index decoding part
def get_image_index(image_path, appendix, rectifier):
    images_posi = []
    images_nega = []
    unvalid_thres = gray_decoding_unvalid_thres
    start_time = time.time()
    ### read projector fully open and fully close images
    prj_area_posi = cv2.imread(image_path + str(24) + appendix, cv2.IMREAD_UNCHANGED)
    prj_area_nega = cv2.imread(image_path + str(25) + appendix, cv2.IMREAD_UNCHANGED)
    prj_valid_map = prj_area_posi - prj_area_nega
    _ = rectifier.rectify_image(prj_area_posi, interpolation=cv2.INTER_NEAREST)  # only to build the internal LUT map
    posi_neg_pattern_avg_thres = (prj_area_posi//2 + prj_area_nega//2)
    thres, prj_valid_map_bin = cv2.threshold(prj_valid_map, unvalid_thres, 255, cv2.THRESH_BINARY)
    ### read gray code and phase shift images
    for i in range(26, 34):  # (0, 24) for pure gray code solutions in dataset
        fname = image_path + str(i) + appendix
        if not os.path.exists(fname): break
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        images_posi.append(img)
        images_nega.append(posi_neg_pattern_avg_thres)
    images_phsft = []
    for i in range(34, 38):  # phase shift patern 
        fname = image_path + str(i) + appendix
        if not os.path.exists(fname): break
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        images_phsft.append(img)
    print("read images using %.3f s" % (time.time() - start_time))
    ### decoding
    height, width = images_posi[0].shape[:2]
    img_index, src_imgs = np.zeros_like(images_posi[0], dtype=np.int16), np.array(images_posi)
    src_imgs_nega = np.array(images_nega)
    start_time = time.time()
    gray_decode(src_imgs, src_imgs_nega, prj_valid_map_bin, len(images_posi), height,width, img_index, unvalid_thres)
    print("gray code index decoding using %.3f s" % (time.time() - start_time))
    img_phase = np.zeros_like(images_posi[0], dtype=np.float32)
    images_phsft_src = np.array(images_phsft)
    start_time = time.time()
    phase_shift_decode(images_phsft_src, prj_valid_map_bin, len(images_posi), height,width, img_phase, img_index, unvalid_thres)
    # rectify image, accroding to left or right
    if appendix == '_l.bmp': rectify_map_x, rectify_map_y, camera_kd = rectifier.remap_x_left_scaled, rectifier.remap_y_left_scaled, rectifier.rectified_camera_kd_l
    else: rectify_map_x, rectify_map_y, camera_kd = rectifier.remap_x_right_scaled, rectifier.remap_y_right_scaled, rectifier.rectified_camera_kd_r
    rectified_img_phase = np.zeros_like(img_phase, dtype=np.float32)
    rectify_phase(img_phase, rectify_map_x, rectify_map_y, height,width, rectified_img_phase)
    print("phase decoding and rectify using %.3f s" % (time.time() - start_time))
    return rectified_img_phase, camera_kd

### the main pipeline
def run_stru_li_pipe(pattern_path, res_path, rectifier=None):
    ### Rectify and Decode to index
    rectifier = StereoRectify(scale=1.0, cali_file=pattern_path+'calib.yml')
    pipe_start_time = start_time = time.time()
    img_index_left, camera_kd_l = get_image_index(pattern_path, '_l.bmp', rectifier=rectifier)
    img_index_right, camera_kd_r = get_image_index(pattern_path, '_r.bmp', rectifier=rectifier)
    print("read image and index decoding in total using %.3f s" % (time.time() - start_time))

    # get camera parameters
    fx = camera_kd_l[0][0]
    cx, cx_r = camera_kd_l[0][2], camera_kd_r[0][2]
    dmap_base = cx_r - cx
    cam_transform = np.array(rectifier.T)[:,0]
    height, width = img_index_left.shape[:2]
    baseline = np.linalg.norm(cam_transform)

    ### Infer DepthMap from Decoded Index
    depth_map = np.zeros_like(img_index_left, dtype=np.float)
    start_time = time.time()
    gen_depth_from_index_matching(depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx)
    print("depth map generating from index %.3f s" % (time.time() - start_time))
    print("Total pipeline time: %.3f s" % (time.time() - pipe_start_time))

    depth_map_uint16 = depth_map * 30000
    cv2.imwrite(res_path + '/depth_alg2.png', depth_map_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    ### Prepare results
    depth_map_mm = depth_map * 1000
    gray_img = cv2.imread(pattern_path + "24_l.bmp", cv2.IMREAD_UNCHANGED)
    gray_img = rectifier.rectify_image(gray_img)
    return gray_img, depth_map_mm, camera_kd_l


# test with existing pattern example: 
#   python structured_light_simplified.py pattern_examples\struli_test1\
if __name__ == "__main__":
    import sys
    import glob
    import shutil
    import open3d as o3d
    import matplotlib.pyplot as plt

    if len(sys.argv) <= 1:
        print("run with args 'pattern_path'")
    image_path = sys.argv[1]

    ### build up runing parameters and run
    res_path = image_path + r'\res' if sys.platform == 'win32' else image_path + '/res'
    if not os.path.exists(res_path): os.system("mkdir " + res_path)
    gray, depth_map_mm, camera_kp = run_stru_li_pipe(image_path, res_path)

    ### build point cloud
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
    print("res saved to:" + res_path)
    
    ### visualize
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.translate(np.zeros(3), relative=False)
    o3d.visualization.draw(geometry=pcd, width=1600, height=900, point_size=1)
