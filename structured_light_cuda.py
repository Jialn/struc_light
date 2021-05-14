
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
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from stereo_rectify import StereoRectify
from structured_light import gray_decode, phase_shift_decode, rectify_phase, rectify_belief_map, get_dmap_from_index_map, optimize_dmap_using_sub_pixel_map, depth_filter, depth_avg_filter

### parameters for the program
phase_decoding_unvalid_thres = 5  # if the diff of pixel in an inversed pattern(has pi phase shift) is smaller than this, consider it's unvalid;
                                  # this value is a balance between valid pts rates and error points rates
                                  # e.g., 1, 2, 5 for low-expo real captured images; 20, 30, 40 for normal expo rendered images.
remove_possibly_outliers_when_matching = True
depth_cutoff_near, depth_cutoff_far = 0.1, 2.0  # depth cutoff
depth_filter_max_distance = 0.005  # about 5-7 times of resolution per pxiel
depth_filter_minmum_points_in_checking_range = 2  # including the point itsself, will also add a ratio of width // 400
use_depth_avg_filter = True
depth_avg_filter_max_length = 2   # 4, from 0 - 6
depth_avg_filter_unvalid_thres = 0.001  # 0.002

roughly_projector_area_in_image = 0.8  # the roughly prjector area in image / image width, e.g., 0.75, 1.0, 1.25
phsift_pattern_period_per_pixel = 10.0  # normalize the index. porjected pattern res width is 1280; 7 graycode pattern = 2^7 = 128 phase shift periods; 1290/128=10 
default_image_seq_start_index = 24  # in some datasets, (0, 24) are for pure gray code solutions 

use_parallel_computing = True
save_mid_res_for_visulize = False
visulize_res = True

### read adn compile cu file
dir_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file
with open(dir_path + "/structured_light_cuda_core.cu", "r") as f:
    cuda_src_string = f.read()
cuda_module = SourceModule(cuda_src_string)

multiply = cuda_module.get_function("multiply")
gray_decode_cuda = cuda_module.get_function("gray_decode")
phase_shift_decode_cuda = cuda_module.get_function("phase_shift_decode")
depth_filter_cuda = cuda_module.get_function("depth_filter")
get_dmap_from_index_map_cuda = cuda_module.get_function("get_dmap_from_index_map")

def gray_decode_cuda_wrapper(src_imgs, images_nega, prj_valid_map_bin, image_num, height,width, img_index, unvalid_thres):
    gray_decode_cuda(drv.In(src_imgs), drv.In(images_nega), drv.In(prj_valid_map_bin),
        drv.In(np.array(image_num).astype(np.int32)),drv.In(np.array(height).astype(np.int32)),drv.In(np.array(width).astype(np.int32)),
        drv.Out(img_index),drv.In(np.array(unvalid_thres).astype(np.int32)),
        block=(width//4, 1, 1), grid=(height*4, 1))

def phase_shift_decode_cuda_wrapper(images_phsft_src, height,width, img_phase, img_index, phase_decoding_unvalid_thres):
    phase_shift_decode_cuda(drv.In(images_phsft_src),
        drv.In(np.array(height).astype(np.int32)),drv.In(np.array(width).astype(np.int32)),
        drv.Out(img_phase),drv.InOut(img_index),drv.In(np.array(phase_decoding_unvalid_thres).astype(np.int32)),
        block=(width//4, 1, 1), grid=(height*4, 1))

def depth_filter_cuda_wrapper(depth_map, depth_map_raw, height, width, camera_kd_l):
    depth_filter_cuda(drv.InOut(depth_map),drv.In(depth_map_raw),
        drv.In(np.array(height).astype(np.int32)),drv.In(np.array(width).astype(np.int32)),
        drv.In(camera_kd_l), drv.In(np.array(depth_filter_max_distance).astype(np.float32)), drv.In(np.array(depth_filter_minmum_points_in_checking_range).astype(np.int32)),
        block=(width//4, 1, 1), grid=(height*4, 1))

def get_dmap_from_index_map_cuda_wrapper(depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right):
    get_dmap_from_index_map_cuda(drv.Out(depth_map),
        drv.In(np.array(height).astype(np.int32)),drv.In(np.array(width).astype(np.int32)),
        drv.In(img_index_left),drv.In(img_index_right), 
        drv.In(np.array(baseline).astype(np.float32)),drv.In(np.array(dmap_base).astype(np.float32)),drv.In(np.array(fx).astype(np.float32)),
        drv.In(img_index_left_sub_px),drv.In(img_index_right_sub_px),drv.In(belief_map_left),drv.In(belief_map_right), 
        drv.In(np.array(roughly_projector_area_in_image).astype(np.float32)),
        block=(96, 1, 1), grid=(height, 1))

# for pycuda test
# h, w = 2048, 2592
# a = np.random.randn(h, w).astype(np.float32)
# b = np.random.randn(h, w).astype(np.float32)
# c = np.random.randn(h, w).astype(np.float32)
# 
# dest = np.zeros_like(a)
# multiply(drv.Out(dest), drv.In(a), drv.In(b), drv.In( np.array(1.0).astype(np.float32) ), block=(w//4,1,1), grid=(h*4,1))
# print(dest)
# print(dest-a*b)
# exit()

### This one has the same logic as cuda implementation. Slower than the original version on CPU.
@numba.jit  ((numba.float32[:,:], numba.int64,numba.int64, numba.float32[:,:],numba.float32[:,:], numba.float32,numba.float32,numba.float32, numba.float32[:,:],numba.float32[:,:],numba.int16[:,:], numba.int16[:,:] ), nopython=True, parallel=use_parallel_computing, nogil=True, cache=True)
def get_dmap_from_index_map2(depth_map, height,width, img_index_left,img_index_right, baseline,dmap_base,fx, img_index_left_sub_px,img_index_right_sub_px, belief_map_l, belief_map_r):
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

### the index decoding part
def get_image_index(image_path, appendix, rectifier, res_path=None, images=None):
    unvalid_thres = 0
    save_mid_res = save_mid_res_for_visulize
    image_seq_start_index = default_image_seq_start_index
    start_time = time.time()
    ### read projector fully open and fully close images
    if images is None:
        images_posi = []
        images_nega = []
        fname = image_path + str(image_seq_start_index) + appendix
        if not os.path.exists(fname): image_seq_start_index = 0
        for i in range(image_seq_start_index, image_seq_start_index+2):
            fname = image_path + str(i) + appendix
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if i % 2 == 0: prj_area_posi = img
            else: prj_area_nega = img
        prj_valid_map = prj_area_posi - prj_area_nega
        if rectifier.remap_x_left_scaled is None: _ = rectifier.rectify_image(prj_area_posi, interpolation=cv2.INTER_NEAREST)  # only to build the internal LUT map
        posi_neg_pattern_avg_thres = (prj_area_posi//2 + prj_area_nega//2)
        thres, prj_valid_map_bin = cv2.threshold(prj_valid_map, unvalid_thres, 255, cv2.THRESH_BINARY)
        ### read gray code and phase shift images
        for i in range(image_seq_start_index+2, image_seq_start_index+10):  # (0, 24) for pure gray code solutions in dataset
            fname = image_path + str(i) + appendix
            if not os.path.exists(fname): break
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            images_posi.append(img)
        images_nega = posi_neg_pattern_avg_thres
        images_phsft = []
        for i in range(image_seq_start_index+10, image_seq_start_index+14):  # phase shift patern 
            fname = image_path + str(i) + appendix
            if not os.path.exists(fname): break
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            images_phsft.append(img)
    else:
        prj_area_posi, prj_area_nega = images[image_seq_start_index], images[image_seq_start_index+1]
        prj_valid_map = prj_area_posi - prj_area_nega
        if rectifier.remap_x_left_scaled is None: _ = rectifier.rectify_image(prj_area_posi, interpolation=cv2.INTER_NEAREST)  # only to build the internal LUT map
        posi_neg_pattern_avg_thres = (prj_area_posi//2 + prj_area_nega//2)
        thres, prj_valid_map_bin = cv2.threshold(prj_valid_map, unvalid_thres, 255, cv2.THRESH_BINARY)
        ### read gray code and phase shift images
        images_posi = images[image_seq_start_index+2:image_seq_start_index+10] # gray code posi images
        images_nega = posi_neg_pattern_avg_thres  # gray code thres
        images_phsft = images[image_seq_start_index+10:image_seq_start_index+14] # phase shift images
    print("read images and build rectify map using %.3f s" % (time.time() - start_time))
    ### decoding
    start_time = time.time()
    height, width = images_posi[0].shape[:2]
    img_index, src_imgs = np.zeros_like(images_posi[0], dtype=np.int16), np.array(images_posi)
    print("build nparray using %.3f s" % (time.time() - start_time))
    start_time = time.time()

    # gray_decode(src_imgs, images_nega, prj_valid_map_bin, len(images_posi), height,width, img_index, unvalid_thres)
    gray_decode_cuda_wrapper(src_imgs, images_nega, prj_valid_map_bin, len(images_posi), height,width, img_index, unvalid_thres)

    print("gray code index decoding using %.3f s" % (time.time() - start_time))
    if save_mid_res and res_path is not None:
        mid_res_corse_gray_index_raw = img_index // 2
        mid_res_corse_gray_index = np.clip(mid_res_corse_gray_index_raw * 80 % 255, 0, 255).astype(np.uint8)
        cv2.imwrite(res_path + "/mid_res_corse_gray_index" + appendix, mid_res_corse_gray_index)

    img_phase = np.zeros_like(images_posi[0], dtype=np.float32)
    images_phsft_src = np.array(images_phsft)
    start_time = time.time()

    # phase_shift_decode(images_phsft_src, height,width, img_phase, img_index, phase_decoding_unvalid_thres)
    phase_shift_decode_cuda_wrapper(images_phsft_src, height,width, img_phase, img_index, phase_decoding_unvalid_thres)

    print("\t phase decoding using %.3f s" % (time.time() - start_time))

    belief_map = img_index
    # rectify image, accroding to left or right
    if appendix == '_l.bmp': rectify_map_x, rectify_map_y, camera_kd = rectifier.remap_x_left_scaled, rectifier.remap_y_left_scaled, rectifier.rectified_camera_kd_l
    else: rectify_map_x, rectify_map_y, camera_kd = rectifier.remap_x_right_scaled, rectifier.remap_y_right_scaled, rectifier.rectified_camera_kd_r

    rectified_img_phase = np.zeros_like(img_phase, dtype=np.float32)
    rectified_belief_map = np.zeros_like(img_phase, dtype=np.int16)
    sub_pixel_map = np.zeros_like(img_phase, dtype=np.float32)
    print("\t phase decoding and build res maps using %.3f s" % (time.time() - start_time))
    rectify_belief_map(belief_map, rectify_map_x, rectify_map_y, height,width, rectified_belief_map)
    rectify_phase(img_phase, rectify_map_x, rectify_map_y, height,width, rectified_img_phase, sub_pixel_map)
    print("phase decoding and rectify using %.3f s" % (time.time() - start_time))
    # cv2.imwrite("./img_phase"+appendix+".png", img_phase.astype(np.uint8))
    # cv2.imwrite("./img_phase"+appendix+"_rectified.png", rectified_img_phase.astype(np.uint8))

    if save_mid_res:
        mid_res_wrapped_phase = (img_phase - mid_res_corse_gray_index_raw * phsift_pattern_period_per_pixel) / phsift_pattern_period_per_pixel
        mid_res_wrapped_phase = (mid_res_wrapped_phase * 254.0)
        cv2.imwrite(res_path + "/mid_res_wrapped_phase"+appendix, mid_res_wrapped_phase.astype(np.uint8))

    return rectified_belief_map, rectified_img_phase, camera_kd, sub_pixel_map


def run_stru_li_pipe(pattern_path, res_path, rectifier=None, images=None):
    if rectifier is None: rectifier = StereoRectify(scale=1.0, cali_file=pattern_path+'calib.yml')
    if images is not None: images_left, images_right = images[0], images[1]
    else: images_left, images_right = None, None

    ### Rectify and Decode to index
    pipe_start_time = start_time = time.time()
    belief_map_left, img_index_left, camera_kd_l, img_index_left_sub_px = get_image_index(pattern_path, '_l.bmp', rectifier=rectifier, res_path=res_path, images=images_left)
    belief_map_right, img_index_right, camera_kd_r, img_index_right_sub_px = get_image_index(pattern_path, '_r.bmp', rectifier=rectifier, res_path=res_path, images=images_right)
    print("read image and index decoding in total using %.3f s" % (time.time() - start_time))

    # get camera parameters
    fx = camera_kd_l[0][0]
    cx, cx_r = camera_kd_l[0][2], camera_kd_r[0][2]
    dmap_base = cx_r - cx
    cam_transform = np.array(rectifier.T)[:,0]
    height, width = img_index_left.shape[:2]
    baseline = np.linalg.norm(cam_transform)  # * ( 0.8/(0.8+0.05*0.001) )  # = 0.9999375039060059

    ### Infer DepthMap from Decoded Index
    unoptimized_depth_map = np.zeros_like(img_index_left, dtype=np.float32)
    depth_map = np.zeros_like(img_index_left, dtype=np.float32)
    start_time = time.time()

    # get_dmap_from_index_map(unoptimized_depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left)
    # get_dmap_from_index_map2(unoptimized_depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right)
    get_dmap_from_index_map_cuda_wrapper(unoptimized_depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right)
    
    optimize_dmap_using_sub_pixel_map(unoptimized_depth_map, depth_map, height,width, img_index_left_sub_px)
    print("depth map generating from index %.3f s" % (time.time() - start_time))

    ### Run Depth Map Filter
    depth_map_raw = depth_map.copy()  # save raw depth map
    start_time = time.time()

    # depth_filter(depth_map, depth_map_raw, height, width, camera_kd_l.astype(np.float32))
    depth_filter_cuda_wrapper(depth_map, depth_map_raw, height, width, camera_kd_l.astype(np.float32))

    print("flying point filter %.3f s" % (time.time() - start_time))
    if use_depth_avg_filter:
        start_time = time.time()
        depth_avg_filter(depth_map, depth_map_raw, height, width)
        print("depth avg filter %.3f s" % (time.time() - start_time))
    print("Total pipeline time: %.3f s" % (time.time() - pipe_start_time))
    
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
#   win: python structured_light_cuda.py pattern_examples\struli_test1\
#   linux: python structured_light_cuda.py pattern_examples/struli_test1/
if __name__ == "__main__":
    import sys
    import glob
    import shutil
    import matplotlib.pyplot as plt

    if len(sys.argv) <= 1:
        print("run with args 'pattern_path'")
    image_path = sys.argv[1]

    ### build up runing parameters and run
    res_path = image_path + r'\res' if sys.platform == 'win32' else image_path + '/res'
    if not os.path.exists(res_path): os.system("mkdir " + res_path)
    gray, depth_map_mm, camera_kp = run_stru_li_pipe(image_path, res_path)

    ### build point cloud
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
    print("res saved to:" + res_path)
    
    def report_depth_error(depth_img, depth_gt):
        gray_img = cv2.imread(image_path + str(default_image_seq_start_index) + "_l.bmp", cv2.IMREAD_UNCHANGED).astype(np.int16)
        gray_img_dark = cv2.imread(image_path + str(default_image_seq_start_index+1) + "_l.bmp", cv2.IMREAD_UNCHANGED).astype(np.int16)
        projector_area_diff = gray_img - gray_img_dark
        projector_area = np.where(projector_area_diff > phase_decoding_unvalid_thres)
        valid_points = np.where(depth_img>=1.0)
        error_img = (depth_img - depth_gt)[valid_points]

        pxiel_num = depth_img.shape[0]*depth_img.shape[1]
        valid_points_num, valid_points_gt_num = len(valid_points[0]), len(projector_area[0])
        print("total pixel: " + str(pxiel_num))
        print("valid points rate: " + str(valid_points_num) + "/" + str(valid_points_gt_num) + ", " + str(100*valid_points_num/valid_points_gt_num)+"%")
        print("average_drift(mm):" + str(np.average(error_img)))
        print("average_error(mm):" + str(np.average(abs(error_img))))
        # error below 10.0mm
        error_img = error_img[np.where((error_img<10.0)&(error_img>-10.0))]
        print("valid points rate below 10mm: " + str(error_img.shape[0]) + "/" + str(valid_points_gt_num) + ", " + str(100*error_img.shape[0]/valid_points_gt_num)+"%")
        print("average_drift(mm):" + str(np.average(error_img)))
        print("average_error(mm):" + str(np.average(abs(error_img))))
        print("diff image:")
        # error below 1.0mm
        error_img = error_img[np.where((error_img<1.0)&(error_img>-1.0))]
        print(error_img.shape[0])
        print("valid points rate below 1mm: " + str(error_img.shape[0]) + "/" + str(valid_points_gt_num) + ", " + str(100*error_img.shape[0]/valid_points_gt_num)+"%")
        print("average_drift(mm):" + str(np.average(error_img)))
        print("average_error(mm):" + str(np.average(abs(error_img))))
        # error below 0.25mm
        error_img = error_img[np.where((error_img<0.25)&(error_img>-0.25))]
        print(error_img.shape[0])
        print("valid points rate below 0.25mm: " + str(error_img.shape[0]) + "/" + str(valid_points_gt_num) + ", " + str(100*error_img.shape[0]/valid_points_gt_num)+"%")
        print("average_drift(mm):" + str(np.average(error_img)))
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
    
    ### visualize
    if visulize_res:
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.translate(np.zeros(3), relative=False)
        o3d.visualization.draw(geometry=pcd, width=1600, height=900, point_size=1)
