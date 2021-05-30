"""
Description:
This program implements the structured light pipeline with graycode and phase shift pattern.
"""
import os
import cv2
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from stereo_rectify import StereoRectify
import depth_map_utils as utils

### parameters 
phase_decoding_unvalid_thres = 2    # if the diff of pixel in an inversed pattern(has pi phase shift) is smaller than this, consider it's unvalid;
                                    # this value is a balance between valid pts rates and error points rates
                                    # e.g., 1, 2, 5 for low-expo real captured images; 2, 5, 20 for normal or high expo rendered images.
                                    # lower value bring many wrong paried left and right indexs
                                    # if phase_decoding_unvalid_thres <= 1, enhanced_belief_map_checking_when_matching is forced enabled
use_belief_map_for_checking = True  # use enhanced matching with belief_map, gives more robust matching result, but slow;
remove_possibly_outliers_when_matching = True
depth_cutoff_near, depth_cutoff_far = 0.1, 2.0  # depth cutoff
flying_points_filter_checking_range = 0.003     # about 5-10 times of resolution per projector pxiel
flying_points_filter_minmum_points_in_checking_range = 5  # including the point itself, will also add a ratio of width // 400
use_depth_filter = True                         # a filter that smothing the image while preserves local structure
depth_filter_max_length = 3                     # from 0 - 6
depth_filter_unvalid_thres = 0.001

roughly_projector_area_ratio_in_image = None    # the roughly prjector area in image / image width, e.g., 0.5, 0.75, 1.0, 1.25
                                                # this parameter assume projector resolution is 1K, and decoded index should have the same value as projector's pix
                                                # if None, will be estimated from image automaticly
phsift_pattern_period_per_pixel = 10.0  # normalize the index. porjected pattern res width is 1280; 7 graycode pattern = 2^7 = 128 phase shift periods; 1290/128=10 
default_image_seq_start_index = 24      # in some datasets, (0, 24) are for pure gray code solutions 

save_mid_res_for_visulize = False
visulize_res = True

enable_depth_map_post_processing = True


### read and compile cu file
dir_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file
with open(dir_path + "/structured_light_cuda_core.cu", "r") as f:
    cuda_src_string = f.read()
if phase_decoding_unvalid_thres <= 1 or use_belief_map_for_checking:
    cuda_src_string = "#define use_belief_map_for_checking\n" + cuda_src_string
cuda_module = SourceModule(cuda_src_string)

convert_bayer = cuda_module.get_function("convert_bayer_to_blue")
gray_decode_cuda_kernel = cuda_module.get_function("gray_decode")
phase_shift_decode_cuda_kernel = cuda_module.get_function("phase_shift_decode")
flying_points_filter_cuda_kernel = cuda_module.get_function("flying_points_filter")
gen_depth_from_index_matching_cuda_kernel = cuda_module.get_function("gen_depth_from_index_matching")
rectify_phase_and_belief_map_cuda_kernel = cuda_module.get_function("rectify_phase_and_belief_map")
depth_filter_cuda_kernel = cuda_module.get_function("depth_filter")
optimize_dmap_using_sub_pixel_map_cuda_kernel = cuda_module.get_function("optimize_dmap_using_sub_pixel_map")
convert_dmap_to_mili_meter = cuda_module.get_function("convert_dmap_to_mili_meter")

def gray_decode_cuda(src_imgs, avg_thres_posi, avg_thres_nega, prj_valid_map, image_num, height,width, img_index, unvalid_thres):
    gray_decode_cuda_kernel(src_imgs, avg_thres_posi, avg_thres_nega, prj_valid_map,
        cuda.In(np.int32(image_num)),cuda.In(np.int32(height)),cuda.In(np.int32(width)),
        img_index,cuda.In(np.int32(unvalid_thres)),
        block=(width//4, 1, 1), grid=(height*4, 1))

def phase_shift_decode_cuda(images_phsft_src, height,width, img_phase, img_index, phase_decoding_unvalid_thres):
    phase_shift_decode_cuda_kernel(images_phsft_src,
        cuda.In(np.int32(height)),cuda.In(np.int32(width)),
        img_phase,img_index,cuda.In(np.int32(phase_decoding_unvalid_thres)),cuda.In(np.float32(phsift_pattern_period_per_pixel)),
        block=(width//4, 1, 1), grid=(height*4, 1))

def rectify_phase_and_belief_map_cuda(img_phase, belief_map, rectify_map_x, rectify_map_y, height,width, rectified_img_phase, rectified_belief_map, sub_pixel_map):
    rectify_phase_and_belief_map_cuda_kernel(img_phase, belief_map, rectify_map_x, rectify_map_y,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        rectified_img_phase, rectified_belief_map, sub_pixel_map,
        block=(width//4, 1, 1), grid=(height*4, 1))

def gen_depth_from_index_matching_cuda(depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right, roughly_projector_area_ratio):
    gen_depth_from_index_matching_cuda_kernel(depth_map,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        img_index_left, img_index_right, 
        cuda.In(np.float32(baseline)), cuda.In(np.float32(dmap_base)),cuda.In(np.float32(fx)),
        img_index_left_sub_px, img_index_right_sub_px, belief_map_left,belief_map_right, 
        cuda.In(np.float32(roughly_projector_area_ratio)), cuda.In(np.float32([depth_cutoff_near, depth_cutoff_far])),
        cuda.In(np.int32(remove_possibly_outliers_when_matching)),
        block=(4, 16, 1), grid=(height, 1))

def optimize_dmap_using_sub_pixel_map_cuda(unoptimized_depth_map, depth_map, height,width, img_index_left_sub_px):
    optimize_dmap_using_sub_pixel_map_cuda_kernel(unoptimized_depth_map,depth_map,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        img_index_left_sub_px,
        block=(width//4, 1, 1), grid=(height*4, 1))

def flying_points_filter_cuda(depth_map, depth_map_raw, height, width, camera_kd_l, belief_map):
    flying_points_filter_cuda_kernel(depth_map, depth_map_raw,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        cuda.In(camera_kd_l), cuda.In(np.float32(flying_points_filter_checking_range)), cuda.In(np.int32(flying_points_filter_minmum_points_in_checking_range)), belief_map,
        block=(width//4, 1, 1), grid=(height*4, 1))

def depth_filter_cuda(depth_map, height, width, belief_map):
    depth_filter_cuda_kernel(depth_map,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        cuda.In(np.int32(depth_filter_max_length)), cuda.In(np.float32(depth_filter_unvalid_thres)), belief_map,
        block=(width//4, 1, 1), grid=(height*4, 1))

def from_gpu(gpu_data, size_sample, dtype):
    nd_array = np.empty_like(size_sample, dtype)
    cuda.memcpy_dtoh(nd_array, gpu_data)
    return nd_array

### the index decoding part
global_reading_img_time = 0
img_phase = None # gpu array, will be faster as global variable(will not free mem every call)
img_index = None
gpu_remap_x_left = None
gpu_remap_y_left = None
gpu_remap_x_right = None
gpu_remap_y_right = None
def index_decoding_from_images(image_path, appendix, rectifier, is_bayer_color_image, res_path=None, images=None):
    global global_reading_img_time, img_phase, img_index, gpu_remap_x_left, gpu_remap_y_left, gpu_remap_x_right, gpu_remap_y_right, roughly_projector_area_ratio_in_image
    save_mid_res = save_mid_res_for_visulize
    image_seq_start_index = default_image_seq_start_index
    start_time = time.time()
    if images is None:
        fname = image_path + str(image_seq_start_index) + appendix
        if not os.path.exists(fname): image_seq_start_index = 0
        # read projector fully open and fully close images
        prj_area_posi = cv2.imread(image_path + str(image_seq_start_index) + appendix, cv2.IMREAD_UNCHANGED)
        prj_area_nega = cv2.imread(image_path + str(image_seq_start_index+1) + appendix, cv2.IMREAD_UNCHANGED)
        # read gray code and phase shift images
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

    if rectifier.remap_x_left_scaled is None: # to build the internal LUT map
        _ = rectifier.rectify_image(prj_area_posi, interpolation=cv2.INTER_NEAREST)
        rectify_map_x_left, rectify_map_y_left, camera_kd_left = rectifier.remap_x_left_scaled, rectifier.remap_y_left_scaled, rectifier.rectified_camera_kd_l
        rectify_map_x_right, rectify_map_y_right, camera_kd_right = rectifier.remap_x_right_scaled, rectifier.remap_y_right_scaled, rectifier.rectified_camera_kd_r
        gpu_remap_x_left =  cuda.mem_alloc(rectify_map_x_left.nbytes)
        gpu_remap_y_left =  cuda.mem_alloc(rectify_map_y_left.nbytes)
        gpu_remap_x_right = cuda.mem_alloc(rectify_map_x_right.nbytes)
        gpu_remap_y_right = cuda.mem_alloc(rectify_map_y_right.nbytes)
        cuda.memcpy_htod(gpu_remap_x_left,  rectify_map_x_left)
        cuda.memcpy_htod(gpu_remap_y_left,  rectify_map_y_left)
        cuda.memcpy_htod(gpu_remap_x_right, rectify_map_x_right)
        cuda.memcpy_htod(gpu_remap_y_right, rectify_map_y_right)
    if roughly_projector_area_ratio_in_image is None:
        valid_map_diff = prj_area_posi.astype(np.int16) - prj_area_nega.astype(np.int16)
        _, prj_valid_map_for_prjratio = cv2.threshold(valid_map_diff.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        total_pix, projector_area_pix = prj_valid_map_for_prjratio.nbytes, len(np.where(prj_valid_map_for_prjratio == 255)[0])
        roughly_projector_area_ratio_in_image = np.sqrt(projector_area_pix/total_pix)
        print("estimated valid_area_ratio: "+str(roughly_projector_area_ratio_in_image))
    if img_phase is None:
        img_phase = cuda.mem_alloc(prj_area_posi.nbytes*4)  # float32
        img_index = cuda.mem_alloc(prj_area_posi.nbytes*2)  # int16
    # print("read images and rectfy map: %.3f s" % (time.time() - start_time))
    global_reading_img_time += (time.time() - start_time)
    
    ### prepare gpu data
    start_time = time.time()
    image_num_gray, image_num_phsft = len(images_graycode), len(images_phsft)
    prj_area_posi_gpu =     cuda.mem_alloc(prj_area_posi.nbytes)
    prj_area_nega_gpu =     cuda.mem_alloc(prj_area_posi.nbytes)
    prj_valid_map =         cuda.mem_alloc(prj_area_posi.nbytes)
    images_gray_src =       cuda.mem_alloc(prj_area_posi.nbytes*image_num_gray) # np.array(images_graycode)
    images_phsft_src =      cuda.mem_alloc(prj_area_posi.nbytes*image_num_phsft) # np.array(images_phsft)
    rectified_img_phase =   cuda.mem_alloc(prj_area_posi.nbytes*4) # np.empty_like(prj_area_posi, dtype=np.float32)
    rectified_belief_map =  cuda.mem_alloc(prj_area_posi.nbytes*2) # np.empty_like(prj_area_posi, dtype=np.int16)
    sub_pixel_map =         cuda.mem_alloc(prj_area_posi.nbytes*4) # np.empty_like(prj_area_posi, dtype=np.float32)
    height, width = images_graycode[0].shape[:2]
    cuda.memcpy_htod(prj_area_posi_gpu, prj_area_posi)
    cuda.memcpy_htod(prj_area_nega_gpu, prj_area_nega)
    for i in range(image_num_gray):
        cuda.memcpy_htod(int(images_gray_src)+i*prj_area_posi.nbytes, images_graycode[i])
    for i in range(image_num_phsft):
        cuda.memcpy_htod(int(images_phsft_src)+i*prj_area_posi.nbytes, images_phsft[i])
    print("alloc gpu mem and copy src images into gpu: %.3f s" % (time.time() - start_time))
    if is_bayer_color_image:
        start_time = time.time()
        convert_bayer(prj_area_posi_gpu, cuda.In(np.int32(height)),cuda.In(np.int32(width)),
                block=(width//4, 1, 1), grid=(height, 1))
        # demosac_img_cuda = from_gpu(prj_area_posi_gpu, size_sample=prj_area_posi, dtype=np.uint8)
        # demosac_img_cuda_opencv = cv2.cvtColor(prj_area_posi, cv2.COLOR_BAYER_BG2BGR)[:,:,0]
        # cv2.imwrite(res_path + "/demosac_img_cuda" + appendix, demosac_img_cuda)
        # cv2.imwrite(res_path + "/demosac_img_cuda_opencv" + appendix, demosac_img_cuda_opencv)
        # cv2.imwrite(res_path + "/diff_tmp" + appendix, 128*abs(demosac_img_cuda_opencv-demosac_img_cuda))
        convert_bayer(prj_area_nega_gpu, cuda.In(np.int32(height)),cuda.In(np.int32(width)),
                block=(width//4, 1, 1), grid=(height, 1))
        for i in range(image_num_gray):
            convert_bayer(images_gray_src, cuda.In(np.int32(height)),cuda.In(np.int32(width)),
                block=(width//4, 1, 1), grid=(height, 1))
        for i in range(image_num_phsft):
            convert_bayer(images_phsft_src, cuda.In(np.int32(height)),cuda.In(np.int32(width)),
                block=(width//4, 1, 1), grid=(height, 1))
        print("demosac using gpu: %.3f s" % (time.time() - start_time))

    ### decoding
    start_time = time.time()
    gray_decode_cuda(images_gray_src, prj_area_posi_gpu, prj_area_nega_gpu, prj_valid_map, len(images_graycode), height,width, img_index, phase_decoding_unvalid_thres+1)
    print("gray code decoding: %.3f s" % (time.time() - start_time))
    if save_mid_res and res_path is not None:
        mid_res_corse_gray_index_raw = from_gpu(img_index, size_sample=prj_area_posi, dtype=np.int16) // 2
        mid_res_corse_gray_index = np.clip(mid_res_corse_gray_index_raw * 80 % 255, 0, 255).astype(np.uint8)
        cv2.imwrite(res_path + "/mid_res_corse_gray_index" + appendix, mid_res_corse_gray_index)
  
    start_time = time.time()
    phase_shift_decode_cuda(images_phsft_src, height,width, img_phase, img_index, phase_decoding_unvalid_thres)
    belief_map = img_index # img_index reused as belief_map when phase_shift_decoding
    print("phase decoding: %.3f s" % (time.time() - start_time))
    if save_mid_res:
        # check for the unrectified phase
        images_phsft_v = (from_gpu(img_phase, size_sample=prj_area_posi, dtype=np.float32)*4.0).astype(np.uint8)
        cv2.imwrite(res_path + "/ph_correspondence_l" + appendix[:2] + "_unrectified.png", images_phsft_v)
        prj_valid_map_bin = from_gpu(prj_valid_map, size_sample=prj_area_posi, dtype=np.uint8)
        cv2.imwrite(res_path + "/prj_valid_map_gpu" + appendix[:2] + "_bin.png", prj_valid_map_bin)
    
    ### rectify the decoding res, accroding to left or right
    start_time = time.time()
    if appendix == '_l.bmp': rectify_map_x, rectify_map_y, camera_kd = gpu_remap_x_left, gpu_remap_y_left, rectifier.rectified_camera_kd_l
    else: rectify_map_x, rectify_map_y, camera_kd = gpu_remap_x_right, gpu_remap_y_right, rectifier.rectified_camera_kd_r
    rectify_phase_and_belief_map_cuda(img_phase, belief_map, rectify_map_x, rectify_map_y, height,width, rectified_img_phase, rectified_belief_map, sub_pixel_map)
    print("rectify: %.3f s" % (time.time() - start_time))

    if save_mid_res:
        mid_res_wrapped_phase = (from_gpu(img_phase, size_sample=prj_area_posi, dtype=np.float32) - mid_res_corse_gray_index_raw * phsift_pattern_period_per_pixel) / phsift_pattern_period_per_pixel
        mid_res_wrapped_phase = (mid_res_wrapped_phase * 254.0)
        cv2.imwrite(res_path + "/mid_res_wrapped_phase"+appendix, mid_res_wrapped_phase.astype(np.uint8))

    return prj_area_posi, rectified_belief_map, rectified_img_phase, camera_kd, sub_pixel_map


def run_stru_li_pipe(pattern_path, res_path, rectifier=None, images=None, is_bayer_color_image=False):
    # return depth map in mili-meter
    global global_reading_img_time
    if rectifier is None: rectifier = StereoRectify(scale=1.0, cali_file=pattern_path+'calib.yml')
    if images is not None: images_left, images_right = images[0], images[1]
    else: images_left, images_right = None, None
    ### Rectify and Decode 
    pipe_start_time = start_time = time.time()
    gray_left, belief_map_left, img_index_left, camera_kd_l, img_index_left_sub_px = index_decoding_from_images(pattern_path, '_l.bmp', rectifier=rectifier, res_path=res_path, images=images_left, is_bayer_color_image=is_bayer_color_image)
    _, belief_map_right, img_index_right, camera_kd_r, img_index_right_sub_px = index_decoding_from_images(pattern_path, '_r.bmp', rectifier=rectifier, res_path=res_path, images=images_right, is_bayer_color_image=is_bayer_color_image)
    print("- left and right decoding in total: %.3f s" % (time.time() - start_time - global_reading_img_time))
    ### Get camera parameters
    fx = camera_kd_l[0][0]
    cx, cx_r = camera_kd_l[0][2], camera_kd_r[0][2]
    dmap_base = cx_r - cx
    cam_transform = np.array(rectifier.T)[:,0]
    height, width = gray_left.shape[:2]
    baseline = np.linalg.norm(cam_transform)  # * ( 0.8/(0.8+0.05*0.001) )  # = 0.9999375039060059
    ### Alloc mem for the fllowing procedures
    # start_time = time.time()
    gpu_unoptimized_depth_map = cuda.mem_alloc(gray_left.nbytes*4) # np.empty_like(gray_left, dtype=np.float32)
    gpu_depth_map_raw = cuda.mem_alloc(gray_left.nbytes*4)
    gpu_depth_map_filtered = cuda.mem_alloc(gray_left.nbytes*4)
    depth_map = np.empty_like(gray_left, dtype=np.float32)
    # print("alloc mem for maps: %.3f s" % (time.time() - start_time))  # less than 1ms
    ### Infer DepthMap from Index Matching
    start_time = time.time()
    gen_depth_from_index_matching_cuda(gpu_unoptimized_depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right, roughly_projector_area_ratio_in_image)
    print("index matching and depth map generating: %.3f s" % (time.time() - start_time))
    start_time = time.time()
    optimize_dmap_using_sub_pixel_map_cuda(gpu_unoptimized_depth_map, gpu_depth_map_raw, height,width, img_index_left_sub_px)
    print("subpix optimize: %.3f s" % (time.time() - start_time))
    ### Run Depth Map Filter
    start_time = time.time()
    flying_points_filter_cuda(gpu_depth_map_filtered, gpu_depth_map_raw, height, width, camera_kd_l.astype(np.float32), belief_map_left)
    print("flying point filter: %.3f s" % (time.time() - start_time))
    if use_depth_filter:
        start_time = time.time()
        depth_filter_cuda(gpu_depth_map_filtered, height, width, belief_map_left)
        print("depth smothing filter: %.3f s" % (time.time() - start_time))
    # readout
    convert_dmap_to_mili_meter(gpu_depth_map_filtered, block=(width//4, 1, 1), grid=(height*4, 1))

    start_time = time.time()
    cuda.memcpy_dtoh(depth_map, gpu_depth_map_filtered)
    print("readout from gpu: %.3f s" % (time.time() - start_time))

    print("- Total time: %.3f s" % (time.time() - pipe_start_time))
    print("- Total time without reading imgs and pre-built rectify maps: %.3f s" % (time.time() - pipe_start_time - global_reading_img_time))

    if enable_depth_map_post_processing:
        start_time = time.time()
        depth_map = utils.depth_map_post_processing(depth_map)
        print("depth post processing: %.3f s" % (time.time() - start_time))
    global_reading_img_time = 0
    ### Save Mid Results for visualizing
    if save_mid_res_for_visulize:   
        depth_map_uint16 = depth_map * 30  # mili-meter
        cv2.imwrite(res_path + '/depth_alg2.png', depth_map_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        depth_map_raw = from_gpu(gpu_depth_map_raw, size_sample=gray_left, dtype=np.float32)
        depth_map_raw_uint16 = depth_map_raw * 30000  # meter
        cv2.imwrite(res_path + '/depth_alg2_raw.png', depth_map_raw_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        belief_map_left = np.clip(from_gpu(belief_map_left, size_sample=gray_left, dtype=np.uint16) * 255, 0, 255).astype(np.uint8)
        belief_map_right = np.clip(from_gpu(belief_map_right, size_sample=gray_left, dtype=np.uint16) * 255, 0, 255).astype(np.uint8)
        images_phsft_left_v = (from_gpu(img_index_left, size_sample=gray_left, dtype=np.float32)*4.0).astype(np.uint8)
        images_phsft_right_v = (from_gpu(img_index_right, size_sample=gray_left, dtype=np.float32)*4.0).astype(np.uint8)
        cv2.imwrite(res_path + "/belief_map_left.png", belief_map_left)
        cv2.imwrite(res_path + "/belief_map_right.png", belief_map_right)
        cv2.imwrite(res_path + "/ph_correspondence_l.png", images_phsft_left_v)
        cv2.imwrite(res_path + "/ph_correspondence_r.png", images_phsft_right_v)
    ### Prepare results
    gray_img = rectifier.rectify_image(gray_left)
    return gray_img, depth_map, camera_kd_l


# test with existing pattern example: 
#   win: python structured_light_cuda.py pattern_examples\struli_test1\
#   linux: python structured_light_cuda.py pattern_examples/struli_test1/
if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("run with args 'pattern_path'")
    image_path = sys.argv[1]

    ### build up runing parameters and run
    res_path = image_path + r'\res' if sys.platform == 'win32' else image_path + '/res'
    if not os.path.exists(res_path): os.system("mkdir " + res_path)
    
    rectifier = StereoRectify(scale=1.0, cali_file=image_path+'calib.yml')
    gray, depth_map_mm, camera_kp = run_stru_li_pipe(image_path, res_path, rectifier=rectifier)
    gray, depth_map_mm, camera_kp = run_stru_li_pipe(image_path, res_path, rectifier=rectifier)
    

    def report_depth_error(depth_img, depth_gt):
        gray_img = cv2.imread(image_path + str(default_image_seq_start_index) + "_l.bmp", cv2.IMREAD_UNCHANGED).astype(np.int16)
        gray_img_dark = cv2.imread(image_path + str(default_image_seq_start_index+1) + "_l.bmp", cv2.IMREAD_UNCHANGED).astype(np.int16)
        projector_area_diff = gray_img - gray_img_dark
        projector_area = np.where(projector_area_diff > 5)
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
        if save_mid_res_for_visulize:
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
    else:
        valid_points = np.where(depth_map_mm>=1.0)
        print("valid points: " + str(len(valid_points[0])))
    
    ### build point cloud and visualize
    if visulize_res:
        import open3d as o3d
        fx, fy, cx, cy = camera_kp[0][0], camera_kp[1][1], camera_kp[0][2], camera_kp[1][2]
        if os.path.exists(image_path + "color.bmp"): gray = cv2.imread(image_path + "color.bmp")
        pcd = utils.gen_point_clouds_from_images(depth_map_mm, camera_kp, gray, save_path=res_path)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.translate(np.zeros(3), relative=False)
        o3d.visualization.draw(geometry=pcd, width=1800, height=1000, point_size=1)
