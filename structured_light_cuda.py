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

save_mid_res_for_visulize = False
visulize_res = False


### read and compile cu file
dir_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file
with open(dir_path + "/structured_light_cuda_core.cu", "r") as f:
    cuda_src_string = f.read()
cuda_module = SourceModule(cuda_src_string)

cuda_test = cuda_module.get_function("cuda_test")
gray_decode_cuda_kernel = cuda_module.get_function("gray_decode")
phase_shift_decode_cuda_kernel = cuda_module.get_function("phase_shift_decode")
depth_filter_cuda_kernel = cuda_module.get_function("depth_filter")
gen_depth_from_index_matching_cuda_kernel = cuda_module.get_function("gen_depth_from_index_matching")
rectify_phase_and_belief_map_cuda_kernel = cuda_module.get_function("rectify_phase_and_belief_map")
depth_avg_filter_cuda_kernel = cuda_module.get_function("depth_avg_filter")
optimize_dmap_using_sub_pixel_map_cuda_kernel = cuda_module.get_function("optimize_dmap_using_sub_pixel_map")

def gray_decode_cuda(src_imgs, avg_thres_posi, avg_thres_nega, prj_valid_map_bin, image_num, height,width, img_index, unvalid_thres):
    gray_decode_cuda_kernel(src_imgs, cuda.In(avg_thres_posi), cuda.In(avg_thres_nega), cuda.In(prj_valid_map_bin),
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
        cuda.In(np.float32(roughly_projector_area_in_image)),
        block=(width//4, 1, 1), grid=(height*4, 1))

def gen_depth_from_index_matching_cuda(depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right):
    gen_depth_from_index_matching_cuda_kernel(depth_map,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        img_index_left, img_index_right, 
        cuda.In(np.float32(baseline)), cuda.In(np.float32(dmap_base)),cuda.In(np.float32(fx)),
        img_index_left_sub_px, img_index_right_sub_px, belief_map_left,belief_map_right, 
        cuda.In(np.float32(roughly_projector_area_in_image)), cuda.In(np.float32([depth_cutoff_near, depth_cutoff_far])),
        block=(48*9, 1, 1), grid=(height, 1))

def optimize_dmap_using_sub_pixel_map_cuda(unoptimized_depth_map, depth_map, height,width, img_index_left_sub_px):
    optimize_dmap_using_sub_pixel_map_cuda_kernel(unoptimized_depth_map,depth_map,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        img_index_left_sub_px,
        block=(width//4, 1, 1), grid=(height*4, 1))

def depth_filter_cuda(depth_map, depth_map_raw, height, width, camera_kd_l):
    depth_filter_cuda_kernel(depth_map, depth_map_raw,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        cuda.In(camera_kd_l), cuda.In(np.float32(depth_filter_max_distance)), cuda.In(np.int32(depth_filter_minmum_points_in_checking_range)),
        block=(width//4, 1, 1), grid=(height*4, 1))

def depth_avg_filter_cuda(depth_map, height, width):
    depth_avg_filter_cuda_kernel(depth_map,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        cuda.In(np.int32(depth_avg_filter_max_length)), cuda.In(np.float32(depth_avg_filter_unvalid_thres)),
        block=(width//4, 1, 1), grid=(height*4, 1))

# ### for simple pycuda test
# h, w = 2048, 2592
# a = np.random.randn(h, w).astype(np.float32)
# b = np.random.randn(h, w).astype(np.float32)
# dest = np.empty_like(a)
# start_time = time.time()
# cuda_test(cuda.Out(dest), cuda.In(a), cuda.In(b), cuda.In( np.array(1.0).astype(np.float32) ), block=(32,1,1), grid=(w*h//32,1))
# print("running time: %.4f s" % ((time.time() - start_time)/3))
# # print(dest-a*b)
# exit()

### the index decoding part
global_reading_img_time = 0
img_phase = None # gpu array, will be faster as global variable(will not free mem every call)
img_index = None
gpu_remap_x_left = None
gpu_remap_y_left = None
gpu_remap_x_right = None
gpu_remap_y_right = None
def index_decoding_from_images(image_path, appendix, rectifier, res_path=None, images=None):
    global global_reading_img_time, img_phase, img_index, gpu_remap_x_left, gpu_remap_y_left, gpu_remap_x_right, gpu_remap_y_right
    unvalid_thres = 0
    save_mid_res = save_mid_res_for_visulize
    image_seq_start_index = default_image_seq_start_index
    # read images
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
    prj_valid_map = prj_area_posi - prj_area_nega
    thres, prj_valid_map_bin = cv2.threshold(prj_valid_map, unvalid_thres, 255, cv2.THRESH_BINARY)
    if img_phase is None:
        img_phase = cuda.mem_alloc(prj_valid_map.nbytes*4)  # float32
        img_index = cuda.mem_alloc(prj_valid_map.nbytes*2)  # int16
    # print("read images and rectfy map: %.3f s" % (time.time() - start_time))
    global_reading_img_time += (time.time() - start_time)
    ### prepare gpu data
    start_time = time.time()
    image_num_gray, image_num_phsft = len(images_graycode), len(images_phsft)
    images_gray_src =       cuda.mem_alloc(prj_valid_map.nbytes*image_num_gray) # np.array(images_graycode)
    images_phsft_src =      cuda.mem_alloc(prj_valid_map.nbytes*image_num_phsft) # np.array(images_phsft)
    rectified_img_phase =   cuda.mem_alloc(prj_valid_map.nbytes*4) # np.empty_like(prj_valid_map, dtype=np.float32)
    rectified_belief_map =  cuda.mem_alloc(prj_valid_map.nbytes*2) # np.empty_like(prj_valid_map, dtype=np.int16)
    sub_pixel_map =         cuda.mem_alloc(prj_valid_map.nbytes*4) # np.empty_like(prj_valid_map, dtype=np.float32)
    for i in range(image_num_gray):
        cuda.memcpy_htod(int(images_gray_src)+i*prj_valid_map.nbytes, images_graycode[i])
    for i in range(image_num_phsft):
        cuda.memcpy_htod(int(images_phsft_src)+i*prj_valid_map.nbytes, images_phsft[i])

    height, width = images_graycode[0].shape[:2]
    print("alloc gpu mem and copy src images into gpu: %.3f s" % (time.time() - start_time))
    ### decoding
    start_time = time.time()
    gray_decode_cuda(images_gray_src, prj_area_posi, prj_area_nega, prj_valid_map_bin, len(images_graycode), height,width, img_index, unvalid_thres)
    print("gray code decoding: %.3f s" % (time.time() - start_time))
    if save_mid_res and res_path is not None:
        mid_res_corse_gray_index_raw = img_index // 2
        mid_res_corse_gray_index = np.clip(mid_res_corse_gray_index_raw * 80 % 255, 0, 255).astype(np.uint8)
        cv2.imwrite(res_path + "/mid_res_corse_gray_index" + appendix, mid_res_corse_gray_index)
  
    start_time = time.time()
    phase_shift_decode_cuda(images_phsft_src, height,width, img_phase, img_index, phase_decoding_unvalid_thres)
    belief_map = img_index # img_index reused as belief_map when phase_shift_decoding
    print("phase decoding: %.3f s" % (time.time() - start_time))
    
    ### rectify the decoding res, accroding to left or right
    start_time = time.time()
    if appendix == '_l.bmp': rectify_map_x, rectify_map_y, camera_kd = gpu_remap_x_left, gpu_remap_y_left, rectifier.rectified_camera_kd_l
    else: rectify_map_x, rectify_map_y, camera_kd = gpu_remap_x_right, gpu_remap_y_right, rectifier.rectified_camera_kd_r
    rectify_phase_and_belief_map_cuda(img_phase, belief_map, rectify_map_x, rectify_map_y, height,width, rectified_img_phase, rectified_belief_map, sub_pixel_map)
    print("rectify: %.3f s" % (time.time() - start_time))

    if save_mid_res:
        mid_res_wrapped_phase = (img_phase - mid_res_corse_gray_index_raw * phsift_pattern_period_per_pixel) / phsift_pattern_period_per_pixel
        mid_res_wrapped_phase = (mid_res_wrapped_phase * 254.0)
        cv2.imwrite(res_path + "/mid_res_wrapped_phase"+appendix, mid_res_wrapped_phase.astype(np.uint8))

    return prj_area_posi, rectified_belief_map, rectified_img_phase, camera_kd, sub_pixel_map


def run_stru_li_pipe(pattern_path, res_path, rectifier=None, images=None):
    global global_reading_img_time
    if rectifier is None: rectifier = StereoRectify(scale=1.0, cali_file=pattern_path+'calib.yml')
    if images is not None: images_left, images_right = images[0], images[1]
    else: images_left, images_right = None, None
    ### Rectify and Decode 
    pipe_start_time = start_time = time.time()
    gray_left, belief_map_left, img_index_left, camera_kd_l, img_index_left_sub_px = index_decoding_from_images(pattern_path, '_l.bmp', rectifier=rectifier, res_path=res_path, images=images_left)
    print("- left decoding total: %.3f s" % (time.time() - start_time - global_reading_img_time))
    _, belief_map_right, img_index_right, camera_kd_r, img_index_right_sub_px = index_decoding_from_images(pattern_path, '_r.bmp', rectifier=rectifier, res_path=res_path, images=images_right)
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
    gen_depth_from_index_matching_cuda(gpu_unoptimized_depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px, belief_map_left, belief_map_right)
    print("index matching and depth map generating: %.3f s" % (time.time() - start_time))
    start_time = time.time()
    optimize_dmap_using_sub_pixel_map_cuda(gpu_unoptimized_depth_map, gpu_depth_map_raw, height,width, img_index_left_sub_px)
    print("subpix optimize: %.3f s" % (time.time() - start_time))
    ### Run Depth Map Filter
    start_time = time.time()
    depth_filter_cuda(gpu_depth_map_filtered, gpu_depth_map_raw, height, width, camera_kd_l.astype(np.float32))
    print("flying point filter: %.3f s" % (time.time() - start_time))
    if use_depth_avg_filter:
        start_time = time.time()
        depth_avg_filter_cuda(gpu_depth_map_filtered, height, width)
        print("depth avg filter: %.3f s" % (time.time() - start_time))
    # readout
    start_time = time.time()
    cuda.memcpy_dtoh(depth_map, gpu_depth_map_filtered)
    print("readout from gpu: %.3f s" % (time.time() - start_time))

    print("- Total time: %.3f s" % (time.time() - pipe_start_time))
    print("- Total time without reading imgs and pre-built rectify maps: %.3f s" % (time.time() - pipe_start_time - global_reading_img_time))
    global_reading_img_time = 0
    ### Save Mid Results for visualizing
    if save_mid_res_for_visulize:   
        depth_map_uint16 = depth_map * 30000
        cv2.imwrite(res_path + '/depth_alg2.png', depth_map_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        depth_map_raw = np.empty_like(gray_left, dtype=np.float32)
        cuda.memcpy_dtoh(depth_map_raw, gpu_depth_map_raw)
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
    gray_img = rectifier.rectify_image(gray_left)
    return gray_img, depth_map_mm, camera_kd_l


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
