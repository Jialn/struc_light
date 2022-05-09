"""
Description:
An example for pycuda depth map filter

Setup:
pip3 install numpy opencv-python open3d pycuda
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import CudaModule
from pycuda.driver import module_from_buffer

def convert_depth_to_color(depth_map_mm, scale=None):
    h, w = depth_map_mm.shape[:2]
    depth_image_color_vis = depth_map_mm.copy()
    valid_points = np.where(depth_image_color_vis>=0.1)
    depth_near_cutoff, depth_far_cutoff = np.min(depth_image_color_vis[valid_points]), np.max(depth_image_color_vis[valid_points])
    # depth_near_cutoff, depth_far_cutoff = np.percentile(depth_image_color_vis[valid_points], 1), np.percentile(depth_image_color_vis[valid_points], 99)
    depth_far_cutoff = depth_near_cutoff + (depth_far_cutoff-depth_near_cutoff) * 1.2
    depth_range = depth_far_cutoff-depth_near_cutoff
    # print((depth_near_cutoff, depth_far_cutoff))
    depth_image_color_vis[valid_points] = depth_far_cutoff - depth_image_color_vis[valid_points]  # - depth_near_cutoff
    depth_image_color_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_color_vis, alpha=255/(depth_range)), cv2.COLORMAP_JET)  #COLORMAP_JET HOT
    if scale is not None:
        depth_image_color_vis = cv2.resize(depth_image_color_vis, ((int)(w*scale), (int)(h*scale)))
    return depth_image_color_vis

def gen_point_clouds_from_images(depth, camera_kp, image, save_path=None):
    """Generate PointCloud from images and camera kp
    """
    import open3d as o3d
    import copy
    convert_rgb_to_intensity = True
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        convert_rgb_to_intensity = False
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(image),
        o3d.geometry.Image(depth.astype(np.float32)),
        convert_rgb_to_intensity=convert_rgb_to_intensity,
        depth_scale=1.0,
        depth_trunc=6000.0)
    h, w = image.shape[:2]
    fx, fy, cx, cy = camera_kp[0][0], camera_kp[1][1], camera_kp[0][2], camera_kp[1][2]
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy))
    if save_path is not None:
        if save_path[-4:] != '.ply': save_path = save_path + "/points.ply"
        pcd_to_write = copy.deepcopy(pcd)
        pcd_to_write.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=8))
        pcd_to_write.orient_normals_towards_camera_location()
        o3d.io.write_point_cloud(save_path, pcd_to_write, write_ascii=False, compressed=False)
        print("res saved to:" + save_path)
    return pcd
    
### compile cu file if needed
dir_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file
cuda_module = CudaModule()
with open(dir_path + "/pycuda_depth_filter.cu", "r", encoding='utf-8-sig') as f:
    cuda_src_string = f.read()
cubin_file = pycuda.compiler.compile(cuda_src_string, nvcc="nvcc", options=None, keep=False, no_extern_c=False, arch=None, code=None, cache_dir=None, include_dirs=[])
cuda_module.module = module_from_buffer(cubin_file)
cuda_module._bind_module()

### cuda funtions
depth_median_filter_w_cuda_kernel = cuda_module.get_function("depth_median_filter_w")
depth_median_filter_h_cuda_kernel = cuda_module.get_function("depth_median_filter_h")
flying_points_filter_cuda_kernel = cuda_module.get_function("flying_points_filter")

gpu_block_div_by_width = 4    # for image whose width is less than 4000. If image is larger, set it to 6 or 8.
depth_smoothing_filter_max_length = 2                      # from 0 - 6
flying_points_filter_checking_range = 0.005     # define the threshod for neighbour when checking for flying points
                                                # empirically, about 5-10 times of resolution per pxiel
flying_points_filter_minmum_points_in_checking_range = 10  # including the point itself, will also add a ratio of width // 300

def flying_points_filter_cuda(depth_map, depth_map_raw, height, width, camera_kd):
    flying_points_filter_cuda_kernel(depth_map, depth_map_raw,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        cuda.In(camera_kd), cuda.In(np.float32(flying_points_filter_checking_range)), cuda.In(np.int32(flying_points_filter_minmum_points_in_checking_range)),
        block=(width//gpu_block_div_by_width, 1, 1), grid=(height*gpu_block_div_by_width, 1))

def depth_median_filter_cuda(depth_map_mid_res, depth_map, height, width):
    depth_median_filter_h_cuda_kernel(depth_map_mid_res, depth_map,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        cuda.In(np.int32(depth_smoothing_filter_max_length)),
        block=(width//gpu_block_div_by_width, 1, 1), grid=(height*gpu_block_div_by_width, 1))
    depth_median_filter_w_cuda_kernel(depth_map, depth_map_mid_res,
        cuda.In(np.int32(height)), cuda.In(np.int32(width)),
        cuda.In(np.int32(depth_smoothing_filter_max_length)),
        block=(width//gpu_block_div_by_width, 1, 1), grid=(height*gpu_block_div_by_width, 1))

if __name__ == "__main__":
    depth_map = cv2.imread("./depth_example_raw.png", cv2.IMREAD_UNCHANGED)/30000.0
    camera_kd = np.loadtxt("./camera_kd.txt")
    depth_map = depth_map.astype(np.float32)
    cv2.imshow("depth", convert_depth_to_color(depth_map))
    cv2.waitKey()
    depth_map_out = np.empty_like(depth_map, dtype=np.float32)
    height, width = depth_map.shape[:2]
    depth_map_gpu = cuda.mem_alloc(depth_map.nbytes)
    depth_map_mid_res = cuda.mem_alloc(depth_map.nbytes)
    # copy to gpu
    cuda.memcpy_htod(depth_map_gpu, depth_map)

    # do the filtering
    start_time = time.time()
    flying_points_filter_cuda(depth_map_mid_res, depth_map_gpu, height, width, camera_kd.astype(np.float32))
    depth_median_filter_cuda(depth_map_gpu, depth_map_mid_res, height, width)
    print("depth filter: %.3f s" % (time.time() - start_time))

    # readout
    start_time = time.time()
    cuda.memcpy_dtoh(depth_map_out, depth_map_mid_res)
    print("readout from gpu: %.3f s" % (time.time() - start_time))

    # show the image
    cv2.imshow("depth", convert_depth_to_color(1000.0*depth_map_out))
    cv2.waitKey()
    import open3d as o3d
    fx, fy, cx, cy = camera_kd[0][0], camera_kd[1][1], camera_kd[0][2], camera_kd[1][2]
    color = cv2.imread("./color.bmp")
    pcd = gen_point_clouds_from_images(depth_map_out, camera_kd, color, save_path=None)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.translate(np.zeros(3), relative=False)
    cv2.destroyAllWindows()
    o3d.visualization.draw(geometry=pcd, width=1800, height=1000, point_size=2,
        bg_color=(0.5, 0.5, 0.5, 0.5), show_ui=False)
