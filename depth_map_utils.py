import cv2
import numpy as np

def calculate_mono_struli_para(depth_map, fx, index_map, line, col_range):
    # usage example:
    # utils.calculate_mono_struli_para(depth_map, fx, from_gpu(img_index_left, size_sample=gray_left, dtype=np.float32),
    #    line=600, col_range=(320, 360))
    from scipy.optimize import least_squares
    test_line = depth_map[line,]
    test_line_index = index_map[line,]
    test_line_valid_pts = np.where(test_line > 0.1)[0][col_range[0]:col_range[1]]
    w_array = test_line_valid_pts
    index_value_array = test_line_index[test_line_valid_pts]
    depth_array = test_line[test_line_valid_pts]
    print("w_array", w_array)
    print("index_value_array", index_value_array)
    print("depth_array", depth_array)
    
    def residuals_for_disparity(p, w_array, index_value_array, depth_array):
        baseline_to_prjector = 0.14
        a, b, c = p
        return (w_array - (a * index_value_array * index_value_array + b * index_value_array + c)) - (fx * baseline_to_prjector / depth_array)
    
    def residuals_for_depth(p, w_array, index_value_array, depth_array):
        # print("call")
        baseline_to_prjector, a, b, c= p
        return depth_array - fx * baseline_to_prjector / (w_array - (a*(index_value_array/1280) + b)* index_value_array - c)
    
    residuals = residuals_for_depth
    p0 = [125, 1.0, -100, 640]
    print("residuals before", residuals(p0, w_array, index_value_array, depth_array))
    leastsq_res = least_squares(residuals, p0, args=(w_array, index_value_array, depth_array), method='trf', jac='3-point',
        ftol=1e-15, xtol=1e-15, gtol=1e-15, x_scale=1.0, loss='soft_l1')
    # baseline_to_prjector, a, b = leastsq_res.x
    print("residuals after", residuals(leastsq_res.x, w_array, index_value_array, depth_array))
    print(leastsq_res)
    return leastsq_res.x

def convert_depth_to_color(depth_map_mm, scale=None):
    h, w = depth_map_mm.shape[:2]
    depth_image_color_vis = depth_map_mm.copy()
    valid_points = np.where(depth_image_color_vis>=0.1)
    # depth_near_cutoff, depth_far_cutoff = np.min(depth_image_color_vis[valid_points]), np.max(depth_image_color_vis[valid_points])
    depth_near_cutoff, depth_far_cutoff = np.percentile(depth_image_color_vis[valid_points], 1), np.percentile(depth_image_color_vis[valid_points], 99)
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

def report_depth_error(depth_img, depth_gt, image_path, default_image_seq_start_index, save_mid_res_for_visulize, res_path):
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


# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_16 = np.ones((16, 16), np.uint8)
def rect_kernel(m, n):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (m, n))

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
    
def cross_kernel(n):
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (n, n))

def depth_map_post_processing(depth_map, max_depth=3000.0,
                              use_morphology_closure=False, large_hole_Fill=False, use_median_filter=True):
    """Optional fill small holes and additional noise removal that provides better qualitative results.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion, in mili-meter
        use_morphology_closure: use morphology closure to fill small holes
        large_hole_Fill: use morphology closure to fill large holes
        use_median_filter: use median filter to remove outliers
    Returns:
        depth_map: the processed depth map
    """
    # Invert (and offset)
    valid_pixels = (depth_map > 0.1)
    inverted_depths = np.copy(depth_map)
    inverted_depths[valid_pixels] = max_depth - inverted_depths[valid_pixels]

    # Small hole closure
    if use_morphology_closure:
        inverted_depths = cv2.morphologyEx(inverted_depths, cv2.MORPH_CLOSE, CROSS_KERNEL_3)

    # Large hole fill
    if large_hole_Fill:
        # Get empty mask
        valid_pixels = (inverted_depths > 0.1)
        empty_pixels = ~valid_pixels
        # Hole fill
        dilated = cv2.dilate(inverted_depths, FULL_KERNEL_9)
        inverted_depths[empty_pixels] = dilated[empty_pixels]

    # Median filter to remove outliers
    if use_median_filter:
        blurred = cv2.medianBlur(inverted_depths, 3)
        valid_pixels = (inverted_depths > 0.1)
        inverted_depths[valid_pixels] = blurred[valid_pixels]

    depths_out = np.copy(inverted_depths)
    valid_pixels = np.where(depths_out > 0.1)
    depths_out[valid_pixels] = max_depth - depths_out[valid_pixels]

    return depths_out