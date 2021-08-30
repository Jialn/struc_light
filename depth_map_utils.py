import cv2
import numpy as np

def calculate_mono_struli_para(depth_map, fx, index_map, line, col_range):
    # not working yet
    # usage example:
    # utils.calculate_mono_struli_para(depth_map, fx, from_gpu(img_index_left, size_sample=gray_left, dtype=np.float32),
    #    line=600, col_range=(420, 450))
    # col_range=(420, 450)
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
    # depth_map[600, test_line_valid_pts] = 0
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
                              use_morphology_closure=True, large_hole_Fill=False, use_median_filter=False):
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


############# HDR methods for raw patterns
def hdr_amplify_diff_of_inv(Config, img_list, img_list_high_exp, save_path=None):
    # hdr2: add (highexp_image - highexp_image_inv) to low exp image
    # for 7 + 1 + 4 phase shift only
    gray_code_range = (0, 10)
    # gray code
    for cnt in range(Config.pattern_start_index+gray_code_range[0], Config.pattern_start_index+gray_code_range[1]):
        img_list[cnt] =  img_list[cnt] // 2 + img_list_high_exp[cnt] // 2
        if Config.save_pattern_to_disk: cv2.imwrite(save_path + str(cnt) + "hdr.jpg", img_list[cnt])
    # phsft
    for cnt in range(Config.pattern_start_index+gray_code_range[1], Config.pattern_start_index+gray_code_range[1]+2, 1):
        high = img_list_high_exp[cnt].astype(np.int16)
        high_inv =  img_list_high_exp[cnt+2].astype(np.int16)
        high_diff = high - high_inv
        # high_diff = (high_diff * high_pattern_weight).astype(np.int16)
        high_diff_inv = - high_diff
        low = img_list[cnt].astype(np.int16)
        low_inv =  img_list[cnt+2].astype(np.int16)
        hdr = low + high_diff
        img_list[cnt] = np.clip(hdr, 0, 255).astype(np.uint8)
        hdr_inv = low_inv + high_diff_inv
        img_list[cnt+2] = np.clip(hdr_inv, 0, 255).astype(np.uint8)
        if Config.save_pattern_to_disk: cv2.imwrite(save_path + str(cnt) + "hdr.jpg", img_list[cnt])
        if Config.save_pattern_to_disk: cv2.imwrite(save_path + str(cnt+2) + "hdr.jpg", img_list[cnt+2])

def hdr_using_reflective_ratio(Config, img_list, img_list_high_exp, save_path=None):
    # for 7 + 1 + 4 phase shift only
    ref = img_list[Config.pattern_start_index+0] # 0 light on, 1 light off
    ref_mean = np.mean(ref)
    ref = ref / ref_mean  # normalize mean to 1.0
    print(ref_mean)
    ref_max = 255.0/ref_mean
    high_weight = ref_max - ref
    low_weight =  ref
    high_weight = high_weight * 0.5 / np.mean(high_weight)
    low_weight = low_weight * 0.5 / np.mean(low_weight)
    # gray code
    image_range = range(Config.pattern_start_index+0, Config.pattern_start_index+10)
    for cnt in image_range:
        img_list[cnt] =  img_list[cnt] // 2 + img_list_high_exp[cnt] // 2
        if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])
    # phsft
    image_range = range(Config.pattern_start_index+10, Config.pattern_start_index+14)
    if Config.use_high_speed_projector: # high spd projector needs fixed expo time, should use diff to elimate env light
        for cnt in image_range:
            diff_of_higher_prj = img_list_high_exp[cnt].astype(np.int16) - img_list[cnt]
            hdr = diff_of_higher_prj * high_weight + img_list[cnt] #  * low_weight
            img_list[cnt] = np.clip(hdr, 0, 255).astype(np.uint8)
            if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])
    else:
        for cnt in image_range:
            hdr = img_list_high_exp[cnt] * high_weight + img_list[cnt] * low_weight
            img_list[cnt] = np.clip(hdr, 0, 255).astype(np.uint8)
            if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])

def hdr_16bit(Config, img_list, img_list_high_exp, save_path=None):
    # return 16 bit hdr images
    # only for 7+1+4 phsft pattern
    # the following procedures should be able to handle 16bit unsigned short images
    image_range = range(Config.pattern_start_index+0, Config.pattern_start_index+14)
    for cnt in image_range:
        # low_expo_image = img_list[cnt].astype(np.uint16)
        # gamma_corrected_low_expo_image = (low_expo_image + 255) / 255 * low_expo_image
        gamma_corrected_low_expo_image = img_list[cnt].astype(np.uint16)
        hdr = (Config.hdr_high_exp_rate * gamma_corrected_low_expo_image).astype(np.uint16)
        low_expo_pts = np.where(img_list[cnt] < 32)
        hdr[low_expo_pts] = img_list_high_exp[cnt][low_expo_pts]
        img_list[cnt] = hdr
        if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])