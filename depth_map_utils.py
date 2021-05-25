import cv2
import numpy as np

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
        pcd_to_write.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=10))
        pcd_to_write.orient_normals_towards_camera_location()
        o3d.io.write_point_cloud(save_path, pcd_to_write, write_ascii=False, compressed=False)
        print("res saved to:" + save_path)
    return pcd

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
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