
__device__ __forceinline__ float get_mid_val(float a, float b, float c)
{
    float max=a, min=a;
    if (b > max) max = b;
    if (c > max) max = c;
    if (b < min) min = b;
    if (c < min) min = c;
    return a+b+c-min-max;
}

// flying points filter
// a point could be considered as not flying when: points in checking range below max_distance > minmum num
__global__ void flying_points_filter(float *depth_map, float *depth_map_raw, int *height_array, int *width_array, float *camera_kp, float *depth_filter_max_distance, int *depth_filter_minmum_points_in_checking_range)
{
    // #define use_fast_distance_checking_for_flying_points_filter // use 3D distance (slower but more precisely) or only distance of axis-z to check flying points
            // enable this will save above 95% time compared with 3D distance checking, while can still remove most of the flying pts.
            // an example (3d vs only_z): render0000_2k avg error @ 10 mm thres: 0.1145mm vs 0.1152mm; cost time: 17ms vs 1ms on TitanRTX
    int height = height_array[0];
    int width = width_array[0];
    float max_distance = depth_filter_max_distance[0];
    int minmum_point_num_in_range = depth_filter_minmum_points_in_checking_range[0] + (width / 300) * (width / 300);
    float checking_range_in_meter = max_distance * 1.2;
    int checking_range_limit = width/50;
    float fx = camera_kp[0];
    float cx = camera_kp[2];
    float fy = camera_kp[1*3+1];
    float cy = camera_kp[1*3+2];

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = current_pix_idx / width;
    int w = current_pix_idx % width;
    float curr_pix_value = depth_map_raw[current_pix_idx];
    depth_map[current_pix_idx] = curr_pix_value;

    if (curr_pix_value != 0) {
        float point_x = curr_pix_value * (w - cx) / fx;
        float point_y = curr_pix_value * (h - cy) / fy;
        int checking_range_in_pix_x = (int)(checking_range_in_meter * fx / curr_pix_value);
        int checking_range_in_pix_y = (int)(checking_range_in_meter * fy / curr_pix_value);
        checking_range_in_pix_x = min(checking_range_in_pix_x, checking_range_limit);
        checking_range_in_pix_y = min(checking_range_in_pix_y, checking_range_limit);
        int is_not_flying_point_flag = 0;
        
        for (unsigned int i = max(0, h-checking_range_in_pix_y); i < min(height, h+checking_range_in_pix_y+1); i++) {
            int line_i_offset = i * width;
            for (unsigned int j = max(0, w-checking_range_in_pix_x); j < min(width, w+checking_range_in_pix_x+1); j++) {
                float checking_pix_value = depth_map_raw[line_i_offset + j];
                float z_diff = abs(curr_pix_value - checking_pix_value);
                if (checking_pix_value != 0.0 & z_diff < max_distance) {
                    #ifndef use_fast_distance_checking_for_flying_points_filter
                    float curr_x = checking_pix_value * (j - cx) / fx;
                    float curr_y = checking_pix_value * (i - cy) / fy;
                    float x_diff = curr_x - point_x, y_diff = curr_y - point_y;
                    float distance = (x_diff)*(x_diff) + (y_diff)*(y_diff) + (z_diff)*(z_diff);
                    if (distance < max_distance*max_distance) is_not_flying_point_flag += 1;
                    #else
                    is_not_flying_point_flag += 1;
                    #endif
                }
            }
            if (is_not_flying_point_flag > minmum_point_num_in_range) break;
        }
        if (is_not_flying_point_flag <= minmum_point_num_in_range) depth_map[current_pix_idx] = 0.0;
    }
}

__global__ void depth_median_filter_w(float *depth_map_out, float *depth_map, int *height_array, int *width_array, int *depth_filter_max_length)
{   // filter_max_length (ksize) fixed to 1 for now
    int height = height_array[0];
    int width = width_array[0];
    // int filter_max_length = 1; //depth_filter_max_length[0]; // 1, 2

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = current_pix_idx / width;
    int w = current_pix_idx % width;
    float curr_pix_value = depth_map[current_pix_idx];
    float mid_val = curr_pix_value;
    if (curr_pix_value != 0 & h != 0 & h!= height-1 & w !=0 & w != width-1) {
        if(depth_map[current_pix_idx-1] != 0 & depth_map[current_pix_idx+1] != 0) {
            mid_val = get_mid_val(depth_map[current_pix_idx-1], depth_map[current_pix_idx], depth_map[current_pix_idx+1]);
        }
    }
    depth_map_out[current_pix_idx] = mid_val;
}

__global__ void depth_median_filter_h(float *depth_map_out, float *depth_map, int *height_array, int *width_array, int *depth_filter_max_length)
{
    int height = height_array[0];
    int width = width_array[0];

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = current_pix_idx / width;
    int w = current_pix_idx % width;
    float curr_pix_value = depth_map[current_pix_idx];
    float mid_val = curr_pix_value;
    if (curr_pix_value != 0 & h != 0 & h!= height-1 & w !=0 & w != width-1) {
        if(depth_map[current_pix_idx-width] != 0 & depth_map[current_pix_idx+width] != 0) {
            mid_val = get_mid_val(depth_map[current_pix_idx-width], depth_map[current_pix_idx], depth_map[current_pix_idx+width]);
        }
    }
    depth_map_out[current_pix_idx] = mid_val;
}

__global__ void convert_dmap_to_mili_meter(float *depth_map)
{
    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    depth_map[current_pix_idx] = 1000.0*depth_map[current_pix_idx];
}
