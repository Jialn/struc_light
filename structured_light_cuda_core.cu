
__global__ void cuda_test(float *dest, float *a, float *b, float *offset) // a simple test function
{
    const int idx = threadIdx.x +  blockIdx.x*blockDim.x;
    dest[idx] = a[idx] + b[idx] + offset[0];
}

__global__ void gray_decode(unsigned char *src, unsigned char *avg_thres_posi, unsigned char *avg_thres_nega, unsigned char *valid_map, int *image_num, int *height, int *width, short *img_index, int *unvalid_thres)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (valid_map[idx] == 0) {
        img_index[idx] = -1;
        return;
    }
    int avg_thres = avg_thres_posi[idx]/2 + avg_thres_nega[idx]/2;
    int bin_code = 0;
    int current_bin_code_bit = 0;
    for (unsigned int i = 0; i < image_num[0]; i++) {
        int src_idx = idx + i * height[0] * width[0];
        if (src[src_idx]>=avg_thres+unvalid_thres[0]) current_bin_code_bit = current_bin_code_bit ^ 1;
        else if (src[src_idx]<=avg_thres-unvalid_thres[0]) current_bin_code_bit = current_bin_code_bit ^ 0;
        else {
            bin_code = -1;
            break;
        }
        bin_code += (current_bin_code_bit <<  (image_num[0]-1-i));
    }
    img_index[idx] = bin_code;
}

__global__ void phase_shift_decode(unsigned char *src, int *height, int *width, float *img_phase, short *img_index, int *unvalid_thres, float *phsift_pattern_period_per_pixel_array)
{
    float phsift_pattern_period_per_pixel = phsift_pattern_period_per_pixel_array[0];
    float unvalid_thres_diff = unvalid_thres[0];
    float outliers_checking_diff_thres = 10.0 + unvalid_thres_diff; //above this, will skip outlier checking
    const float pi = 3.14159265358979;

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (img_index[idx] == -1) {
        img_phase[idx] = nanf("");
        img_index[idx] = 0;  //reuse img_index as belief map
        return;
    }
    float i1 = src[idx];
    float i2 = src[idx + height[0] * width[0]];
    float i3 = src[idx + 2 * height[0] * width[0]];
    float i4 = src[idx + 3 * height[0] * width[0]];
    bool unvalid_flag = (abs(i4 - i2) <= unvalid_thres_diff & abs(i3 - i1) <= unvalid_thres_diff);
    bool need_outliers_checking_flag = (abs(i4 - i2) <= outliers_checking_diff_thres & abs(i3 - i1) <= outliers_checking_diff_thres);
    if (unvalid_flag) {
        img_phase[idx] = nanf("");
        img_index[idx] = 0;  //reuse img_index as belief map
        return;
    }
    float phase = - atan2f(i4-i2, i3-i1) + pi;
    int phase_main_index = img_index[idx] / 2 ;
    int phase_sub_index = img_index[idx] & 0x01;
    if((phase_sub_index == 0) & (phase > pi*1.5))  phase -= 2.0*pi; 
    if((phase_sub_index == 1) & (phase < pi*0.5))  phase += 2.0*pi; 
    img_phase[idx] = phase_main_index * phsift_pattern_period_per_pixel + (phase * phsift_pattern_period_per_pixel / (2*pi));
    //reuse img_index as belief map
    if (need_outliers_checking_flag) img_index[idx] = 0;
    else img_index[idx] = abs(i4 - i2) + abs(i3 - i1);  //reuse img_index as belief map, last bit is need_outliers_checking_flag
}

__global__ void rectify_phase_and_belief_map(float *img_phase, short *bfmap, float *rectify_map_x, float *rectify_map_y, int *height_array, int *width_array, float *rectified_img_phase, short *rectified_bfmap, float *sub_pixel_map_x)
{
    const bool use_interpo_for_y_aixs = true;
    int width = width_array[0], height = height_array[0];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int w = idx % width;
    float src_x = rectify_map_x[idx], src_y = rectify_map_y[idx];
    int round_y = int(src_y+0.499999), round_x = int(src_x+0.499999);
    int src_pix_idx = round_y*width + round_x;

    if (use_interpo_for_y_aixs) {
        int src_y_int = int(src_y);
        if (src_y_int == height-1) src_y_int = height - 2;
        float upper = img_phase[src_y_int*width+round_x];
        float lower = img_phase[src_y_int*width+round_x+width];
        float diff = lower - upper;
        if ( abs(diff) >= 1.0 | isnan(diff)) rectified_img_phase[idx] = img_phase[src_pix_idx];
        else rectified_img_phase[idx] = upper + diff * (src_y-src_y_int);
    }
    else rectified_img_phase[idx] = img_phase[src_pix_idx];
    rectified_bfmap[idx] = bfmap[src_pix_idx];
    sub_pixel_map_x[idx] = w + (round_x - src_x);
}

__global__ void gen_depth_from_index_matching(float *depth_map, int *height_array, int *width_array, float *img_index_left, float *img_index_right, float *baseline,float *dmap_base,float *fx, float *img_index_left_sub_px,float *img_index_right_sub_px, short *belief_map_l, short *belief_map_r, float *roughly_projector_area_in_image, float *depth_cutoff, int *remove_possibly_outliers_when_matching)
{
    float depth_cutoff_near = depth_cutoff[0], depth_cutoff_far = depth_cutoff[1];
    int width = width_array[0];
    float projector_area_ratio = roughly_projector_area_in_image[0];
    float index_thres_for_matching = 1.5 * 1280.0 / (width*projector_area_ratio);  //the smaller projector_area in image, the larger index_offset cloud be
    int right_corres_point_offset_range = (1.333 * projector_area_ratio * width) / 128;
    bool check_outliers = (remove_possibly_outliers_when_matching[0] != 0);
    // if another pixel has similar index( < index_thres_for_outliers_checking) has a distance > max_allow_pixel_per_index, consider it's an outlier 
    float max_allow_pixel_per_index_for_outliers_checking = 2.5 + 1.0 * projector_area_ratio * width / 1280.0;
    float index_thres_for_outliers_checking = index_thres_for_matching * 1.2;
    const bool use_belief_map_checking_when_matching = false;

    int h = blockIdx.x, stride = blockDim.x, offset = threadIdx.x;  //blockIdx.x is current working line; blockDim.x is stride
    int thread_work_length = width / blockDim.y;  //blockDim.y is the num of threads group per line
    int start = thread_work_length*threadIdx.y, end = thread_work_length+start;
    int line_start_addr_offset = h * width;
    float *line_r = img_index_right + line_start_addr_offset, *line_l = img_index_left + line_start_addr_offset;
    int last_right_corres_point = -1;
    for (int w = start+offset; w < end; w+=stride) {
        int curr_pix_idx = line_start_addr_offset + w;
        depth_map[curr_pix_idx] = 0.0;
        if (isnan(line_l[w])) {
            last_right_corres_point = -1;
            continue;
        }
        // find the nearest left and right corresponding points in right image
        int most_corres_pts_l = -1, most_corres_pts_r = -1;
        int checking_left_edge = 0, checking_right_edge = width;
        int cnt_l = 0, cnt_r = 0;
        float average_corres_position_in_thres_l = 0, average_corres_position_in_thres_r = 0;
        if (last_right_corres_point > 0) {
            checking_left_edge = last_right_corres_point - right_corres_point_offset_range;
            checking_right_edge = last_right_corres_point + right_corres_point_offset_range + stride;
            if (checking_left_edge <=0) checking_left_edge=0;
            if (checking_right_edge >=width) checking_right_edge=width;
            for (int i=checking_left_edge; i < checking_right_edge; i++) {  // fast checking around last_right_corres_point
                if (isnan(line_r[i])) continue;
                if (use_belief_map_checking_when_matching) {
                    float bfmap_thres = 10 + (belief_map_l[curr_pix_idx] + belief_map_r[line_start_addr_offset+i]) * 0.75;
                    if (abs(belief_map_l[curr_pix_idx] - belief_map_r[line_start_addr_offset+i]) >= bfmap_thres) continue;
                }
                float thres = index_thres_for_matching + abs(img_index_left_sub_px[line_start_addr_offset+w] - w - img_index_right_sub_px[line_start_addr_offset+i] + i)/projector_area_ratio;
                if ((line_l[w]-thres <= line_r[i]) & (line_r[i] <= line_l[w])) {
                    if (most_corres_pts_l==-1) most_corres_pts_l = i;
                    else if (line_r[i] >= line_r[most_corres_pts_l]) most_corres_pts_l = i;
                    cnt_l += 1;
                    average_corres_position_in_thres_l += i;
                }
                else if ((line_l[w] <= line_r[i]) & (line_r[i] <= line_l[w]+thres)) {
                    if (most_corres_pts_r==-1) most_corres_pts_r = i;
                    else if (line_r[i] <= line_r[most_corres_pts_r]) most_corres_pts_r = i;
                    cnt_r += 1;
                    average_corres_position_in_thres_r += i;
                }
            }
        }
        // last_right_corres_point is invalid or not found most_corres_pts, expand the searching range and try searching again
        if (most_corres_pts_l == -1 & most_corres_pts_r == -1) {
            for (int i=0; i < width; i++) { 
                if (isnan(line_r[i])) continue;
                if (use_belief_map_checking_when_matching) {
                    float bfmap_thres = 10 + (belief_map_l[curr_pix_idx] + belief_map_r[line_start_addr_offset+i]) * 0.75;
                    if (abs(belief_map_l[curr_pix_idx] - belief_map_r[line_start_addr_offset+i]) >= bfmap_thres) continue;
                }
                // if (!belief_map_r[line_start_addr_offset+i]) continue;
                if ((line_l[w]-index_thres_for_matching <= line_r[i]) & (line_r[i] <= line_l[w])) {
                    if (most_corres_pts_l==-1) most_corres_pts_l = i;
                    else if (line_r[i] >= line_r[most_corres_pts_l]) most_corres_pts_l = i;
                    cnt_l += 1;
                    average_corres_position_in_thres_l += i;
                }
                else if ((line_l[w] <= line_r[i]) & (line_r[i] <= line_l[w]+index_thres_for_matching)) {
                    if (most_corres_pts_r==-1) most_corres_pts_r = i;
                    else if (line_r[i] <= line_r[most_corres_pts_r]) most_corres_pts_r = i;
                    cnt_r += 1;
                    average_corres_position_in_thres_r += i;
                }
            }
        }
        // get the right index
        float w_r = 0;
        bool outliers_flag = false;
        if (most_corres_pts_l == -1 & most_corres_pts_r == -1) continue;
        else if (most_corres_pts_l==-1) w_r = img_index_right_sub_px[line_start_addr_offset+most_corres_pts_r]+0.2; // add 0.2 pix offset as we know it's on the right side
        else if (most_corres_pts_r==-1) w_r = img_index_right_sub_px[line_start_addr_offset+most_corres_pts_l]-0.2;
        else {
            // get the interpo right index 'w_r'
            float left_pos = line_r[most_corres_pts_l], right_pos = line_r[most_corres_pts_r];
            float left_value = img_index_right_sub_px[line_start_addr_offset+most_corres_pts_l], right_value = img_index_right_sub_px[line_start_addr_offset+most_corres_pts_r];
            if (right_pos-left_pos != 0) w_r = left_value + (right_value-left_value) * (line_l[w]-left_pos)/(right_pos-left_pos);
            else w_r = left_value;
        }
        if (cnt_l != 0) average_corres_position_in_thres_l = average_corres_position_in_thres_l / cnt_l;
        if (cnt_r != 0) average_corres_position_in_thres_r = average_corres_position_in_thres_r / cnt_r;
        // check possiblely outliers using max_allow_pixel_per_index and belief_map
        if (check_outliers==true & belief_map_r[line_start_addr_offset+(int)(w_r+0.499999)]==0) {  // & belief_map_r[line_start_addr_offset+(int)(w_r+0.499999)]==0
            if (most_corres_pts_l != -1 & abs((float)(most_corres_pts_l-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
            if (most_corres_pts_r != -1 & abs((float)(most_corres_pts_r-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
            if (average_corres_position_in_thres_l != 0 & abs((float)(average_corres_position_in_thres_l-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
            if (average_corres_position_in_thres_r != 0 & abs((float)(average_corres_position_in_thres_r-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
        }
        if (outliers_flag==true) continue;
        last_right_corres_point = (int)(w_r+0.499999);
        // get left index
        float w_l = img_index_left_sub_px[curr_pix_idx];
        // check possiblely left outliers
        if (check_outliers==true & belief_map_l[curr_pix_idx]==0) {
            for (int i=0; i < width; i++) {
                if ((line_l[w]-index_thres_for_outliers_checking <= line_l[i]) & (line_l[i] <= line_l[w]+index_thres_for_outliers_checking)) {
                    if (abs((float)(w_l-i)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
                }
            }
        }
        if (outliers_flag==true) continue;
        // get stereo diff and depth
        float stereo_diff = dmap_base[0] + w_l - w_r;
        if (stereo_diff > 0.000001) {
            float depth = fx[0] * baseline[0] / stereo_diff;
            if ((depth_cutoff_near < depth) & (depth < depth_cutoff_far)) depth_map[curr_pix_idx] = depth;
        }
    }
}

__global__ void optimize_dmap_using_sub_pixel_map(float *depth_map, float *optimized_depth_map, int *height_array, int *width_array, float *img_index_left_sub_px)
{
    // interpo for depth map using sub_pixel
    // this does not improve a lot on rendered data because no distortion and less stereo rectify for left camera, but useful for real captures
    int width = width_array[0];
    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int w = current_pix_idx % width;
    if (w == 0 | w == width-1) {
        optimized_depth_map[current_pix_idx] = depth_map[current_pix_idx];
        return;
    }

    float left_value = 0.0, right_value = 0.0;
    float left_pos = 0.0, right_pos = 0.0;
    float real_pos_for_current_depth = img_index_left_sub_px[current_pix_idx];
    if (depth_map[current_pix_idx] <= 0.00001) {
        if (depth_map[current_pix_idx-1] >= 0.00001 & depth_map[current_pix_idx+1] >= 0.00001) {
            right_pos = img_index_left_sub_px[current_pix_idx+1];
            right_value = depth_map[current_pix_idx+1];
            left_pos = img_index_left_sub_px[current_pix_idx-1];
            left_value = depth_map[current_pix_idx-1];
        }
    }
    else if (real_pos_for_current_depth >= w) {
        right_pos = real_pos_for_current_depth;
        right_value = depth_map[current_pix_idx];
        left_pos = img_index_left_sub_px[current_pix_idx-1];
        left_value = depth_map[current_pix_idx-1];
    }
    else {
        right_pos = img_index_left_sub_px[current_pix_idx+1];
        right_value = depth_map[current_pix_idx+1];
        left_pos = real_pos_for_current_depth;
        left_value = depth_map[current_pix_idx];
    }
    if (left_value >= 0.00001 & right_value >= 0.00001) {
        optimized_depth_map[current_pix_idx] = left_value + (right_value-left_value) * (w-left_pos)/(right_pos-left_pos);
    }
    else {
        optimized_depth_map[current_pix_idx] = 0.0;
    }
}

__global__ void flying_points_filter(float *depth_map, float *depth_map_raw, int *height_array, int *width_array, float *camera_kp, float *depth_filter_max_distance, int *depth_filter_minmum_points_in_checking_range, int *belief_map)
{
    // a point could be considered as not flying when: points in checking range below max_distance > minmum num 
    const bool use_3d_distance = false; // use 3D distance (slower but more precisely) or only distance of axis-z to check flying points
                                        // setting to false will save above 95% time compared with 3D distance checking, while can still remove most of the flying pts.
                                        // an example (3d vs only_z): render0000_2k avg error @ 10 mm thres: 0.1145mm vs 0.1152mm; cost time: 17ms vs 1ms; total time 80ms vs 66ms
    int height = height_array[0];
    int width = width_array[0];
    float max_distance = depth_filter_max_distance[0];
    int minmum_point_num_in_range = depth_filter_minmum_points_in_checking_range[0] + (width / 400) * (width / 400);
    float checking_range_in_meter = max_distance * 1.2;
    int checking_range_limit = width/50;
    float fx = camera_kp[0];
    float cx = camera_kp[2];
    float fy = camera_kp[1*3+1];
    float cy = camera_kp[1*3+2];

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = blockIdx.x / 4;
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
                    if (use_3d_distance) {
                        float curr_x = checking_pix_value * (j - cx) / fx;
                        float curr_y = checking_pix_value * (i - cy) / fy;
                        float x_diff = curr_x - point_x, y_diff = curr_y - point_y;
                        float distance = (x_diff)*(x_diff) + (y_diff)*(y_diff) + (z_diff)*(z_diff);
                        if (distance < max_distance*max_distance) is_not_flying_point_flag += 1;
                    }
                    else is_not_flying_point_flag += 1; 
                }
            }
            if (is_not_flying_point_flag > minmum_point_num_in_range) break;
        }
        if (is_not_flying_point_flag <= minmum_point_num_in_range) depth_map[current_pix_idx] = 0.0;
    }
}

__global__ void depth_filter(float *depth_map, int *height_array, int *width_array, int *depth_filter_max_length, float *depth_filter_unvalid_thres, int *belief_map)
{
    // a point could be considered as not flying when: points in checking range below max_distance > minmum num 
    int height = height_array[0];
    int width = width_array[0];
    int filter_max_length = depth_filter_max_length[0];
    float filter_thres = depth_filter_unvalid_thres[0];
    const float filter_weights[6] = {1.0, 0.8, 0.6, 0.5, 0.4, 0.2};

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = blockIdx.x / 4;
    int w = current_pix_idx % width;
    float curr_pix_value = depth_map[current_pix_idx];
    if (curr_pix_value != 0) {
        int line_start_addr_offset = h * width;
        // horizontal
        float left_weight = 0.0, right_weight = 0.0, depth_sum = curr_pix_value*filter_weights[0];
        bool stop_flag = false;
        for (int i=1; i< filter_max_length+1; i++) {
            int l_idx = w-i, r_idx = w+i;
            // if (belief_map[current_pix_idx] == 1) filter_thres = depth_filter_unvalid_thres[0];
            // else filter_thres = depth_filter_unvalid_thres[0] * 3;
            if(depth_map[line_start_addr_offset+l_idx] != 0 & depth_map[line_start_addr_offset+r_idx] != 0 & l_idx > 0 & r_idx < width & \
                abs(depth_map[line_start_addr_offset+l_idx] - curr_pix_value) < filter_thres & abs(depth_map[line_start_addr_offset+r_idx] - curr_pix_value) < filter_thres) {
                left_weight += filter_weights[i];
                right_weight += filter_weights[i];
                depth_sum += (depth_map[line_start_addr_offset+r_idx] + depth_map[line_start_addr_offset+l_idx]) * filter_weights[i];
            }
            else {
                stop_flag = true; 
                break;
            }
        }
        if (!stop_flag) depth_map[current_pix_idx] = depth_sum / (filter_weights[0] + left_weight + right_weight);
        __threadfence();  // make sure the modification to depth_map is visiable for all other threads
        __syncthreads();
        // vertical
        curr_pix_value = depth_map[current_pix_idx];
        left_weight = 0.0, right_weight = 0.0, depth_sum = curr_pix_value*filter_weights[0];
        stop_flag = false;
        for (int i=1; i< filter_max_length+1; i++) {
            int l_idx = h-i, r_idx = h+i;
            //if (belief_map[current_pix_idx] == 1) filter_thres = depth_filter_unvalid_thres[0];
            //else filter_thres = depth_filter_unvalid_thres[0] * 3;
            if(depth_map[l_idx*width+w] != 0 & depth_map[r_idx*width+w] != 0 & l_idx > 0 & r_idx < height & \
                abs(depth_map[l_idx*width+w] - curr_pix_value) < filter_thres & abs(depth_map[r_idx*width+w] - curr_pix_value) < filter_thres) {
                left_weight += filter_weights[i];
                right_weight += filter_weights[i];
                depth_sum += (depth_map[r_idx*width+w] + depth_map[l_idx*width+w]) * filter_weights[i];
            }
            else {
                stop_flag = true; 
                break;
            }
        }
        if (!stop_flag) depth_map[current_pix_idx] = depth_sum / (filter_weights[0] + left_weight + right_weight);
    }
}

__global__ void convert_dmap_to_mili_meter(float *depth_map)
{
    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    depth_map[current_pix_idx] = 1000.0*depth_map[current_pix_idx];
}
