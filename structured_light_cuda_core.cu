
// a simple test function
__global__ void cuda_test(float *dest, float *a, float *b, float *offset) 
{
    const int idx = threadIdx.x +  blockIdx.x*blockDim.x;
    dest[idx] = a[idx] + b[idx] + offset[0];
}

// a function to convert RGGB bayer image to single blue channle image for stru-light
__global__ void convert_bayer_to_blue(unsigned char *src, int *height_array, int *width_array)
{   
    int width = width_array[0], width_half = width_array[0] / 2, height_div2 = height_array[0] / 2;
    int idx_div4 = threadIdx.x + blockIdx.x*blockDim.x;
    int h_div2 = idx_div4 / width_half, w_div2 = idx_div4 % width_half;
    int idx = h_div2 * 2 * width + w_div2 * 2;
    if (h_div2 % height_div2 != 0) {    // if not the first line
        src[idx] = ((int)src[idx-width-1] + (int)src[idx-width+1] + (int)src[idx+width-1] + (int)src[idx+width+1]+2) / 4;  // R
        src[idx+1] = ((int)src[idx-width+1] + (int)src[idx+width+1]+1) /2;  //G
    }
    else {
        src[idx] = ((int)src[idx+width-1] + (int)src[idx+width+1]+1) / 2;   //R
        src[idx+1] = src[idx+width+1];                                      //G
    }
    src[idx+width] = ((int)src[idx+width-1] + (int)src[idx+width+1]+1) / 2; //G
    //src[idx+width+1] = src[idx+width+1]; //B
}

// a function to convert RGGB bayer image to single channle gray image for stru-light
__global__ void convert_bayer_to_gray(unsigned char *src, int *height_array, int *width_array)
{   
    int width = width_array[0], width_half = width_array[0] / 2, height_div2 = height_array[0] / 2;
    int idx_div4 = threadIdx.x + blockIdx.x*blockDim.x;
    int h_div2 = idx_div4 / width_half, w_div2 = idx_div4 % width_half;
    int idx = h_div2 * 2 * width + w_div2 * 2;
    int r_value[4], b_value[4], g_value[4];
    int idx_r=idx, idx_g=idx+1, idx_g2=idx+width, idx_b=idx+width+1;
    if (h_div2 % height_div2 == 0) {    // if the first line
        // R
        r_value[0] = src[idx];
        b_value[0] = ((int)src[idx+width-1] + (int)src[idx+width+1]+1) / 2;
        if (w_div2 == 0) g_value[0] = ((int)src[idx+1] + (int)src[idx+width] + 1) / 2;
        else g_value[0] = ((int)src[idx-1] + (int)src[idx+1] + (int)src[idx+width] + 1) / 3;
        // G
        r_value[1] = ((int)src[idx_g-1] + (int)src[idx_g+1] + 1) / 2;
        g_value[1] = src[idx_g];
        b_value[1] = src[idx_g+width];
        // G
        r_value[2] = src[idx_g2+width];
        g_value[2] = src[idx_g2];
        b_value[2] = ((int)src[idx_g2-1] + (int)src[idx_g2+1] + 1) / 2;
        // B
        r_value[3] = ((int)src[idx_b+width-1] + (int)src[idx_b+width+1] + 1) / 2;
        g_value[3] = ((int)src[idx_b+width] + (int)src[idx_b-1] + (int)src[idx_b+1] + 2) / 3;
        b_value[3] = src[idx_b];
    }
    else if (h_div2 % height_div2 == height_div2-1) {   // if the last line
        // R
        r_value[0] = src[idx];
        g_value[0] = ((int)src[idx-width] + (int)src[idx-1] + (int)src[idx+1] + 1) / 3;
        if (w_div2 == width_half-1) b_value[0] = src[idx-width-1];
        else b_value[0] = ((int)src[idx-width-1] + (int)src[idx-width+1] + 1) / 2;
        // G
        r_value[1] = ((int)src[idx_g-1] + (int)src[idx_g+1] + 1) / 2;
        g_value[1] = src[idx_g];
        b_value[1] = ((int)src[idx_g-width] + (int)src[idx_g+width] + 1) /2;
        // G
        r_value[2] = src[idx_g2-width];
        g_value[2] = src[idx_g2];
        b_value[2] = ((int)src[idx_g2-1] + (int)src[idx_g2+1] + 1) /2;
        // B
        r_value[3] = ((int)src[idx_b-width-1] + (int)src[idx_b-width+1] + 1) / 2;
        g_value[3] = ((int)src[idx_b-width] + (int)src[idx_b-1] + (int)src[idx_b+1] + 2) / 4;
        b_value[3] = src[idx_b];

    }
    else {
        // R
        r_value[0] = src[idx];
        g_value[0] = ((int)src[idx-width] + (int)src[idx+width] + (int)src[idx-1] + (int)src[idx+1] + 2) / 4;
        b_value[0] = ((int)src[idx-width-1] + (int)src[idx-width+1] + (int)src[idx+width-1] + (int)src[idx+width+1] + 2) / 4;
        // G
        r_value[1] = ((int)src[idx_g-1] + (int)src[idx_g+1] + 1) / 2;
        g_value[1] = src[idx_g];
        b_value[1] = ((int)src[idx_g-width] + (int)src[idx_g+width] + 1) / 2;
        // G
        r_value[2] = ((int)src[idx_g2-width] + (int)src[idx_g2+width] + 1) / 2;
        g_value[2] = src[idx_g2];
        b_value[2] = ((int)src[idx_g2-1] + (int)src[idx_g2+1] + 1) / 2;
        // B
        r_value[3] = ((int)src[idx_b-width-1] + (int)src[idx_b-width+1] + (int)src[idx_b+width-1] + (int)src[idx_b+width+1] + 2) / 4;
        g_value[3] = ((int)src[idx_b-width] + (int)src[idx_b+width] + (int)src[idx_b-1] + (int)src[idx_b+1] + 2) / 4;
        b_value[3] = src[idx_b];
    }
    src[idx_r]  = (r_value[0] + g_value[0] + b_value[0] + 1) / 3;
    src[idx_g]  = (r_value[1] + g_value[1] + b_value[1] + 1) / 3;
    src[idx_g2] = (r_value[2] + g_value[2] + b_value[2] + 1) / 3;
    src[idx_b]  = (r_value[3] + g_value[3] + b_value[3] + 1) / 3;
}

__global__ void gray_decode(unsigned char *src, unsigned char *avg_thres_posi, unsigned char *avg_thres_nega, unsigned char *valid_map, int *image_num,
    int *height, int *width, short *img_index, int *unvalid_thres)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    bool pix_is_valid = ((int)avg_thres_posi[idx] - (int)avg_thres_nega[idx]) > unvalid_thres[0];
    // valid_map[idx] = pix_is_valid*255;  // if visulize valid map is needed
    if (! pix_is_valid) {
        img_index[idx] = -1;
        return;
    }
    int avg_thres = ((int)avg_thres_posi[idx] + (int)avg_thres_nega[idx] + 1) / 2;
    int bin_code = 0;
    int current_bin_code_bit = 0;
    for (unsigned int i = 0; i < image_num[0]; i++) {
        int src_idx = idx + i * height[0] * width[0];
        if (src[src_idx]>=avg_thres) current_bin_code_bit = current_bin_code_bit ^ 1;
        else if (src[src_idx]<=avg_thres) current_bin_code_bit = current_bin_code_bit ^ 0;
        else {
            bin_code = -1;
            break;
        }
        bin_code += (current_bin_code_bit <<  (image_num[0]-1-i));
    }
    img_index[idx] = bin_code;
}

// 16bit version of gray_decode
__global__ void gray_decode_hdr(unsigned short *src, unsigned short *avg_thres_posi, unsigned short *avg_thres_nega, unsigned char *valid_map,
    int *image_num, int *height, int *width, short *img_index, int *unvalid_thres)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    bool pix_is_valid = ((int)avg_thres_posi[idx] - (int)avg_thres_nega[idx]) > unvalid_thres[0];
    // valid_map[idx] = pix_is_valid*255;  // if visulize valid map is needed
    if (! pix_is_valid) {
        img_index[idx] = -1;
        return;
    }
    int avg_thres = ((int)avg_thres_posi[idx] + (int)avg_thres_nega[idx] + 1) / 2;
    int bin_code = 0;
    int current_bin_code_bit = 0;
    for (unsigned int i = 0; i < image_num[0]; i++) {
        int src_idx = idx + i * height[0] * width[0];
        if (src[src_idx]>=avg_thres) current_bin_code_bit = current_bin_code_bit ^ 1;
        else if (src[src_idx]<=avg_thres) current_bin_code_bit = current_bin_code_bit ^ 0;
        else {
            bin_code = -1;
            break;
        }
        bin_code += (current_bin_code_bit <<  (image_num[0]-1-i));
    }
    img_index[idx] = bin_code;
}

#define PI 3.14159265358979
// #define gamma_linear_correction_for_phsft_decode
__global__ void phase_shift_decode(unsigned char *src, int *height, int *width, float *img_phase, short *img_index,
    int *unvalid_thres, float *phsift_pattern_period_per_pixel_array)
{
    float phsift_pattern_period_per_pixel = phsift_pattern_period_per_pixel_array[0];
    float unvalid_thres_diff = unvalid_thres[0];
    float outliers_checking_diff_thres = 10.0 + unvalid_thres_diff; //above this, will skip outlier re-check when matching

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (img_index[idx] == -1) {
        img_phase[idx] = nanf("");
        img_index[idx] = 0;  //reuse img_index as belief map
        return;
    }
    float i1 = src[idx], i2 = src[idx + height[0] * width[0]];
    float i3 = src[idx + 2 * height[0] * width[0]], i4 = src[idx + 3 * height[0] * width[0]];
    // the gray value diff of inv pattern(phaseshift is pi) can be seen as belief_value
    int belief_value = abs(i4 - i2) + abs(i3 - i1);
    bool unvalid_flag = (belief_value <= unvalid_thres_diff);
    if (unvalid_flag) {
        img_phase[idx] = nanf("");
        img_index[idx] = 0;
        return;
    }
    #ifdef gamma_linear_correction_for_phsft_decode
    float gamma_correction = 2.4;
    float i1_c = powf(i1, gamma_correction);
    float i2_c = powf(i2, gamma_correction);
    float i3_c = powf(i3, gamma_correction);
    float i4_c = powf(i4, gamma_correction);
    float phase = - atan2f(i4_c-i2_c, i3_c-i1_c) + PI;
    #else
    float phase = - atan2f(i4-i2, i3-i1) + PI;
    #endif
    int phase_main_index = img_index[idx] / 2 ;
    int phase_sub_index = img_index[idx] & 0x01;
    if((phase_sub_index == 0) & (phase > PI*1.5))  phase -= 2.0*PI; 
    if((phase_sub_index == 1) & (phase < PI*0.5))  phase += 2.0*PI; 
    img_phase[idx] = phase_main_index * phsift_pattern_period_per_pixel + (phase * phsift_pattern_period_per_pixel / (2*PI));
    //reuse img_index as belief map
    bool need_outliers_checking_flag = (belief_value <= outliers_checking_diff_thres);
    if (need_outliers_checking_flag) img_index[idx] = 0;  // can not trust
    else img_index[idx] = belief_value;
}

// 16bit version of phase_shift_decode
__global__ void phase_shift_decode_hdr(unsigned short *src, int *height, int *width, float *img_phase, short *img_index, int *unvalid_thres, float *phsift_pattern_period_per_pixel_array)
{
    float phsift_pattern_period_per_pixel = phsift_pattern_period_per_pixel_array[0];
    float unvalid_thres_diff = unvalid_thres[0];
    float outliers_checking_diff_thres = 10.0 + unvalid_thres_diff; //above this, will skip outlier re-check when matching
    float gamma_correction = 2.0;

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (img_index[idx] == -1) {
        img_phase[idx] = nanf("");
        img_index[idx] = 0;  //reuse img_index as belief map
        return;
    }
    float i1 = src[idx], i2 = src[idx + height[0] * width[0]];
    float i3 = src[idx + 2 * height[0] * width[0]], i4 = src[idx + 3 * height[0] * width[0]];
    int belief_value = abs(i4 - i2) + abs(i3 - i1);
    bool unvalid_flag = (belief_value <= unvalid_thres_diff);
    if (unvalid_flag) {
        img_phase[idx] = nanf("");
        img_index[idx] = 0;
        return;
    }
    float i1_c = powf(i1, gamma_correction);
    float i2_c = powf(i2, gamma_correction);
    float i3_c = powf(i3, gamma_correction);
    float i4_c = powf(i4, gamma_correction);
    float phase = - atan2f(i4_c-i2_c, i3_c-i1_c) + PI;
    int phase_main_index = img_index[idx] / 2 ;
    int phase_sub_index = img_index[idx] & 0x01;
    if((phase_sub_index == 0) & (phase > PI*1.5))  phase -= 2.0*PI; 
    if((phase_sub_index == 1) & (phase < PI*0.5))  phase += 2.0*PI; 
    img_phase[idx] = phase_main_index * phsift_pattern_period_per_pixel + (phase * phsift_pattern_period_per_pixel / (2*PI));
    //reuse img_index as belief map
    bool need_outliers_checking_flag = (belief_value <= outliers_checking_diff_thres);
    if (need_outliers_checking_flag) img_index[idx] = 0;
    else img_index[idx] = belief_value;
}

#define use_interpo_for_y_aixs
__global__ void rectify_phase_and_belief_map(float *img_phase, short *bfmap, float *rectify_map_x, float *rectify_map_y, int *height_array, int *width_array, float *rectified_img_phase, short *rectified_bfmap, float *sub_pixel_map_x)
{
    int width = width_array[0], height = height_array[0];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int w = idx % width;
    float src_x = rectify_map_x[idx], src_y = rectify_map_y[idx];
    int round_y = int(src_y+0.499999), round_x = int(src_x+0.499999);
    int src_pix_idx = round_y*width + round_x;

    #ifdef use_interpo_for_y_aixs
        int src_y_int = int(src_y);
        if (src_y_int == height-1) src_y_int = height - 2;
        float upper = img_phase[src_y_int*width+round_x];
        float lower = img_phase[src_y_int*width+round_x+width];
        float diff = lower - upper;
        if ( abs(diff) >= 1.0 | isnan(diff)) rectified_img_phase[idx] = img_phase[src_pix_idx];
        else rectified_img_phase[idx] = upper + diff * (src_y-src_y_int);
    #else
        rectified_img_phase[idx] = img_phase[src_pix_idx];
    #endif
    rectified_bfmap[idx] = bfmap[src_pix_idx];
    sub_pixel_map_x[idx] = w + (round_x - src_x);
}

__device__ __forceinline__ void pix_index_matching(float *line_l, float *line_r, int w, int curr_pix_idx, int i, int line_start_addr_offset,
    float thres, short *belief_map_l, short *belief_map_r, int *most_corres_pts_l, int *most_corres_pts_r, int *most_corres_pts_l_bf,
    int *most_corres_pts_r_bf, int *cnt_l, int *cnt_r, float *average_corres_position_in_thres_l, float *average_corres_position_in_thres_r)
{
    if ((line_l[w]-thres <= line_r[i]) & (line_r[i] <= line_l[w])) {
        if (*most_corres_pts_l==-1) *most_corres_pts_l = i;
        else if (line_r[i] >= line_r[*most_corres_pts_l]) *most_corres_pts_l = i;
        #ifdef use_belief_map_for_checking
        int bfmap_thres = 10 + (belief_map_l[curr_pix_idx] + belief_map_r[line_start_addr_offset+i]) / 2;
        if (abs(belief_map_l[curr_pix_idx] - belief_map_r[line_start_addr_offset+i]) < bfmap_thres) {
            if (*most_corres_pts_l_bf==-1) *most_corres_pts_l_bf = i;
            else if (line_r[i] >= line_r[*most_corres_pts_l_bf]) *most_corres_pts_l_bf = i;
        }
        #endif
        *cnt_l += 1; 
        *average_corres_position_in_thres_l += i;
    }
    else if ((line_l[w] <= line_r[i]) & (line_r[i] <= line_l[w]+thres)) {
        if (*most_corres_pts_r==-1) *most_corres_pts_r = i;
        else if (line_r[i] <= line_r[*most_corres_pts_r]) *most_corres_pts_r = i;
        #ifdef use_belief_map_for_checking
        int bfmap_thres = 10 + (belief_map_l[curr_pix_idx] + belief_map_r[line_start_addr_offset+i]) / 2;
        if (abs(belief_map_l[curr_pix_idx] - belief_map_r[line_start_addr_offset+i]) < bfmap_thres) {
            if (*most_corres_pts_r_bf==-1) *most_corres_pts_r_bf = i;
            else if (line_r[i] <= line_r[*most_corres_pts_r_bf]) *most_corres_pts_r_bf = i;
        }
        #endif
        *cnt_r += 1;
        *average_corres_position_in_thres_r += i;
    }
}

__global__ void gen_depth_from_index_matching(float *depth_map, int *height_array, int *width_array, float *img_index_left, float *img_index_right, float *baseline,float *dmap_base,float *fx,
    float *img_index_left_sub_px,float *img_index_right_sub_px, short *belief_map_l, short *belief_map_r, float *roughly_projector_area_in_image, float *depth_cutoff)
{
    float depth_cutoff_near = depth_cutoff[0], depth_cutoff_far = depth_cutoff[1];
    int width = width_array[0];
    float projector_area_ratio = roughly_projector_area_in_image[0];
    float index_thres_for_matching = 1.5 * 1280.0 / (width*projector_area_ratio);  //the smaller projector_area in image, the larger index_offset cloud be
    int right_corres_point_offset_range = (1.333 * projector_area_ratio * width) / 128;
    // if another pixel has similar index( < index_thres_for_outliers_checking) has a distance > max_allow_pixel_per_index, consider it's an outlier 
    float max_allow_pixel_per_index_for_outliers_checking = 2.5 + 1.0 * projector_area_ratio * width / 1280.0;
    float index_thres_for_outliers_checking = index_thres_for_matching * 1.2;

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
        int most_corres_pts_l = -1, most_corres_pts_r = -1, most_corres_pts_l_bf = -1, most_corres_pts_r_bf = -1;
        int checking_left_edge = 0, checking_right_edge = width;
        int cnt_l = 0, cnt_r = 0;
        float average_corres_position_in_thres_l = 0, average_corres_position_in_thres_r = 0;
        if (last_right_corres_point > 0) {  // fast checking around last_right_corres_point
            checking_left_edge = last_right_corres_point - right_corres_point_offset_range + 1;
            checking_right_edge = last_right_corres_point + right_corres_point_offset_range + stride;
            if (checking_left_edge <=0) checking_left_edge=0;
            if (checking_right_edge >=width) checking_right_edge=width;
            for (int i=checking_left_edge; i < checking_right_edge; i++) {
                if (isnan(line_r[i])) continue;
                float thres = index_thres_for_matching + abs(img_index_left_sub_px[line_start_addr_offset+w] - w - img_index_right_sub_px[line_start_addr_offset+i] + i)/projector_area_ratio;
                pix_index_matching(line_l, line_r, w, curr_pix_idx, i, line_start_addr_offset, thres, belief_map_l, belief_map_r, &most_corres_pts_l, &most_corres_pts_r, &most_corres_pts_l_bf, &most_corres_pts_r_bf, &cnt_l, &cnt_r, &average_corres_position_in_thres_l, &average_corres_position_in_thres_r);
            }
        }
        if (most_corres_pts_l == -1 & most_corres_pts_r == -1) {
            // last_right_corres_point is invalid or not found most_corres_pts, expand the searching range and try searching again
            for (int i=0; i < width; i++) { 
                if (isnan(line_r[i])) continue;
                float thres = index_thres_for_matching;
                pix_index_matching(line_l, line_r, w, curr_pix_idx, i, line_start_addr_offset, thres, belief_map_l, belief_map_r, &most_corres_pts_l, &most_corres_pts_r, &most_corres_pts_l_bf, &most_corres_pts_r_bf, &cnt_l, &cnt_r, &average_corres_position_in_thres_l, &average_corres_position_in_thres_r);
            }
        }
        // refine index of right 'w_r' by matching results
        float w_r = 0;
        bool outliers_flag = false;
        #ifdef use_belief_map_for_checking
        if (most_corres_pts_l_bf != -1) most_corres_pts_l = most_corres_pts_l_bf;
        if (most_corres_pts_r_bf != -1) most_corres_pts_r = most_corres_pts_r_bf;
        #endif
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
        int w_r_int = (int)(w_r+0.499999);
        if (cnt_l != 0) average_corres_position_in_thres_l = average_corres_position_in_thres_l / cnt_l;
        if (cnt_r != 0) average_corres_position_in_thres_r = average_corres_position_in_thres_r / cnt_r;
        // check possiblely outliers
        #ifdef use_belief_map_for_checking
        int bfmap_thres_for_outliers = 64 + (belief_map_l[curr_pix_idx] + belief_map_r[line_start_addr_offset+w_r_int]) / 2;
        bool belief_map_pair_mismatch = (abs(belief_map_l[curr_pix_idx] - belief_map_r[line_start_addr_offset+w_r_int]) > bfmap_thres_for_outliers) &  (most_corres_pts_r_bf==-1 | most_corres_pts_l_bf==-1);
        bool checkright = belief_map_pair_mismatch | (belief_map_r[line_start_addr_offset+w_r_int]==0);
        bool checkleft = belief_map_pair_mismatch | (belief_map_l[curr_pix_idx]==0);
        #ifdef strong_outliers_checking
        if (belief_map_pair_mismatch & (belief_map_r[line_start_addr_offset+w_r_int]==0) ) outliers_flag=true;
        if (belief_map_pair_mismatch & (belief_map_l[curr_pix_idx]==0)) outliers_flag=true;
        #endif
        #else
        bool checkright = (belief_map_r[line_start_addr_offset+w_r_int]==0);
        bool checkleft = (belief_map_l[curr_pix_idx]==0);
        #endif
        if (checkright) {
            if (most_corres_pts_l != -1 & abs((float)(most_corres_pts_l-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
            if (most_corres_pts_r != -1 & abs((float)(most_corres_pts_r-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
            if (average_corres_position_in_thres_l != 0 & abs((float)(average_corres_position_in_thres_l-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
            if (average_corres_position_in_thres_r != 0 & abs((float)(average_corres_position_in_thres_r-w_r)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
        }
        if (outliers_flag==true) continue;
        last_right_corres_point = w_r_int;
        // left index
        float w_l = img_index_left_sub_px[curr_pix_idx];
        // check possiblely left outliers
        if (checkleft) {
            for (int i=0; i < width; i++) {
                if ((line_l[w]-index_thres_for_outliers_checking <= line_l[i]) & (line_l[i] <= line_l[w]+index_thres_for_outliers_checking)) {
                    if (abs((float)(w_l-i)) > max_allow_pixel_per_index_for_outliers_checking) outliers_flag = true;
                }
            }
        }
        if (outliers_flag==true) continue;
        // get stereo diff and depth
        float stereo_diff = dmap_base[0] + w_l - w_r;
        if (dmap_base[0] < 0) stereo_diff = - stereo_diff;
        if (stereo_diff > 0.000001) {
            float depth = fx[0] * baseline[0] / stereo_diff;
            if ((depth_cutoff_near < depth) & (depth < depth_cutoff_far)) depth_map[curr_pix_idx] = depth;
        }
    }
}

#define drop_possible_outliers_edges_during_sub_pix true
__global__ void optimize_dmap_using_sub_pixel_map(float *depth_map, float *optimized_depth_map, int *height_array, int *width_array, float *img_index_left_sub_px)
{
    // interpo for depth map using sub-pixel map
    // this does not improve a lot on rendered datasets because no distortion and less stereo rectify for left camera, but very useful for real captures
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
    if (depth_map[current_pix_idx] == 0.0) {
        if (depth_map[current_pix_idx-1] != 0.0 & depth_map[current_pix_idx+1] != 0.0 & abs(depth_map[current_pix_idx-1] - depth_map[current_pix_idx+1]) < subpix_optimize_unconsis_thres) {
            right_pos = img_index_left_sub_px[current_pix_idx+1];
            right_value = depth_map[current_pix_idx+1];
            left_pos = img_index_left_sub_px[current_pix_idx-1];
            left_value = depth_map[current_pix_idx-1];
        } else {
            optimized_depth_map[current_pix_idx] = 0;
            return;
        }
    }
    else if (real_pos_for_current_depth >= w) {
        right_pos = real_pos_for_current_depth;
        right_value = depth_map[current_pix_idx];
        if (depth_map[current_pix_idx-1] == 0) {
            #ifdef drop_possible_outliers_edges_during_sub_pix
            if (depth_map[current_pix_idx-2] == 0) optimized_depth_map[current_pix_idx] = 0;
            else optimized_depth_map[current_pix_idx] = right_value;
            #else
            optimized_depth_map[current_pix_idx] = right_value;
            #endif
            return;
        }
        else if (abs(right_value - depth_map[current_pix_idx-1]) < subpix_optimize_unconsis_thres) {
            left_pos = img_index_left_sub_px[current_pix_idx-1];
            left_value = depth_map[current_pix_idx-1];
        } 
        else {
            optimized_depth_map[current_pix_idx] = right_value;
            return;
        }
    }
    else {
        left_pos = real_pos_for_current_depth;
        left_value = depth_map[current_pix_idx];
        if (depth_map[current_pix_idx+1] == 0) {
            #ifdef drop_possible_outliers_edges_during_sub_pix
            if (depth_map[current_pix_idx+2] == 0) optimized_depth_map[current_pix_idx] = 0;
            else optimized_depth_map[current_pix_idx] = left_value;
            #else
            optimized_depth_map[current_pix_idx] = left_value;
            #endif
        } else if (abs(left_value - depth_map[current_pix_idx+1]) < subpix_optimize_unconsis_thres) {
            right_pos = img_index_left_sub_px[current_pix_idx+1];
            right_value = depth_map[current_pix_idx+1];
        } else{
            optimized_depth_map[current_pix_idx] = left_value;
            return;
        }
    }
    optimized_depth_map[current_pix_idx] = left_value + (right_value-left_value) * (w-left_pos)/(right_pos-left_pos);
}

// flying points filter
// a point could be considered as not flying when: points in checking range below max_distance > minmum num
__global__ void flying_points_filter(float *depth_map, float *depth_map_raw, int *height_array, int *width_array, float *camera_kp, float *depth_filter_max_distance, int *depth_filter_minmum_points_in_checking_range, short *belief_map)
{
    #define use_fast_distance_checking_for_flying_points_filter // use 3D distance (slower but more precisely) or only distance of axis-z to check flying points
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
        if (belief_map[current_pix_idx] >= 1) max_distance = depth_filter_max_distance[0];
        else max_distance = depth_filter_max_distance[0] / 2;
        
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

// truncated smoothing filter for depth map
__global__ void depth_smoothing_filter_w(float *depth_map_out, float *depth_map, int *height_array, int *width_array, int *depth_filter_max_length, float *depth_filter_unvalid_thres, short *belief_map)
{
    int width = width_array[0];
    int filter_max_length = depth_filter_max_length[0];
    float filter_thres = depth_filter_unvalid_thres[0];
    const float filter_weights[5] = {1.0, 0.667, 0.4, 0.2, 0.1};

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = blockIdx.x / 4;
    int w = current_pix_idx % width;
    float curr_pix_value = depth_map[current_pix_idx];
    if (curr_pix_value != 0) {
        int line_start_addr_offset = h * width;
        float left_weight = 0.0, right_weight = 0.0, depth_sum = curr_pix_value*filter_weights[0];
        for (int i=1; i< filter_max_length+1; i++) {
            int l_idx = w-i, r_idx = w+i;
            if (!(l_idx > 0 & r_idx < width)) break;
            // if (belief_map[current_pix_idx] >= 1) filter_thres = depth_filter_unvalid_thres[0];
            // else filter_thres = depth_filter_unvalid_thres[0] / 4;
            if (depth_map[line_start_addr_offset+l_idx] != 0 & abs(depth_map[line_start_addr_offset+l_idx] - curr_pix_value) < filter_thres & \
                depth_map[line_start_addr_offset+r_idx] != 0 & abs(depth_map[line_start_addr_offset+r_idx] - curr_pix_value) < filter_thres) {
                left_weight += filter_weights[i];
                right_weight += filter_weights[i];
                depth_sum += (depth_map[line_start_addr_offset+r_idx] + depth_map[line_start_addr_offset+l_idx]) * filter_weights[i];
            }
        }
        depth_map_out[current_pix_idx] = depth_sum / (filter_weights[0] + left_weight + right_weight);
    }
    else {
        depth_map_out[current_pix_idx] = depth_map[current_pix_idx];
    }
}

__global__ void depth_smoothing_filter_h(float *depth_map_out, float *depth_map, int *height_array, int *width_array, int *depth_filter_max_length, float *depth_filter_unvalid_thres, short *belief_map)
{
    int height = height_array[0];
    int width = width_array[0];
    int filter_max_length = depth_filter_max_length[0];
    float filter_thres = depth_filter_unvalid_thres[0];
    const float filter_weights[5] = {1.0, 0.667, 0.4, 0.2, 0.1};

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = blockIdx.x / 4;
    int w = current_pix_idx % width;
    float curr_pix_value = depth_map[current_pix_idx];
    if (curr_pix_value != 0) {
        float left_weight = 0.0, right_weight = 0.0, depth_sum = curr_pix_value*filter_weights[0];
        for (int i=1; i< filter_max_length+1; i++) {
            int l_idx = h-i, r_idx = h+i;
            if (!(l_idx > 0 & r_idx < height)) break;
            if (depth_map[l_idx*width+w] != 0 & abs(depth_map[l_idx*width+w] - curr_pix_value) < filter_thres & \
                depth_map[r_idx*width+w] != 0 & abs(depth_map[r_idx*width+w] - curr_pix_value) < filter_thres) {
                left_weight += filter_weights[i];
                right_weight += filter_weights[i];
                depth_sum += (depth_map[r_idx*width+w] + depth_map[l_idx*width+w]) * filter_weights[i];
            }
        }
        depth_map_out[current_pix_idx] = depth_sum / (filter_weights[0] + left_weight + right_weight);
    }
    else {
        depth_map_out[current_pix_idx] = depth_map[current_pix_idx];
    }
}

__device__ __forceinline__ float get_mid_val(float a, float b, float c)
{
    float max=a, min=a;
    if (b > max) max = b;
    if (c > max) max = c;
    if (b < min) min = b;
    if (c < min) min = c;
    return a+b+c-min-max;
}

__global__ void depth_median_filter_w(float *depth_map_out, float *depth_map, int *height_array, int *width_array, int *depth_filter_max_length)
{   // filter_max_length (ksize) fixed to 1 for now
    int height = height_array[0];
    int width = width_array[0];
    // int filter_max_length = 1; //depth_filter_max_length[0]; // 1, 2

    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int h = blockIdx.x / 4;
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
    int h = blockIdx.x / 4;
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

// one iteration of total variational filter
// iter_depth_map_out should be copied from depth_map before calling this function
__global__ void total_variational_filter_one_iter(float *iter_depth_map_out, float *original_depth_map, int *height_array, int *width_array, float *lambda_array)
{
    // dt: dt   - time step [0.2]
    // epsilon: epsilon (of gradient regularization) [1]
    // lambda: lam  - fidelity term lambda [0]
    float dt=0.2, epsilon=1.0, lambda=lambda_array[0]; //To tune
    float ep2 = epsilon * epsilon;
    
    int width = width_array[0], height = height_array[0];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int i = idx / width, j = idx % width; // h, w
    
    int iUp = i - 1, iDown = i + 1;
    int jLeft = j - 1, jRight = j + 1;
    // bounds
    if (0 == i) iUp = i; if (height - 1 == i) iDown = i;
    if (0 == j) jLeft = j; if (width - 1 == j) jRight = j;

    if (original_depth_map[idx] == 0.0 | 
        original_depth_map[i*width+jRight] == 0.0 | original_depth_map[i*width+jLeft] == 0.0   |
        original_depth_map[iDown*width+j] == 0.0  | original_depth_map[iUp*width+j]==0.0       | 
        original_depth_map[iDown*width+jRight]==0.0 | original_depth_map[iUp*width+jLeft] == 0.0 | 
        original_depth_map[iUp*width+jRight] == 0.0 | original_depth_map[iDown*width+jLeft] == 0.0)
        return;
    
    float tmp_x = (iter_depth_map_out[i*width+jRight] - iter_depth_map_out[i*width+jLeft]) / 2.0;
    float tmp_y = (iter_depth_map_out[iDown*width+j] - iter_depth_map_out[iUp*width+j]) / 2.0;
    float tmp_xx = iter_depth_map_out[i*width+jRight] + iter_depth_map_out[i*width+jLeft] - 2 * iter_depth_map_out[idx];
    float tmp_yy = iter_depth_map_out[iDown*width+j] + iter_depth_map_out[iUp*width+j] - 2 * iter_depth_map_out[idx];
    float tmp_xy = (iter_depth_map_out[iDown*width+jRight] + iter_depth_map_out[iUp*width+jLeft] - iter_depth_map_out[iUp*width+jRight] - iter_depth_map_out[iDown*width+jLeft]) / 4.0;
    float tmp_num = tmp_yy * (tmp_x * tmp_x + ep2) + tmp_xx * (tmp_y * tmp_y + ep2) - 2 * tmp_x * tmp_y * tmp_xy;
    float tmp_den = powf(tmp_x * tmp_x + tmp_y * tmp_y + ep2, 1.5);
    float diff = original_depth_map[idx] - iter_depth_map_out[idx];
    if (abs(diff) > subpix_optimize_unconsis_thres) return;
    iter_depth_map_out[idx] += dt*(tmp_num / tmp_den + lambda*(diff));
}

// Anisotropic Filter, or P-M Filter, 各向异性扩散滤波 
#define anisotropic_filter_fast_impl_without_exp
__global__ void anisotropic_filter_one_iter(float *iter_depth_map_out, float *iter_depth_map_in, int *height_array, int *width_array, float *kappa_input)
{
    float dt=0.25, kappa=kappa_input[0];
    const float unvalid_thres = 0.005; //need tuning

    int width = width_array[0], height = height_array[0];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (iter_depth_map_in[idx] == 0.0) return;
    int i = idx / width, j = idx % width; // h, w
    int iUp = i - 1, iDown = i + 1;
    int jLeft = j - 1, jRight = j + 1;
    // bounds
    if (0 == i) iUp = i; if (height - 1 == i) iDown = i;
    if (0 == j) jLeft = j; if (width - 1 == j) jRight = j;

    float deltaN = iter_depth_map_in[iUp*width+j] - iter_depth_map_in[i*width+j];
    float deltaS = iter_depth_map_in[iDown*width+j] - iter_depth_map_in[i*width+j];
    float deltaE = iter_depth_map_in[i*width+jRight] - iter_depth_map_in[i*width+j];
    float deltaW = iter_depth_map_in[i*width+jLeft] - iter_depth_map_in[i*width+j];
    if (deltaN >= unvalid_thres | deltaS >= unvalid_thres | deltaE >= unvalid_thres | deltaW >= unvalid_thres) return;

    #ifndef anisotropic_filter_fast_impl_without_exp
    float cN = expf(-(deltaN / kappa) * (deltaN / kappa));
    float cS = expf(-(deltaS / kappa) * (deltaS / kappa));
    float cE = expf(-(deltaE / kappa) * (deltaE / kappa));
    float cW = expf(-(deltaW / kappa) * (deltaW / kappa));
    #else
    float cN = 1.0 / (1 + (deltaN / kappa) * (deltaN / kappa));
    float cS = 1.0 / (1 + (deltaS / kappa) * (deltaS / kappa));
    float cE = 1.0 / (1 + (deltaE / kappa) * (deltaE / kappa));
    float cW = 1.0 / (1 + (deltaW / kappa) * (deltaW / kappa));
    #endif
    iter_depth_map_out[idx] += dt * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
}

__global__ void convert_dmap_to_mili_meter(float *depth_map)
{
    int current_pix_idx = threadIdx.x + blockIdx.x*blockDim.x;
    depth_map[current_pix_idx] = 1000.0*depth_map[current_pix_idx];
}
