#ifndef __ADD_KERNELS__
#define __ADD_KERNELS__

#include "cuda_utils.h"

//////////////////////////////////////
// THE Kernel (add)
// Kernel function to add the elements 
// of two arrays x_vals and y_vals. 
//////////////////////////////////////
__global__ void add_rolled(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
    float4* const __restrict__ d_y_vals, const int num_items );

__global__ void add_unrolled_four(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
    float4* const __restrict__ d_y_vals, const int num_items );

__global__ void add_unrolled_eight(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
    float4* const __restrict__ d_y_vals, const int num_items );


#endif
