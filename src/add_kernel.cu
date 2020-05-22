#include "add_kernel.cuh"

//////////////////////////////////////
// THE Kernel (add)
// Kernel function to add the elements 
// of two arrays x_vals and y_vals. 
//////////////////////////////////////
__global__ void add(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
    float4* const __restrict__ d_y_vals, const int num_items ) {

  // Assuming one stream
  int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
  // stride is set to the total number of threads in the grid
  int stride = blockDim.x * gridDim.x;
  for (int index = global_index; index < num_items; index+=stride) {
    float4 t_x_val = d_x_vals[index];
    float4 t_y_val = d_y_vals[index];

    float4 t_result = {0.0,0.0,0.0,0.0};

    t_result.x = t_x_val.x + t_y_val.x;
    t_result.y = t_x_val.y + t_y_val.y;
    t_result.z = t_x_val.z + t_y_val.z;
    t_result.w = t_x_val.w + t_y_val.w;

    d_results[index] = t_result;
  }
}

