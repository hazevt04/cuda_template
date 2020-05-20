#include "add_kernels.cuh"

//////////////////////////////////////
// THE Kernel (add)
// Kernel function to add the elements 
// of two arrays x_vals and y_vals. 
//////////////////////////////////////
__global__ void add_rolled(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
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


__global__ void add_unrolled_four(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
    float4* const __restrict__ d_y_vals, const int num_items ) {

  // Assuming one stream
  int global_index = 4*(blockIdx.x * blockDim.x) + 4*(threadIdx.x);
  // stride is set to the total number of threads in the grid
  int stride = blockDim.x * gridDim.x;
  // Unrolled the loop by 4
  for (int index = global_index; index < num_items; index+=stride) {
    float4 t_x_val1 = d_x_vals[index];
    float4 t_x_val2 = d_x_vals[index+1];
    float4 t_x_val3 = d_x_vals[index+2];
    float4 t_x_val4 = d_x_vals[index+3];
    
    float4 t_y_val1 = d_y_vals[index];
    float4 t_y_val2 = d_y_vals[index+1];
    float4 t_y_val3 = d_y_vals[index+2];
    float4 t_y_val4 = d_y_vals[index+3];

    float4 t_result1 = {0.0,0.0,0.0,0.0};
    float4 t_result2 = {0.0,0.0,0.0,0.0};
    float4 t_result3 = {0.0,0.0,0.0,0.0};
    float4 t_result4 = {0.0,0.0,0.0,0.0};

    t_result1.x = t_x_val1.x + t_y_val1.x;
    t_result1.y = t_x_val1.y + t_y_val1.y;
    t_result1.z = t_x_val1.z + t_y_val1.z;
    t_result1.w = t_x_val1.w + t_y_val1.w;

    t_result2.x = t_x_val2.x + t_y_val2.x;
    t_result2.y = t_x_val2.y + t_y_val2.y;
    t_result2.z = t_x_val2.z + t_y_val2.z;
    t_result2.w = t_x_val2.w + t_y_val2.w;

    t_result3.x = t_x_val3.x + t_y_val3.x;
    t_result3.y = t_x_val3.y + t_y_val3.y;
    t_result3.z = t_x_val3.z + t_y_val3.z;
    t_result3.w = t_x_val3.w + t_y_val3.w;

    t_result4.x = t_x_val4.x + t_y_val4.x;
    t_result4.y = t_x_val4.y + t_y_val4.y;
    t_result4.z = t_x_val4.z + t_y_val4.z;
    t_result4.w = t_x_val4.w + t_y_val4.w;

    d_results[index] = t_result1;
    d_results[index+1] = t_result2;
    d_results[index+2] = t_result3;
    d_results[index+3] = t_result4;
  } // end of for
 
} // end of add_unrolled_four

__global__ void add_unrolled_eight(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
    float4* const __restrict__ d_y_vals, const int num_items ) {

  // Assuming one stream
  int global_index = 8*(blockIdx.x * blockDim.x) + 8*(threadIdx.x);
  // stride is set to the total number of threads in the grid
  int stride = blockDim.x * gridDim.x;
  // Unrolled the loop by 8
  for (int index = global_index; index < num_items; index+=stride) {
    float4 t_x_val1 = d_x_vals[index];
    float4 t_x_val2 = d_x_vals[index+1];
    float4 t_x_val3 = d_x_vals[index+2];
    float4 t_x_val4 = d_x_vals[index+3];
    float4 t_x_val5 = d_x_vals[index+4];
    float4 t_x_val6 = d_x_vals[index+5];
    float4 t_x_val7 = d_x_vals[index+6];
    float4 t_x_val8 = d_x_vals[index+7];
    
    float4 t_y_val1 = d_y_vals[index];
    float4 t_y_val2 = d_y_vals[index+1];
    float4 t_y_val3 = d_y_vals[index+2];
    float4 t_y_val4 = d_y_vals[index+3];
    float4 t_y_val5 = d_y_vals[index+4];
    float4 t_y_val6 = d_y_vals[index+5];
    float4 t_y_val7 = d_y_vals[index+6];
    float4 t_y_val8 = d_y_vals[index+7];

    float4 t_result1 = {0.0,0.0,0.0,0.0};
    float4 t_result2 = {0.0,0.0,0.0,0.0};
    float4 t_result3 = {0.0,0.0,0.0,0.0};
    float4 t_result4 = {0.0,0.0,0.0,0.0};
    float4 t_result5 = {0.0,0.0,0.0,0.0};
    float4 t_result6 = {0.0,0.0,0.0,0.0};
    float4 t_result7 = {0.0,0.0,0.0,0.0};
    float4 t_result8 = {0.0,0.0,0.0,0.0};

    t_result1.x = t_x_val1.x + t_y_val1.x;
    t_result1.y = t_x_val1.y + t_y_val1.y;
    t_result1.z = t_x_val1.z + t_y_val1.z;
    t_result1.w = t_x_val1.w + t_y_val1.w;

    t_result2.x = t_x_val2.x + t_y_val2.x;
    t_result2.y = t_x_val2.y + t_y_val2.y;
    t_result2.z = t_x_val2.z + t_y_val2.z;
    t_result2.w = t_x_val2.w + t_y_val2.w;

    t_result3.x = t_x_val3.x + t_y_val3.x;
    t_result3.y = t_x_val3.y + t_y_val3.y;
    t_result3.z = t_x_val3.z + t_y_val3.z;
    t_result3.w = t_x_val3.w + t_y_val3.w;

    t_result4.x = t_x_val4.x + t_y_val4.x;
    t_result4.y = t_x_val4.y + t_y_val4.y;
    t_result4.z = t_x_val4.z + t_y_val4.z;
    t_result4.w = t_x_val4.w + t_y_val4.w;

    t_result5.x = t_x_val5.x + t_y_val5.x;
    t_result5.y = t_x_val5.y + t_y_val5.y;
    t_result5.z = t_x_val5.z + t_y_val5.z;
    t_result5.w = t_x_val5.w + t_y_val5.w;

    t_result6.x = t_x_val6.x + t_y_val6.x;
    t_result6.y = t_x_val6.y + t_y_val6.y;
    t_result6.z = t_x_val6.z + t_y_val6.z;
    t_result6.w = t_x_val6.w + t_y_val6.w;

    t_result7.x = t_x_val7.x + t_y_val7.x;
    t_result7.y = t_x_val7.y + t_y_val7.y;
    t_result7.z = t_x_val7.z + t_y_val7.z;
    t_result7.w = t_x_val7.w + t_y_val7.w;

    t_result8.x = t_x_val8.x + t_y_val8.x;
    t_result8.y = t_x_val8.y + t_y_val8.y;
    t_result8.z = t_x_val8.z + t_y_val8.z;
    t_result8.w = t_x_val8.w + t_y_val8.w;

    d_results[index] = t_result1;
    d_results[index+1] = t_result2;
    d_results[index+2] = t_result3;
    d_results[index+3] = t_result4;
    d_results[index+4] = t_result5;
    d_results[index+5] = t_result6;
    d_results[index+6] = t_result7;
    d_results[index+7] = t_result8;
  } // end of for loop

} // end of add_unrolled_eight
