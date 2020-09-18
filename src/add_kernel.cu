#include "add_kernel.cuh"

//////////////////////////////////////
// THE Kernel (add)
// Kernel function to add the elements 
// of two arrays x_vals and y_vals. 
//////////////////////////////////////
template<typename T>
__global__ void add_kernel( T* __restrict__ sums, T* const __restrict__ lvals, 
   T* const __restrict__ rvals, const int num_vals ) {

   // Assuming one stream
   int global_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for (int index = global_index; index < num_vals; index+=stride) {
      sums[index] = lvals[index] + rvals[index];
  }
}

template
__global__ void add_kernel<int>( int* __restrict__ sums, int* const __restrict__ lvals, 
   int* const __restrict__ rvals, const int num_vals );

template
__global__ void add_kernel<float>( float* __restrict__ sums, float* const __restrict__ lvals, 
   float* const __restrict__ rvals, const int num_vals );

