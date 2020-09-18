#pragma once

#include "my_cuda_utils.hpp"

//////////////////////////////////////
// THE Kernel (add)
// Kernel function to add the elements 
// of two arrays lvals and rvals. 
//////////////////////////////////////
template<typename T>
__global__ void add_kernel( T* __restrict__ sums, T* const __restrict__ lvals, 
   T* const __restrict__ rvals, const int num_vals );


