#include <cuda_runtime.h>
#include "my_cuda_utils.hpp"

#include "add_kernel.cuh"

#include "AddGPU.cuh"
#include "managed_allocator_host.hpp"

template<typename T>
managed_vector_host<T> AddGPU<T>::run() {
   cudaError_t cerror = cudaSuccess;
   int num_shared_bytes = 0;
   int threads_per_block = 256;
   int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;

   debug_cout( debug, __func__, "(): num_vals is ", num_vals, "\n" ); 
   debug_cout( debug, __func__, "(): threads_per_block is ", threads_per_block, "\n" ); 
   debug_cout( debug, __func__, "(): num_blocks is ", num_blocks, "\n" ); 

   add_kernel<T><<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
      sums.data(), lvals.data(), rvals.data(), num_vals 
   );
   
   try_cuda_func_throw( cerror, cudaStreamSynchronize( *(stream_ptr.get())  ) );

   debug_cout( debug, __func__, "(): sums.size() is ", sums.size(), "\n" ); 
   return sums;
}

template<typename T>
void AddGPU<T>::gen_kernel_data( int seed ) {
   for ( auto& lval: lvals ) {
      lval = (T)2;
   } 
   for ( auto& rval: rvals ) {
      rval = (T)2;
   }
}

template class AddGPU<int>;
template class AddGPU<float>;
