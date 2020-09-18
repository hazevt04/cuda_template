#include <cuda_runtime.h>

#include "my_utils.hpp"
#include "my_cuda_utils.hpp"

#include "add_kernel.cuh"

#include "AddGPU.cuh"
#include "managed_allocator_host.hpp"


template<typename T>
managed_vector_global<T> AddGPU<T>::run() {
   cudaError_t cerror = cudaSuccess;
   int num_shared_bytes = 0;
   int threads_per_block = 64;
   int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;

   debug_cout( debug, __func__, "(): num_vals is ", num_vals, "\n" ); 
   debug_cout( debug, __func__, "(): threads_per_block is ", threads_per_block, "\n" ); 
   debug_cout( debug, __func__, "(): num_blocks is ", num_blocks, "\n" ); 

   gen_data();
   
   cudaStreamAttachMemAsync( *(stream_ptr.get()), lvals.data(), 0, cudaMemAttachGlobal );
   cudaStreamAttachMemAsync( *(stream_ptr.get()), rvals.data(), 0, cudaMemAttachGlobal );
   
   add_kernel<T><<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
      sums.data(), lvals.data(), rvals.data(), num_vals 
   );

   
   // Prefetch fspecs from the GPU
   cudaStreamAttachMemAsync( *(stream_ptr.get()), sums.data(), 0, cudaMemAttachHost );   
   
   try_cuda_func_throw( cerror, cudaStreamSynchronize( *(stream_ptr.get())  ) );
   
   // sums.size() is 0 because the add_kernel modified the data and not a std::vector function
   sums.resize(num_vals);

   debug_cout( debug, __func__, "(): sums.size() is ", sums.size(), "\n" ); 

   if (debug) {
      print_sums();
   }
   return sums;
}

template class AddGPU<int>;
template class AddGPU<float>;

