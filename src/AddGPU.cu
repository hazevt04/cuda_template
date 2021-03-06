#include <cuda_runtime.h>

#include "my_utils.hpp"
#include "my_cuda_utils.hpp"

#include "add_kernel.cuh"

#include "AddGPU.cuh"
#include "managed_allocator_host.hpp"


template<typename T>
void AddGPU<T>::run() {
   try {
      cudaError_t cerror = cudaSuccess;
      int num_shared_bytes = 0;
      int threads_per_block = 64;
      int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;

      debug_cout( debug, __func__, "(): num_vals is ", num_vals, "\n" ); 
      debug_cout( debug, __func__, "(): threads_per_block is ", threads_per_block, "\n" ); 
      debug_cout( debug, __func__, "(): num_blocks is ", num_blocks, "\n" ); 

      gen_data();
      
      debug_cout( debug, __func__, "(): lvals.size() is ", lvals.size(), "\n" ); 
      debug_cout( debug, __func__, "(): rvals.size() is ", rvals.size(), "\n" ); 
      
      if ( debug ) {
         print_vec<T>( lvals, num_vals, "Lvals: ", " " ); 
         print_vec<T>( rvals, num_vals, "Rvals: ", " " ); 
      }

      cudaStreamAttachMemAsync( *(stream_ptr.get()), lvals.data(), 0, cudaMemAttachGlobal );
      cudaStreamAttachMemAsync( *(stream_ptr.get()), rvals.data(), 0, cudaMemAttachGlobal );

      add_kernel<T><<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         sums.data(), lvals.data(), rvals.data(), num_vals 
      );

      // Prefetch fspecs from the GPU
      cudaStreamAttachMemAsync( *(stream_ptr.get()), sums.data(), 0, cudaMemAttachHost );   
      
      try_cuda_func_throw( cerror, cudaStreamSynchronize( *(stream_ptr.get())  ) );
      
      compare_vecs<T>( sums.data(), exp_sums.data(), num_vals, "Sums: ", "Expected Sums: ", debug );

      // sums.size() is 0 because the add_kernel modified the data and not a std::vector function
      debug_cout( debug, __func__, "(): sums.size() is ", sums.size(), "\n" ); 

      if ( debug ) print_sums( "Sums: " );

   } catch( std::exception& ex ) {
      throw;
   }
}

template class AddGPU<int>;
template class AddGPU<float>;

