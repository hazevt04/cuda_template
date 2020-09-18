#include <memory>
#include <exception>

#include <cuda_runtime.h>
#include "my_cuda_utils.hpp"

#include "add_kernel.cuh"

#include "managed_allocator_global.hpp"
#include "managed_allocator_host.hpp"

template<typename T>
class AddGPU {
private:
   managed_vector_global<T> lvals;
   managed_vector_global<T> rvals;
   managed_vector_host<T> sums;
   int num_vals;
   bool debug;

   std::unique_ptr<cudaStream_t> stream_ptr;
   void gen_kernel_data( int seed );

public:
   AddGPU():
      num_vals(0),
      debug(false) {}
   
   AddGPU( int new_num_vals ):
      num_vals( new_num_vals ),
      debug( new_debug ) {
      
      try {
         cudaError_t cerror = cudaSuccess;
         debug_cout( debug, __func__, "(): num_vals is ", num_vals, "\n" );
         lvals.resize( num_vals );
         rvals.resize( num_vals );
         sums.resize( num_vals );
         
         debug_cout( debug, __func__, "(): after resizing vectors\n" );

         try_cuda_func_throw( cerror, cudaStreamCreate( stream_ptr.get() ) );

         debug_cout( debug, __func__,  "(): after cudaStreamCreate()\n" ); 

      } catch( std::exception& ex ) {
         debug_cout( debug, __func__, "(): ERROR: ", ex.what(), "\n" ); 
      }
   }

   ~AddGPU() {
      lvals.clear();
      rvals.clear();
      sums.clear();    
      cudaStreamDestroy( *(stream_ptr.get()) );
   }

   managed_vector_host<T> run<T>();

};


