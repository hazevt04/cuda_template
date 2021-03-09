#include "pinned_mapped_vector_utils.hpp"
#include "pinned_mapped_allocator.hpp"

#include "my_cuda_utils.hpp"
#include "my_printers.hpp"
#include "my_utils.hpp"

#define RAND_MAX_RECIP (1.0f/(float)RAND_MAX)

//////////////////////////////////////
// THE Kernel (add)
// Kernel function to add the elements 
// of two arrays x_vals and y_vals. 
//////////////////////////////////////
__global__ void add(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
   float4* const __restrict__ d_y_vals, const int num_vals ) {

   // Assuming one stream
   int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for (int index = global_index; index < num_vals; index+=stride) {
      float4 t_x_val = {
         d_x_vals[index].x,
         d_x_vals[index].y,
         d_x_vals[index].z,
         d_x_vals[index].w
      };
      float4 t_y_val = {
         d_y_vals[index].x,
         d_y_vals[index].y,
         d_y_vals[index].z,
         d_y_vals[index].w
      };

      float4 t_result = {0.0,0.0,0.0,0.0};

      t_result.x = t_x_val.x + t_y_val.x;
      t_result.y = t_x_val.y + t_y_val.y;
      t_result.z = t_x_val.z + t_y_val.z;
      t_result.w = t_x_val.w + t_y_val.w;

      d_results[index] = {
         t_result.x,
         t_result.x,
         t_result.x,
         t_result.x,
      };
   }
}



int main(int argc, char **argv) {
   try {
      cudaError_t cerror = cudaSuccess;
      bool debug = true;
      
      // Empirically-determined maximum number
      int num_vals = 512;//1<<21;
      size_t num_bytes = num_vals * sizeof(float4);

      ////////////////////////////////////////////////////////////////////
      // ALLOCATE KERNEL DATA
      ////////////////////////////////////////////////////////////////////
      std::cout << "Trying to initialize memory for input and output data...\n";
      // Allocate pinned host memory that is also accessible by the device.
      pinned_mapped_vector<float4> x_vals;
      pinned_mapped_vector<float4> y_vals;
      pinned_mapped_vector<float4> results;
      std::vector<float4> exp_results;
      
      x_vals.reserve( num_vals );
      y_vals.reserve( num_vals );
      results.reserve( num_vals );
      exp_results.reserve( num_vals );
      exp_results.resize( num_vals );

      ////////////////////////////////////////////////////////////////////
      // GENERATE KERNEL DATA
      ////////////////////////////////////////////////////////////////////
      std::cout << "Trying to generate input data...\n";
      
      // initialize x_vals and y arrays on the host
      int seed  = 5;
      float4 lower = { 1.0f, 1.0f, 1.0f, 1.0f };
      float4 upper = { 50.0f, 50.0f, 50.0f, 50.0f };
      gen_float4s( x_vals, num_vals, lower, upper, seed );
      gen_float4s( y_vals, num_vals, lower, upper, seed );

      ////////////////////////////////////////////////////////////////////
      // GENERATE EXPECTED DATA
      ////////////////////////////////////////////////////////////////////
      std::cout << "Trying to generate expected outputs...\n";
      Time_Point start = Steady_Clock::now();

      // Generate expected results
      std::transform( x_vals.begin(), x_vals.end(), y_vals.begin(), exp_results.begin(), 
            []( const float4& x, const float4& y ) { return x + y; } );

      print_vals<float4>( x_vals.data(), 10, "Other X Vals:\n", "", "\n", "\n" );
      print_vals<float4>( y_vals.data(), 10, "Other Y Vals:\n", "", "\n", "\n" );

      std::cout << __func__ << "(): Expected Results size is " << exp_results.size() << "\n"; 

      print_vals<float4>( exp_results.data(), 10, "Expected Results:\n", "", "\n", "\n" );
      Duration_ms duration_ms = Steady_Clock::now() - start;

      std::cout << __func__ << "(): CPU: It took " << duration_ms.count() 
         << " milliseconds to add " << num_vals << " values\n"; 
      
      float seconds = duration_ms.count() * std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
      float actual_num_vals = num_vals * 4.0;
      
      std::cout << __func__ << "(): CPU: That's a rate of " << (actual_num_vals/seconds) 
         << " float additions per second\n"; 

      //////////////////////////////////////////////////////////////////////
      //// RUN KERNEL
      ///////////////////////////////////////////////////////////////////////
      cudaEvent_t start_event;
      cudaEvent_t stop_event;
      std::string kernel_name = "Add";

      int threads_per_block = 256;
      int num_blocks = (num_vals + threads_per_block - 1) / threads_per_block;
    
      std::cout << "Trying to run CUDA kernel\n";

      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n"; 

      try_cuda_func( cerror, cudaEventCreate(&start_event) );
      try_cuda_func( cerror, cudaEventCreate(&stop_event) );

      try_cuda_func( cerror, cudaEventRecord(start_event) );

      add<<<num_blocks, threads_per_block, 0>>>( results.data(), x_vals.data(), y_vals.data(), num_vals );

      try_cuda_func( cerror, cudaEventRecord(stop_event) );
      try_cuda_func( cerror, cudaEventSynchronize(stop_event) );

      float add_milliseconds = 0;
      try_cuda_func( cerror, cudaEventElapsedTime(&add_milliseconds, start_event, stop_event) );

      std::cout << "GPU: Time for asynchronous transfer and " << kernel_name.c_str() 
         << " kernel execution: " << add_milliseconds << " milliseconds for " 
         << num_vals << " items\n";

      float add_seconds = add_milliseconds * std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
      actual_num_vals = num_vals * 4.0;

      std::cout << "GPU: That's a rate of " << (actual_num_vals/add_seconds) << " float additions per second\n"; 

      try_cuda_func( cerror, cudaEventDestroy(start_event) );
      try_cuda_func( cerror, cudaEventDestroy(stop_event) );

      //check_kernel( results, exp_results, num_vals );
      //float4 max_error = {2.0, 2.0, 2.0, 2.0};
      //std::cout << "Checking kernel outputs against the expected outputs...\n";
      //for ( int index = 0; index < num_vals; ++index ) {
      //   if ( fabs( exp_results[index] - results[index] ) > max_error ) {
      //      std::cout << "ERROR: Mismatch for Result " << index << ": " 
      //         << results[index] << " when the expected result was "
      //         << exp_results[index] << "\n";
      //   }
      //}
      
      std::cout << "Freeing memory used for input and output data...\n";

      //try_cuda_free_host( cerror, x_vals );
      //try_cuda_free_host( cerror, y_vals );
      //try_cuda_free_host( cerror, results );
      x_vals.clear();
      y_vals.clear();
      results.clear();
      exp_results.clear();

      //delete_array( exp_results );
      return SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << "ERROR: " << ex.what() << "\n"; 
      return FAILURE;
   }
}
