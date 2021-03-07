#include "my_cuda_utils.hpp"
#include "my_utils.hpp"

#define RAND_MAX_RECIP (1.0f/(float)RAND_MAX)

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
      int num_items = 1<<21;
      size_t num_bytes = num_items * sizeof(float4);

      float4* x_vals = nullptr;
      float4* y_vals = nullptr;
      float4* results = nullptr;
      float4* exp_vals = nullptr;
      float4* d_x_vals = nullptr;
      float4* d_y_vals = nullptr;
      float4* d_results = nullptr;

      ////////////////////////////////////////////////////////////////////
      // ALLOCATE KERNEL DATA
      ////////////////////////////////////////////////////////////////////
      std::cout << "Trying to initialize memory for input and output data...\n";
      // Allocate pinned host memory that is also accessible by the device.
      try_cuda_func( cerror, cudaHostAlloc( (void**)&x_vals, num_bytes, cudaHostAllocMapped ) ); 
      try_cuda_func( cerror, cudaHostAlloc( (void**)&y_vals, num_bytes, cudaHostAllocMapped ) ); 
      try_cuda_func( cerror, cudaHostAlloc( (void**)&results, num_bytes, cudaHostAllocMapped ) ); 

      try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)&d_x_vals, (void*)x_vals, 0 ) ); 
      try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)&d_y_vals, (void*)y_vals, 0 ) ); 
      try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)&d_results, (void*)results, 0 ) ); 

      exp_vals = new float4[num_items];      

      ////////////////////////////////////////////////////////////////////
      // GENERATE KERNEL DATA
      ////////////////////////////////////////////////////////////////////
      std::cout << "Trying to generate input data...\n";
      
      // initialize x_vals and y arrays on the host
      int seed  = 5;
      srand(seed);
      for ( int index = 0; index < num_items; index++ ) {
         x_vals[index] = (float4){
            ((float)rand() * RAND_MAX_RECIP * (float)num_items),
            ((float)rand() * RAND_MAX_RECIP * (float)num_items),
            ((float)rand() * RAND_MAX_RECIP * (float)num_items),
            ((float)rand() * RAND_MAX_RECIP * (float)num_items)
         };
         y_vals[index] = (float4){
            ((float)rand() * RAND_MAX_RECIP * (float)num_items),
            ((float)rand() * RAND_MAX_RECIP * (float)num_items),
            ((float)rand() * RAND_MAX_RECIP * (float)num_items),
            ((float)rand() * RAND_MAX_RECIP * (float)num_items)
         };
      }

      ////////////////////////////////////////////////////////////////////
      // GENERATE EXPECTED DATA
      ////////////////////////////////////////////////////////////////////
      std::cout << "Trying to generate expected outputs...\n";
      Time_Point start = Steady_Clock::now();

      // Generate expected results
      for( int index = 0; index < num_items; index++ ) {
         exp_vals[index] = x_vals[index] + y_vals[index];

      } 
      Duration_ms duration_ms = Steady_Clock::now() - start;

      std::cout << __func__ << "(): CPU: It took " << duration_ms.count() 
         << " milliseconds to add " << num_items << " values\n"; 
      
      float seconds = duration_ms.count() * std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
      float actual_num_items = num_items * 4.0;
      
      std::cout << __func__ << "(): CPU: That's a rate of " << (actual_num_items/seconds) 
         << " float additions per second\n"; 

      ////////////////////////////////////////////////////////////////////
      // RUN KERNEL
      /////////////////////////////////////////////////////////////////////
      cudaEvent_t start_event;
      cudaEvent_t stop_event;
      std::string kernel_name = "Add";

      int threads_per_block = 256;
      int num_blocks = (num_items + threads_per_block - 1) / threads_per_block;
    
      std::cout << "Trying to run CUDA kernel\n";

      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n"; 

      try_cuda_func( cerror, cudaEventCreate(&start_event) );
      try_cuda_func( cerror, cudaEventCreate(&stop_event) );

      try_cuda_func( cerror, cudaEventRecord(start_event) );

      add<<<num_blocks, threads_per_block, 0>>>( d_results, d_x_vals, d_y_vals, num_items );

      try_cuda_func( cerror, cudaEventRecord(stop_event) );
      try_cuda_func( cerror, cudaEventSynchronize(stop_event) );

      float add_milliseconds = 0;
      try_cuda_func( cerror, cudaEventElapsedTime(&add_milliseconds, start_event, stop_event) );

      std::cout << "GPU: Time for asynchronous transfer and " << kernel_name.c_str() 
         << " kernel execution: " << add_milliseconds << " milliseconds for " 
         << num_items << " items\n";

      float add_seconds = add_milliseconds * std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
      actual_num_items = num_items * 4.0;

      std::cout << "GPU: That's a rate of " << (actual_num_items/add_seconds) << " float additions per second\n"; 

      try_cuda_func( cerror, cudaEventDestroy(start_event) );
      try_cuda_func( cerror, cudaEventDestroy(stop_event) );

      //check_kernel( results, exp_vals, num_items );
      float4 max_error = {2.0, 2.0, 2.0, 2.0};
      std::cout << "Checking kernel outputs against the expected outputs...\n";
      for ( int index = 0; index < num_items; ++index ) {
         if ( fabs( exp_vals[index] - results[index] ) > max_error ) {
            std::cout << "ERROR: Mismatch for Result " << index << ": " 
               << results[index] << " when the expected result was "
               << exp_vals[index] << "\n";
         }
      }
      
      std::cout << "Freeing memory used for input and output data...\n";

      try_cuda_free_host( cerror, x_vals );
      try_cuda_free_host( cerror, y_vals );
      try_cuda_free_host( cerror, results );

      delete_array( exp_vals );
      return SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << "ERROR: " << ex.what() << "\n"; 
      return FAILURE;
   }
}
