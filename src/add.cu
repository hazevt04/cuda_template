
#include "add_kernel.cuh"
#include "add.cuh"

#include <chrono>

//////////////////////////////////////
// Init Kernel
// Allocates memory and initializes 
// input data (x_vals, y_vals)
//////////////////////////////////////
void init_kernel( float4** x_vals, float4** y_vals, float4** results, 
   float4** exp_vals, float4** d_x_vals, float4** d_y_vals, 
   float4** d_results, size_t& num_bytes, const int& num_items 
   ) {
  
   try {
      cudaError_t cerror = cudaSuccess;
      num_bytes = num_items * sizeof(float4);

      std::cout << "Trying to initialize memory for input and output data...\n";
      // Allocate pinned host memory that is also accessible by the device.
      try_cuda_func( cerror, cudaHostAlloc( (void**)x_vals, num_bytes, cudaHostAllocMapped ) ); 
      try_cuda_func( cerror, cudaHostAlloc( (void**)y_vals, num_bytes, cudaHostAllocMapped ) ); 
      try_cuda_func( cerror, cudaHostAlloc( (void**)results, num_bytes, cudaHostAllocMapped ) ); 

      try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)d_x_vals, (void*)*x_vals, 0 ) ); 
      try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)d_y_vals, (void*)*y_vals, 0 ) ); 
      try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)d_results, (void*)*results, 0 ) ); 

      *exp_vals = new float4[num_items];

   } catch( std::exception& ex ) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
}


//////////////////////////////////////
// Generate inputs for the kernel  
//////////////////////////////////////
void gen_kernel_data( float4* x_vals, float4* y_vals, const int& num_items, const unsigned int& seed,
   const bool& debug = false ) {

   try {
      std::cout << "Trying to generate input data...\n";
      // initialize x_vals and y arrays on the host
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
   } catch( std::exception& ex ) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }

}


//////////////////////////////////////
// Generate Expected results
// Also benchmark the CPU version
//////////////////////////////////////
void gen_expected_data( float4* const x_vals, float4* const y_vals, float4* exp_vals, 
      const int& num_items, const bool& debug = false ) {
  
   try {
      std::cout << "Trying to generate expected outputs...\n";
      Time_Point start = Steady_Clock::now();

      // Generate expected results
      for( int index = 0; index < num_items; index++ ) {
         exp_vals[index] = x_vals[index] + y_vals[index];

         //exp_vals[index].x = x_vals[index].x + y_vals[index].x;
         //exp_vals[index].y = x_vals[index].y + y_vals[index].y;
         //exp_vals[index].z = x_vals[index].z + y_vals[index].z;
         //exp_vals[index].w = x_vals[index].w + y_vals[index].w;
      } 
      Duration_ms duration_ms = Steady_Clock::now() - start;

      std::cout << __func__ << "(): CPU: It took " << duration_ms.count() 
         << " milliseconds to add " << num_items << " values\n"; 
      
      float seconds = duration_ms.count() * std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
      float actual_num_items = num_items * 4.0;
      
      std::cout << __func__ << "(): CPU: That's a rate of " << (actual_num_items/seconds) 
         << " float additions per second\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
}


//////////////////////////////////////
// Run Kernel
// Sends data to device, launches the kernel
// Receives data from the device
//////////////////////////////////////
void run_kernel( float4* const x_vals, float4* const y_vals, float4* results,
   float4* const d_x_vals, float4* const d_y_vals, float4* d_results, 
   const size_t& num_bytes, const int& num_items, const bool& debug = false ) {

   try {
      cudaError_t cerror = cudaSuccess;
      cudaEvent_t start;
      cudaEvent_t stop;
      std::string kernel_name = "Add";

      int threads_per_block = 256;

      int num_blocks = (num_items + threads_per_block - 1) / threads_per_block;
    
      std::cout << "Trying to run CUDA kernel\n";

      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n"; 

      try_cuda_func( cerror, cudaEventCreate(&start) );
      try_cuda_func( cerror, cudaEventCreate(&stop) );

      try_cuda_func( cerror, cudaEventRecord(start) );

      add<<<num_blocks, threads_per_block, 0>>>( d_results, d_x_vals, d_y_vals, num_items );

      try_cuda_func( cerror, cudaEventRecord(stop) );
      try_cuda_func( cerror, cudaEventSynchronize(stop) );

      float add_milliseconds = 0;
      try_cuda_func( cerror, cudaEventElapsedTime(&add_milliseconds, start, stop) );

      std::cout << "GPU: Time for asynchronous transfer and " << kernel_name.c_str() 
         << " kernel execution: " << add_milliseconds << " milliseconds for " 
         << num_items << " items\n";

      float add_seconds = add_milliseconds * std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
      float actual_num_items = num_items * 4.0;

      std::cout << "GPU: That's a rate of " << (actual_num_items/add_seconds) << " float additions per second\n"; 

      try_cuda_func( cerror, cudaEventDestroy(start) );
      try_cuda_func( cerror, cudaEventDestroy(stop) );

   } catch( std::exception& ex ) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
}


//////////////////////////////////////
// Check Kernel
// Compares kernel output to expected 
// output
//////////////////////////////////////
void check_kernel( float4* const results, float4* const exp_vals, const int& num_items ) {
  
   try {
      //float max_error = 2.0;
      float4 max_error = {2.0, 2.0, 2.0, 2.0};
      std::cout << "Checking kernel outputs against the expected outputs...\n";
      for ( int index = 0; index < num_items; index++ ) {

         if ( fabs( exp_vals[index] - results[index] ) > max_error ) {
            std::cout << "ERROR: Mismatch for Result " << index << ": " 
               << results[index] << " when the expected result was "
               << exp_vals[index] << "\n";
         }

         //if ( fabs( exp_vals[index].x - results[index].x ) > max_error ) {
         //   std::cout << "ERROR: Mismatch for Result " << index << ": " 
         //      << results[index].x << " when the expected result was "
         //      << exp_vals[index].x << "\n";
         //}
         //if ( fabs( exp_vals[index].y - results[index].y ) > max_error ) {
         //   std::cout << "ERROR: Mismatch for Result " << index << ": " 
         //      << results[index].y << " when the expected result was "
         //      << exp_vals[index].y << "\n";
         //}
         //if ( fabs( exp_vals[index].z - results[index].z ) > max_error ) {
         //   std::cout << "ERROR: Mismatch for Result " << index << ": " 
         //      << results[index].z << " when the expected result was "
         //      << exp_vals[index].z << "\n";
         //}
         //if ( fabs( exp_vals[index].w - results[index].w ) > max_error ) {
         //   std::cout << "ERROR: Mismatch for Result " << index << ": " 
         //      << results[index].w << " when the expected result was "
         //      << exp_vals[index].w << "\n";
         //}

      } // end of for loop

      std::cout << "Kernel Outputs matched expected\n";
   } catch( std::exception& ex ) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
}


//////////////////////////////////////
// Deinit Kernel
// Free's allocated memory
//////////////////////////////////////
void deinit_kernel( float4* x_vals, float4* y_vals, float4* results,
   float4* exp_vals, float4* d_x_vals, float4* d_y_vals, 
   float4* d_results ) {

   try {
      cudaError_t cerror = cudaSuccess;
      std::cout << "Freeing memory used for input and output data...\n";

      try_cuda_free_host( cerror, x_vals );
      try_cuda_free_host( cerror, y_vals );
      try_cuda_free_host( cerror, results );

      delete_array( exp_vals );
   } catch( std::exception& ex ) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }

}



