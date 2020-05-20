
#include "add_kernels.cuh"
#include "add.cuh"


//////////////////////////////////////
// Init Kernel
// Allocates memory and initializes 
// input data (x_vals, y_vals)
//////////////////////////////////////
int init_kernel( float4** x_vals, float4** y_vals, float4** results, 
    float4** exp_vals, float4** d_x_vals, float4** d_y_vals, 
    float4** d_results, size_t* num_bytes, const int num_items ) {
  
  cudaError_t cerror = cudaSuccess;
  *num_bytes = num_items * sizeof(float4);
 
  // Allocate host memory that is also accessible by the device. Now assuming
  // that the host and device share memory (Tegra Xavier);
  // For zero-copy access
  try_cuda_func( cerror, cudaHostAlloc( (void**)x_vals, *num_bytes, cudaHostAllocMapped ) ); 
  try_cuda_func( cerror, cudaHostAlloc( (void**)y_vals, *num_bytes, cudaHostAllocMapped ) ); 
  try_cuda_func( cerror, cudaHostAlloc( (void**)results, *num_bytes, cudaHostAllocMapped ) ); 
  
  try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)d_x_vals, (void*)*x_vals, 0 ) ); 
  try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)d_y_vals, (void*)*y_vals, 0 ) ); 
  try_cuda_func( cerror, cudaHostGetDevicePointer( (void**)d_results, (void*)*results, 0 ) ); 

  try_new( float4, *exp_vals, num_items );
  return SUCCESS;
}


//////////////////////////////////////
// Generate inputs for the kernel  
//////////////////////////////////////
int gen_kernel_data( float4* x_vals, float4* y_vals, const int num_items, const unsigned int seed ) {
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

  return SUCCESS;
}


//////////////////////////////////////
// Generate Expected results
// Also benchmark the CPU version
//////////////////////////////////////
int gen_expected_data( float4* const x_vals, float4* const y_vals, float4* exp_vals, const int num_items ) {
  
  float milliseconds = 0.0;
  Time_Point start = Steady_Clock::now();

  // Generate expected results
  for( int index = 0; index < num_items; index++ ) {
    exp_vals[index].x = x_vals[index].x + y_vals[index].x;
    exp_vals[index].y = x_vals[index].y + y_vals[index].y;
    exp_vals[index].z = x_vals[index].z + y_vals[index].z;
    exp_vals[index].w = x_vals[index].w + y_vals[index].w;
  } 
  Time_Point stop = Steady_Clock::now();
  Duration_ms duration_ms = stop - start;
  milliseconds = duration_ms.count();
  
  printf( "CPU: It took %f milliseconds for the CPU to add %d values\n", milliseconds, num_items );
  float seconds = milliseconds / MILLISECONDS_PER_SECOND;
  float actual_num_items = num_items * 4.0;
  printf( "CPU: That's a rate of %f float additions per second\n", (actual_num_items/seconds) );
  return SUCCESS;
}

std::string get_kernel_name( int kernel_sel ) {
  std::string kernel_name;
  switch( kernel_sel ) {
    case ROLLED_SEL_VAL:
      kernel_name = "Rolled Add";
      break;
    case UNROLLED_FOUR_SEL_VAL:
      kernel_name = "Unrolled by Four Add";
      break;
    case UNROLLED_EIGHT_SEL_VAL:
      kernel_name = "Unrolled by Eight Add";
      break;
    default:
      kernel_name = "Unknown name";
      break;
  }
  return kernel_name;
}


//////////////////////////////////////
// Run Kernel
// Sends data to device, launches the kernel
// Receives data from the device
//////////////////////////////////////
int run_kernel( float4* const x_vals, float4* const y_vals, float4* results,
    float4* const d_x_vals, float4* const d_y_vals, float4* d_results, 
    const size_t num_bytes, const int num_items, const int kernel_sel ) {

  cudaError_t cerror = cudaSuccess;
  cudaEvent_t start;
  cudaEvent_t stop;
  std::string kernel_name = get_kernel_name( kernel_sel );
  
  int threads_per_block = 256;

  int num_blocks = (num_items + threads_per_block - 1) / threads_per_block;

  printf( "threads_per_block is %d\n", threads_per_block );
  printf( "num_blocks is %d\n", num_blocks );
  
  try_cuda_func( cerror, cudaEventCreate(&start) );
  try_cuda_func( cerror, cudaEventCreate(&stop) );

  try_cuda_func( cerror, cudaEventRecord(start) );

  switch( kernel_sel ) {
    case ROLLED_SEL_VAL:
      add_rolled<<<num_blocks, threads_per_block, 0>>>( d_results, d_x_vals, d_y_vals, num_items);
      break;
    case UNROLLED_FOUR_SEL_VAL:
      add_unrolled_four<<<num_blocks, threads_per_block, 0>>>( d_results, d_x_vals, d_y_vals, num_items);
      break;
    case UNROLLED_EIGHT_SEL_VAL:
      add_unrolled_eight<<<num_blocks, threads_per_block, 0>>>( d_results, d_x_vals, d_y_vals, num_items);
      break;
    default:
      add_rolled<<<num_blocks, threads_per_block, 0>>>( d_results, d_x_vals, d_y_vals, num_items);
  }
  try_cuda_func( cerror, cudaEventRecord(stop) );
  try_cuda_func( cerror, cudaEventSynchronize(stop) );
  
  float milliseconds = 0;
  try_cuda_func( cerror, cudaEventElapsedTime(&milliseconds, start, stop) );

  printf( "GPU: Time for asynchronous transfer and %s kernel execution: "
      "%f milliseconds for %d items\n", kernel_name.c_str(), milliseconds, num_items );
  float seconds = milliseconds / MILLISECONDS_PER_SECOND;
  float actual_num_items = num_items * 4.0;
  printf( "GPU: That's a rate of %f float additions per second\n", (actual_num_items/seconds) );
  
  try_cuda_func( cerror, cudaEventDestroy(start) );
  try_cuda_func( cerror, cudaEventDestroy(stop) );
  
  return SUCCESS;
}


//////////////////////////////////////
// Check Kernel
// Compares kernel output to expected 
// output
//////////////////////////////////////
int check_kernel( float4* const results, float4* const exp_vals, const int num_items ) {
  
  float max_error = 2.0;
  for ( int index = 0; index < num_items; index++ ) {

    if ( fabs( exp_vals[index].x - results[index].x ) > max_error ) {
      printf( "ERROR: Mismatch for Result %d.x: %f when the expected "
          "result was %f\n", index, results[index].x, exp_vals[index].x );
      return FAILURE;
    }
    if ( fabs( exp_vals[index].y - results[index].y ) > max_error ) {
      printf( "ERROR: Mismatch for Result %d.y: %f when the expected "
          "result was %f\n", index, results[index].y, exp_vals[index].y );
      return FAILURE;
    }
    if ( fabs( exp_vals[index].z - results[index].z ) > max_error ) {
      printf( "ERROR: Mismatch for Result %d.z: %f when the expected "
          "result was %f\n", index, results[index].z, exp_vals[index].z );
      return FAILURE;
    }
    if ( fabs( exp_vals[index].w - results[index].w ) > max_error ) {
      printf( "ERROR: Mismatch for Result %d.w: %f when the expected "
          "result was %f\n", index, results[index].w, exp_vals[index].w );
      return FAILURE;
    }
  } // end of for loop

  printf( "Kernel Outputs matched expected\n" );
  return SUCCESS;
}


//////////////////////////////////////
// Deinit Kernel
// Free's allocated memory
//////////////////////////////////////
int deinit_kernel( float4* x_vals, float4* y_vals, float4* results,
    float4* exp_vals, float4* d_x_vals, float4* d_y_vals, 
    float4* d_results ) {

  cudaError_t cerror = cudaSuccess;

  try_cuda_free_host( cerror, x_vals );
  try_cuda_free_host( cerror, y_vals );
  try_cuda_free_host( cerror, results );
  
  try_delete( exp_vals );

  return SUCCESS;
}



