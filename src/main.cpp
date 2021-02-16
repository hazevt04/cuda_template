// C++ File for main

#include "add.cuh"

#include "my_cuda_utils.hpp"

#include "my_utils.hpp"

// Empirically-determined maximum value for num_items. 
// 1<<29 exits with the vague 'killed' message. 
// 1<<30 causes a cuda error.
// Assuming floats, this makes each array take 1 GiB
#define MAX_NUM_ITEMS (1<<28)


void test_ilog2( const int& val ) {
   std::cout << "Integer Log2(" << val << ") is " << ilog2( val ) << "\n"; 
}


void test_is_divisible_by( const int& val, const int& div ) {
   if ( is_divisible_by( val, div ) ) {
      std::cout << val << " is divisible by " << div << "\n"; 
   }
}


int main(int argc, char **argv) {
  try {
    float4* x_vals = nullptr;
    float4* y_vals = nullptr;
    float4* results = nullptr;
    float4* exp_vals = nullptr;
    float4* d_x_vals = nullptr;
    float4* d_y_vals = nullptr;
    float4* d_results = nullptr;
    bool debug = true;

    // Empirically-determined maximum number
    int num_items = 1<<21;
    size_t num_bytes = num_items * sizeof(float4);

    // Initialize the memory
    int status = SUCCESS;

    int seed  = 5;
      
    init_kernel( &x_vals, &y_vals, &results, &exp_vals, 
      &d_x_vals, &d_y_vals, &d_results, num_bytes, num_items );

    gen_kernel_data( x_vals, y_vals, num_items, seed, debug );

    gen_expected_data( x_vals, y_vals, exp_vals, num_items, debug );

    run_kernel( x_vals, y_vals, results, d_x_vals, d_y_vals, d_results, 
      num_bytes, num_items, debug );

    check_kernel( results, exp_vals, num_items );

    deinit_kernel( x_vals, y_vals, results, exp_vals, 
      d_x_vals, d_y_vals, d_results );

    return SUCCESS;
  } catch ( std::exception& ex ) {

    std::cout << "ERROR: " << ex.what() << "\n"; 
    return FAILURE;
  }

}
// end of C++ file for main
