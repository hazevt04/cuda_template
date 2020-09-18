// C++ File for main

#include "AddGPU.cuh"
#include "managed_allocator_host.hpp"

int main(int argc, char **argv) {
   try {
      int num_vals = 1 << 21;
      bool debug = true;
      AddGPU<int> add_gpu{ num_vals, debug };
      add_gpu.run();
      return EXIT_SUCCESS;
   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;
   }
}
// end of C++ file for main
