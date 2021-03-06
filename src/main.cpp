// C++ File for main

#include "AddGPU.cuh"

int main(int argc, char **argv) {
   try {
      int num_vals = 1 << 22;
      bool debug = false;
      AddGPU<int> add_gpu{ num_vals, debug };
      add_gpu.run();
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;

   }
}
// end of C++ file for main
