#include <numeric>
#include <memory>
#include <exception>

//#include <cuda_runtime.h>
#include "my_cuda_utils.hpp"

#include "add_kernel.cuh"

#include "managed_allocator_global.hpp"
#include "managed_allocator_host.hpp"

#include "VariadicToOutputStream.hpp"

#include <iterator>
#include <algorithm>

template<typename T>
class AddGPU {
private:
   managed_vector_host<T> lvals;
   managed_vector_host<T> rvals;
   managed_vector_host<T> exp_sums;
   managed_vector_global<T> sums;
   int num_vals;
   bool debug;

   std::unique_ptr<cudaStream_t> stream_ptr;

public:
   AddGPU():
      num_vals(0),
      debug(false) {}
   
   
   AddGPU( int new_num_vals, const bool new_debug ):
      num_vals( new_num_vals ),
      debug( new_debug ) {
   
      try {
         debug_cout( debug, __func__, "(): num_vals is ", num_vals, "\n" );
         lvals.reserve( num_vals );
         rvals.reserve( num_vals );
         exp_sums.reserve( num_vals );
         sums.reserve( num_vals );
         debug_cout( debug, __func__, "(): after reserving vectors for sums, exp_sums, lvals and rvals\n" );

         stream_ptr = my_make_unique<cudaStream_t>();
         try_cudaStreamCreate( stream_ptr.get() );
         debug_cout( debug, __func__,  "(): after cudaStreamCreate()\n" ); 

      } catch( std::exception& ex ) {
         std::cout <<  __func__ << "(): ERROR: " << ex.what() << "\n"; 
         throw;
      }
   }

   void gen_data( int seed = 0 ) {
      lvals.resize(num_vals);
      rvals.resize(num_vals);
      exp_sums.resize(num_vals);
      std::iota( lvals.begin(), lvals.end(), 1 );
      std::iota( rvals.begin(), rvals.end(), 1 );
      std::iota( exp_sums.begin(), exp_sums.end(), 1 );

      std::for_each( exp_sums.begin(), exp_sums.end(), [](T& a){ a *= 2; });
      
      if (debug) {
         int num_vals_to_print = 10;
         print_vec<T>( lvals, num_vals_to_print, "Generated Lvals:\n", " " ); 
         print_vec<T>( rvals, num_vals_to_print, "Generated Rvals:\n", " " ); 
         print_vec<T>( exp_sums, num_vals_to_print, "Expected Sums:\n", " " ); 
      }
   }

   void run();
   
   void print_sums( const int& num_vals_to_print, const std::string& prefix = "Sums: " ) {
      print_vec<T>( sums, num_vals_to_print, prefix.data(), " " );
   }

   ~AddGPU() {
      debug_cout( debug, "dtor called\n" );
      lvals.clear();
      rvals.clear();
      sums.clear();    
      if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );
      debug_cout( debug, "dtor done\n" );
   }

};


