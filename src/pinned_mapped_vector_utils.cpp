// C++ File for pinned_mapped_vector_utils

#include "pinned_mapped_vector_utils.hpp"


void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, const float4& lower, const float4& upper, const int& seed ) {
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<float> dist_x(lower.x, upper.x);
   std::uniform_real_distribution<float> dist_y(lower.y, upper.y);
   std::uniform_real_distribution<float> dist_z(lower.z, upper.z);
   std::uniform_real_distribution<float> dist_w(lower.w, upper.w);

   for( int index = 0; index < num_vals; ++index ) {
      vals.emplace_back( float4{ 
         dist_x( mersenne_gen ),
         dist_y( mersenne_gen ),
         dist_z( mersenne_gen ),
         dist_w( mersenne_gen )
      } );
   }
}


void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, const float& lower, const float& upper, const int& seed ) {
   gen_float4s( vals, num_vals, float4{ lower, lower, lower, lower },
      float4{ upper, upper, upper, upper }, seed );
}


void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, 
   const float4& lower, const float4& upper ) {
   
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<float> dist_x(lower.x, upper.x);
   std::uniform_real_distribution<float> dist_y(lower.y, upper.y);
   std::uniform_real_distribution<float> dist_z(lower.z, upper.z);
   std::uniform_real_distribution<float> dist_w(lower.w, upper.w);

   for( int index = 0; index < num_vals; ++index ) {
      vals.emplace_back( float4{ 
         dist_x( mersenne_gen ),
         dist_y( mersenne_gen ),
         dist_z( mersenne_gen ),
         dist_w( mersenne_gen )
      } );
   } 
}

void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, 
   const float& lower, const float& upper ) {

   gen_float4s( vals, num_vals, float4{ lower, lower, lower, lower },
      float4{ upper, upper, upper, upper } );
}


void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals ) {
   gen_float4s( vals, num_vals, float4{ 0.f, 0.f, 0.f, 0.f },
      float4{ float(num_vals), float(num_vals), float(num_vals), float(num_vals) } );
}


bool all_float4s_close( 
   const pinned_mapped_vector<float4>& actual_vals, const std::vector<float4>& exp_vals, 
   const float& max_diff, const bool& debug ) {

   // TODO: Fill this in with std::mismatch() or something like that
   auto mispair = std::mismatch( actual_vals.begin(), actual_vals.end(), exp_vals.begin(), 
      [&]( const float4& act, const float4& exp ) { return (fabs(act-exp) > float4{max_diff, max_diff, max_diff, max_diff}); } );
   
   if( ( mispair.first == actual_vals.end() ) ) {
      dout << "No Mismatches\n";
      // No mismatch, do something sensible
      return true;
   } else {
      dout << "Mismatch: at " << std::distance(actual_vals.begin(), mispair.first) << ":\n";
      if ( mispair.first != actual_vals.end() ) {
         dout << "First:" << *(mispair.first) << "\n";
      }
      if ( mispair.second != exp_vals.end() ) {
         dout << "Second:" << *(mispair.second) << "\n";
      }

      return false;
   }   
}

// end of C++ file for pinned_mapped_vector_utils
