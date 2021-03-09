// C++ File for my_cuda_utils

#include "my_cuda_utils.hpp"


// operator overloads for float4
// Overload the + operator
float4 operator+( const float4& lval, const float4& rval ) {
  float4 temp;
  temp.x = lval.x + rval.x;
  temp.y = lval.y + rval.y;
  temp.z = lval.z + rval.z;
  temp.w = lval.w + rval.w;
  return temp;
}

float4 operator-( const float4& lval, const float4& rval ) {
  float4 temp;
  temp.x = lval.x - rval.x;
  temp.y = lval.y - rval.y;
  temp.z = lval.z - rval.z;
  temp.w = lval.w - rval.w;
  return temp;
}

bool operator>( const float4& lval, const float4& rval ) {
  return (
    (lval.x > rval.x) &&
    (lval.y > rval.y) &&
    (lval.z > rval.z) &&
    (lval.w > rval.w) );
}

bool operator<( const float4& lval, const float4& rval ) {
  return (
    (lval.x < rval.x) &&
    (lval.y < rval.y) &&
    (lval.z < rval.z) &&
    (lval.w < rval.w) );
}

bool operator==( const float4& lval, const float4& rval ) {
  return (
    (lval.x == rval.x) &&
    (lval.y == rval.y) &&
    (lval.z == rval.z) &&
    (lval.w == rval.w) );
}


float4 fabs( const float4& val ) {
   return float4{
      (float)fabs( val.x ),
      (float)fabs( val.y ),
      (float)fabs( val.z ),
      (float)fabs( val.w )
   };
}


void gen_float4s( float4* vals, const int& num_vals, 
    const float4& lower, const float4& upper, const int& seed ) {
   
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<float> dist_x(lower.x, upper.x);
   std::uniform_real_distribution<float> dist_y(lower.y, upper.y);
   std::uniform_real_distribution<float> dist_z(lower.z, upper.z);
   std::uniform_real_distribution<float> dist_w(lower.w, upper.w);

   for( int index = 0; index < num_vals; ++index ) {
      vals[index].x = dist_x( mersenne_gen );
      vals[index].y = dist_y( mersenne_gen );
      vals[index].z = dist_z( mersenne_gen );
      vals[index].w = dist_w( mersenne_gen );
   } 
}


void gen_float4s( float4* vals, const int& num_vals, 
   const float& lower, const float& upper, const int& seed ) {

   gen_float4s( vals, num_vals, float4{ lower, lower, lower, lower },
         float4{ upper, upper, upper, upper }, seed );
}

//std::random_device random_dev;
void gen_float4s( float4* vals, const int& num_vals, 
   const float4& lower, const float4& upper ) {

   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<float> dist_x(lower.x, upper.x);
   std::uniform_real_distribution<float> dist_y(lower.y, upper.y);
   std::uniform_real_distribution<float> dist_z(lower.z, upper.z);
   std::uniform_real_distribution<float> dist_w(lower.w, upper.w);

   for( int index = 0; index < num_vals; ++index ) {
      vals[index].x = dist_x( mersenne_gen );
      vals[index].y = dist_y( mersenne_gen );
      vals[index].z = dist_z( mersenne_gen );
      vals[index].w = dist_w( mersenne_gen );
   } 
}

void gen_float4s( float4* vals, const int& num_vals, 
   const float& lower, const float& upper ) {

   gen_float4s( vals, num_vals, float4{ lower, lower, lower, lower },
      float4{ upper, upper, upper, upper } );
}


void gen_float4s( std::vector<float4>& vals, const int& num_vals, 
   const float4& lower, const float4& upper, const int& seed ) {
   
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

void gen_float4s( std::vector<float4>& vals, const int& num_vals, 
   const float& lower, const float& upper, const int& seed ) {

   gen_float4s( vals, num_vals, float4{ lower, lower, lower, lower },
      float4{ upper, upper, upper, upper }, seed );
}


void gen_float4s( std::vector<float4>& vals, const int& num_vals, 
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

void gen_float4s( std::vector<float4>& vals, const int& num_vals, 
   const float& lower, const float& upper ) {

   gen_float4s( vals, num_vals, float4{ lower, lower, lower, lower },
      float4{ upper, upper, upper, upper } );
}
// end of C++ file for my_cuda_utils
