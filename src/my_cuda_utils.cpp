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

// end of C++ file for my_cuda_utils
