#pragma once

// My Utility Macros for CUDA

#include <cuda_runtime.h>
#include "my_utils.hpp"

// Use for Classes with CUDA (for example)
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#define check_cuda_error(cerror,loc) { \
  if ( (cerror) != cudaSuccess ) { \
    throw std::runtime_error{ ""#loc ": " + \
      std::string{cudaGetErrorString((cerror))} + \
      "(" + std::to_string((cerror)) + ")" }; \
  } \
}


#define try_cuda_func(cerror, func) { \
  (cerror) = func; \
  check_cuda_error( (cerror), func ); \
}


#define try_cuda_free( cerror, ptr ) { \
  if ((ptr)) { \
    try_cuda_func( (cerror), cudaFree((ptr))); \
    (ptr) = nullptr; \
  } \
}

#define try_cuda_free_host( cerror, ptr ) { \
  if ((ptr)) { \
    try_cuda_func( (cerror), cudaFreeHost((ptr))); \
    (ptr) = nullptr; \
  } \
}


float4 operator+( const float4& lval, const float4& rval );
float4 operator-( const float4& lval, const float4& rval );
bool operator>( const float4& lval, const float4& rval );
bool operator<( const float4& lval, const float4& rval );
bool operator==( const float4& lval, const float4& rval );
float4 fabs( const float4& val );

template<class _CharT, class _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os, const float4& val) {
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << "{" << val.x << ", " << val.y << ", " << val.z << ", " << val.w << "}";
    return __os << __s.str();
}


