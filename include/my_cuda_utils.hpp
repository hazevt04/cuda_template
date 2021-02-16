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



