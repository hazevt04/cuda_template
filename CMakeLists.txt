cmake_minimum_required( VERSION 3.10.0 )

project( foo LANGUAGES CXX CUDA )


# Force CUDA code to use c++11 AND 
# host code to use C++17 AND 
# make sure it doesn't use GNU std
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

SET(CUDA_SEPARABLE_COMPILATION ON)

# "NVIDIA, from whence does thine scorn for devs cometh from?"
# As usual, NVIDIA libraries are a pain to use sometimes....
# CUDA has "first-class support" with CMake... but I still need
# find_package(CUDA REQUIRED) to define variables for the 
# CUDA libraries and headers, e.g. $CUDA_INCLUDE_DIRS, $CUDA_cufft_LIBRARIES, and etc.
# #FACEPALM
find_package(CUDA REQUIRED)

# Selects GPU arch flags for nvcc based on target_CUDA_architectures
# Xavier is compute capability 7.2
# Support included for Pascal and Volta microarchitectures
#cuda_select_nvcc_arch_flags(ARCH_FLAGS 6.0 6.1 6.2 7.0 7.2)
#cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

set(INC_DIRS
   "include"
   ${CUDA_INCLUDE_DIRS}
   )

set(SRCS
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/my_cuda_utils.cpp"
   "src/pinned_mapped_vector_utils.cpp"
   "src/main.cu"
   )

message("INC_DIRS: ${INC_DIRS}")

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

include_directories( "${INC_DIRS}" )

add_executable( ${EXEC_NAME} "${SRCS}" )
target_link_libraries( ${EXEC_NAME} ${CUDA_cudadevrt_LIBRARY} m )

