cmake_minimum_required( VERSION 3.8 )

include(CheckLanguage)
check_language(CUDA)

project( foo LANGUAGES CXX CUDA )

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

set(CUDA_VERBOSE_BUILD ON)

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(SRCS
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/add_kernel.cu"
   "src/AddGPU.cu"
   "src/main.cpp"
   )

include_directories( 
   "include"
   "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)
get_property(inc_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(inc_dir ${inc_dirs})
  message(STATUS "include dir='${inc_dir}'")
endforeach()

add_executable( ${EXEC_NAME} "${SRCS}" )
target_link_libraries( ${EXEC_NAME} ${CUDA_cudadevrt_LIBRARY} m )

set_target_properties( ${EXEC_NAME}
   PROPERTIES
   CXX_STANDARD 11
   CXX_STANDARD_REQUIRED YES
   CXX_EXTENSIONS NO
   CUDA_SEPARABLE_COMPILATION ON
   CMAKE_CUDA_STANDARD 11
   CMAKE_CUDA_STANDARD_REQUIRED ON
)
