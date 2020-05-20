#ifndef __MY_FILE_IO_FUNCS_H__
#define __MY_FILE_IO_FUNCS_H__

#include "utils.h"
#include <string.h>
#include <sstream>
#include <fstream>

void write_binary_floats_file( float* vals, const char* filename, const int num_vals, const bool debug );

void check_file_size( long long& file_size, const char* filename, const bool debug );

void read_binary_floats_file( float* vals, const char* filename, const int num_vals, const bool debug );

void test_my_file_io_funcs( std::string filename, const int num_vals, const bool inject_error, const bool debug );

#endif // end of #ifndef __MY_FILE_IO_FUNCS_H__
