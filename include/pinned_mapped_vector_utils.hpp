#pragma once

#include "pinned_mapped_allocator.hpp"

#include "my_cuda_utils.hpp"

#include "my_utils.hpp"

void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, const float4& lower, const float4& upper, const int& seed ); 
void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, const float& lower, const float& upper, const int& seed ); 
void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, const float4& lower, const float4& upper ); 
void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals, const float& lower, const float& upper ); 
void gen_float4s( pinned_mapped_vector<float4>& vals, const int& num_vals ); 

bool all_float4s_close( const pinned_mapped_vector<float4>& actual_vals, const std::vector<float4>& exp_vals, 
    const float& max_diff, const bool& debug );
