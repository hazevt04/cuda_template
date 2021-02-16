// C++ File for utils

#include "my_utils.hpp"

// Just in case there is no intrinsic
// From Hacker's Delight
int my_popcount(unsigned int x) {
   x -= ((x >> 1) & 0x55555555);
   x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
   x = (x + (x >> 4)) & 0x0F0F0F0F;
   x += (x >> 8);
   x += (x >> 16);    
   return x & 0x0000003F;
}


// From stack overflow:
// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
// for string delimiter
std::vector<std::string> split_string(std::string str, std::string delimiter=" ") {
   size_t pos_start = 0;
   size_t pos_end = 0; 
   size_t delim_len = delimiter.length();
   std::string token;
   std::vector<std::string> split_strings;

   while ( (pos_end = str.find(delimiter, pos_start) ) != std::string::npos ) {
      token = str.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + delim_len;
      split_strings.push_back(token);
   }

   split_strings.push_back(str.substr(pos_start));
   return split_strings;
}


// variadic free function!
int free_these(void *arg1, ...) {
   va_list args;
   void *vp;
   if ( arg1 ) free(arg1);
   va_start(args, arg1);
   while ((vp = va_arg(args, void *)) != 0)
      if ( vp ) free(vp);
   va_end(args);
   return SUCCESS;
}


// end of C++ file for utils
