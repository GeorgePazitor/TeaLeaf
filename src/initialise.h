#ifndef INITIALISE_H
#define INITIALISE_H

#include <iostream>
#include <fstream>
#include <string>
#include "data.h"

// Forward declarations of functions likely in other modules
void read_input(); 
void start();

std::ostream* g_out_stream = &std::cout;
std::ofstream g_file_stream;

bool file_exists(const std::string& name);

// Helper to strip comments and copy (Replaces parse_module logic for this snippet)
void clean_input_file(const std::string& input_file, const std::string& output_file);

void initialise();

#endif 