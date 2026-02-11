#pragma once
#include <string>

bool file_exists(const std::string& name);

// Helper to strip comments and copy (Replaces parse_module logic for this snippet)
void clean_input_file(const std::string& input_file, const std::string& output_file);

void initialise();

