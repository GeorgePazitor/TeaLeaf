#include "tea.h"
#include "data.h"
#include "definitions.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

class InputParser {

std::ifstream file;
std::istringstream current_line_ss;
std::string current_line;

public:
    InputParser(const std::string& filename);

    void reset();

    bool next_line();

    std::string get_word();

    int get_int();
};

void read_input();