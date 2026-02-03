cmake_minimum_required(VERSION 3.25)
project(TeaLeaf)

# Define variables
set( headers
    src/definitions.hpp
    
)

set( sources
    src/definitions.hpp
    
)

# create library (not yet executable)
# STATIC creates .a or .lib. 
add_library(tealeaf_lib STATIC ${sources})

# configure header files with FILE_SET (CMake 3.25+)
# Means ".hpp are the public interface of the library"
target_sources(tealeaf_lib PUBLIC
    FILE_SET HEADERS
    BASE_DIRS .
    FILES ${headers}
)

# create main.cpp executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(TeaLeaf main.cpp)
    
    # link executable to library
    target_link_libraries(TeaLeaf PRIVATE tealeaf_lib)
endif()