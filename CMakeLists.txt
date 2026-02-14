cmake_minimum_required(VERSION 3.25)
project(TeaLeaf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# optimization flags(Release Mode by default)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Define sources and headers
# Source files in src/
set(SRC_FILES
    src/data.cpp
    src/build_field.cpp
    src/definitions.cpp
    src/field_summary.cpp
    src/generate_chunk.cpp
    src/global_mpi.cpp
    src/initialise_chunk.cpp
    src/initialise.cpp
    src/pack.cpp
    src/read_input.cpp
    src/set_field.cpp
    src/start.cpp
    src/tea.cpp
    src/update_halo.cpp
    src/diffuse.cpp
    src/timestep.cpp
    src/tea_solve.cpp
    src/tea_leaf_common.cpp
    src/field_summary.cpp
    src/tea_leaf_jacobi.cpp
    src/visit.cpp
    src/calc_dt.cpp
)

# Kernel files in src/kernels/
set(KERNEL_FILES
    src/kernels/generate_chunk_kernel.cpp
    src/kernels/initialise_chunk_kernel.cpp
    src/kernels/pack_kernel.cpp
    src/kernels/set_field_kernel.cpp
    src/kernels/field_summary_kernel.cpp
    src/kernels/update_halo_kernel.cpp
    src/kernels/update_internal_halo_kernel.cpp
    src/kernels/tea_leaf_common_kernel.cpp
    src/kernels/tea_leaf_jacobi_kernel.cpp
)

# Combine all library sources
set(LIB_SOURCES ${SRC_FILES} ${KERNEL_FILES})

# Headers
set(LIB_HEADERS
    src/include/data.h
    src/include/build_field.h
    src/include/definitions.h
    src/include/field_summary.h
    src/include/generate_chunk.h
    src/include/global_mpi.h
    src/include/initialise_chunk.h
    src/include/initialise.h
    src/include/pack.h
    src/include/read_input.h
    src/include/set_field.h
    src/include/start.h
    src/include/tea.h
    src/include/update_halo.h
    src/include/diffuse.h
    src/include/timestep.h
    src/include/tea_solve.h
    src/include/tea_leaf_common.h
    src/include/field_summary.h
    src/include/visit.h
    src/include/calc_dt.h
    src/include/tea_leaf_jacobi.h
    src/include/kernels/tea_leaf_common_kernel.h
    src/include/kernels/tea_leaf_jacobi_kernel.h
    src/include/kernels/generate_chunk_kernel.h
    src/include/kernels/initialise_chunk_kernel.h
    src/include/kernels/pack_kernel.h
    src/include/kernels/set_field_kernel.h
    src/include/kernels/field_summary_kernel.h
    src/include/kernels/update_halo_kernel.h
    src/include/kernels/update_internal_halo_kernel.h
)

# create library (not yet executable)
# STATIC creates .a or .lib. 
add_library(tealeaf_lib STATIC ${LIB_SOURCES} ${LIB_HEADERS})

# Link MPI and OpenMP to the library
target_link_libraries(tealeaf_lib PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)


# Include Directories
# This allows you to do #include "include/tea.h" AND #include "include/kernels/pack_kernel.h"
target_include_directories(tealeaf_lib PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
)

# create main.cpp executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(tealeaf src/main.cpp)
    
    # link executable to library
    target_link_libraries(tealeaf PRIVATE tealeaf_lib)
endif()