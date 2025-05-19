cmake_minimum_required(VERSION 3.18)
project(xen_moe_tpu_kernels)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(XLA REQUIRED)
find_package(JAX REQUIRED)
find_package(CUDA REQUIRED)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${XLA_INCLUDE_DIRS}
    ${JAX_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Set source files
set(SOURCES
    moe_all_to_all_tpu.cc
)

# Create library
add_library(xen_moe_tpu_kernels SHARED ${SOURCES})

# Set library properties
set_target_properties(xen_moe_tpu_kernels PROPERTIES
    PREFIX ""
    OUTPUT_NAME "xen_moe_tpu_kernels"
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(xen_moe_tpu_kernels
    ${XLA_LIBRARIES}
    ${JAX_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Set compile definitions
target_compile_definitions(xen_moe_tpu_kernels PRIVATE
    XLA_CUSTOM_CALL_TARGET
    JAX_CUSTOM_CALL_TARGET
)

# Set compile options
target_compile_options(xen_moe_tpu_kernels PRIVATE
    -Wall
    -Wextra
    -Werror
    -O3
)

# Install rules
install(TARGETS xen_moe_tpu_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    moe_all_to_all_tpu.h
    DESTINATION include/xen_moe/tpu_kernels
) 