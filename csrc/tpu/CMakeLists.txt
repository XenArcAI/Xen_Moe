cmake_minimum_required(VERSION 3.18)
project(xen_moe_tpu)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(absl REQUIRED)
find_package(xla REQUIRED)
find_package(pjrt REQUIRED)

# Add source files
set(SOURCES
    moe_all_to_all.cc
)

# Create library
add_library(xen_moe_tpu SHARED ${SOURCES})

# Set include directories
target_include_directories(xen_moe_tpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${absl_INCLUDE_DIRS}
        ${xla_INCLUDE_DIRS}
        ${pjrt_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(xen_moe_tpu
    PUBLIC
        absl::status
        absl::statusor
        xla::custom_call_target_registry
        xla::primitive_util
        pjrt::client
        pjrt::device
)

# Set compile options
target_compile_options(xen_moe_tpu
    PRIVATE
        -Wall
        -Wextra
        -Werror
        -O3
)

# Install rules
install(TARGETS xen_moe_tpu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    moe_all_to_all.h
    DESTINATION include/xen_moe/tpu
) 