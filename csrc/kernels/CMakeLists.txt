function(add_xen_moe_library target_name source_file)
    add_library(${target_name} SHARED ${source_file})
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(${target_name} PRIVATE ${EP_CUDA_LIBRARIES})
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)
endfunction()

add_xen_moe_library(intranode_cuda intranode.cu)
add_xen_moe_library(runtime_cuda runtime.cu)
add_xen_moe_library(internode_cuda internode.cu)
add_xen_moe_library(internode_ll_cuda internode_ll.cu)

# Later, we should link all libraries in `EP_CUDA_LIBRARIES`
set(EP_CUDA_LIBRARIES intranode_cuda runtime_cuda internode_cuda internode_ll_cuda PARENT_SCOPE)
