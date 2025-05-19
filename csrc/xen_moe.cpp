#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/cast.h>
#include <pybind11/attr.h>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/class.h>
#include <pybind11/detail/common.h>
#include <pybind11/detail/descr.h>
#include <pybind11/detail/init.h>
#include <pybind11/detail/internals.h>
#include <pybind11/detail/type_caster_base.h>
#include <pybind11/detail/typeid.h>
#include <pybind11/eval.h>
#include <pybind11/options.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl/filesystem.h>

#include "xen_moe.hpp"

namespace xen_moe {
// ... existing code ...
} // namespace xen_moe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<xen_moe::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>())
        .def("get_nvl_buffer_size_hint", &xen_moe::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &xen_moe::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &xen_moe::get_low_latency_rdma_size_hint);

    pybind11::class_<xen_moe::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &xen_moe::EventHandle::current_stream_wait);

    pybind11::class_<xen_moe::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int, int, bool>())
        .def("is_available", &xen_moe::Buffer::is_available)
        .def("get_num_rdma_ranks", &xen_moe::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &xen_moe::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &xen_moe::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &xen_moe::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &xen_moe::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &xen_moe::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &xen_moe::Buffer::get_local_buffer_tensor)
        .def("sync", &xen_moe::Buffer::sync)
        .def("get_dispatch_layout", &xen_moe::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &xen_moe::Buffer::intranode_dispatch)
        .def("intranode_combine", &xen_moe::Buffer::intranode_combine)
        .def("internode_dispatch", &xen_moe::Buffer::internode_dispatch)
        .def("internode_combine", &xen_moe::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &xen_moe::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &xen_moe::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &xen_moe::Buffer::low_latency_combine)
        .def("get_next_low_latency_combine_buffer", &xen_moe::Buffer::get_next_low_latency_combine_buffer);
}
