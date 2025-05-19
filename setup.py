import os
import subprocess
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import Extension

if __name__ == '__main__':
    # Get device type from environment variable
    device_type = os.getenv('XEN_MOE_DEVICE', 'gpu').lower()
    if device_type not in ['gpu', 'tpu']:
        raise ValueError("XEN_MOE_DEVICE must be either 'gpu' or 'tpu'")

    # Common compiler flags
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']

    # Source files
    sources = ['csrc/xen_moe.cpp']

    # Setup extension modules
    ext_modules = []

    if device_type == 'gpu':
        # GPU-specific configuration
        nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
        if nvshmem_dir is not None:
            assert os.path.exists(nvshmem_dir), 'Failed to find NVSHMEM'
            print(f'NVSHMEM directory: {nvshmem_dir}')

        # Set CUDA architecture for GPU builds
        if os.getenv('TORCH_CUDA_ARCH_LIST', None) is None:
            os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # Default to Hopper architecture

        # GPU-specific compiler flags
        nvcc_flags = ['-O3', '-Xcompiler', '-O3', '-rdc=true', '--ptxas-options=--register-usage-level=10']

        # Add GPU kernel sources
        sources.extend([
            'csrc/kernels/runtime.cu',
            'csrc/kernels/intranode.cu',
            'csrc/kernels/internode.cu',
            'csrc/kernels/internode_ll.cu'
        ])

        if nvshmem_dir is not None:
            include_dirs = ['csrc/', f'{nvshmem_dir}/include']
            library_dirs = [f'{nvshmem_dir}/lib']

            # Disable aggressive PTX instructions if requested
            if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '0')):
                cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
                nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

            # Disable DLTO (default by PyTorch)
            nvcc_dlink = ['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem']
            extra_link_args = ['-l:libnvshmem.a', '-l:nvshmem_bootstrap_uid.so', f'-Wl,-rpath,{nvshmem_dir}/lib']
            extra_compile_args = {
                'cxx': cxx_flags,
                'nvcc': nvcc_flags,
                'nvcc_dlink': nvcc_dlink
            }

            ext_modules.append(
                CUDAExtension(
                    name='xen_moe_cpp',
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    sources=sources,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    define_macros=[('XEN_MOE_GPU', None)]
                )
            )
    else:  # TPU build
        # TPU-specific configuration
        xla_dir = os.getenv('XLA_DIR', None)
        if xla_dir is None:
            raise ValueError("XLA_DIR environment variable must be set for TPU builds")

        # Add TPU kernel sources
        sources.extend([
            'csrc/tpu_kernels/moe_all_to_all_tpu.cc'
        ])

        # TPU-specific include directories
        include_dirs = [
            'csrc/',
            f'{xla_dir}/include',
            f'{xla_dir}/include/xla',
            f'{xla_dir}/include/tensorflow',
            f'{xla_dir}/include/absl',
            f'{xla_dir}/include/protobuf'
        ]

        # TPU-specific compiler flags
        cxx_flags.extend([
            '-std=c++17',
            '-D_GLIBCXX_USE_CXX11_ABI=1',
            '-DXLA_ENABLE_XLIR=1'
        ])

        ext_modules.append(
            Extension(
                name='xen_moe_cpp',
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=cxx_flags,
                libraries=['xla', 'tensorflow_framework', 'protobuf'],
                library_dirs=[f'{xla_dir}/lib'],
                define_macros=[('XEN_MOE_TPU', None)]
            )
        )

    # Get git revision for version
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='xen_moe',
        version='1.0.0' + revision,
        packages=setuptools.find_packages(
            include=['xen_moe']
        ),
        ext_modules=ext_modules,
        cmdclass={
            'build_ext': BuildExtension
        }
    )
