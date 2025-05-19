import os
import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union

# noinspection PyUnresolvedReferences
import xen_moe_cpp
# noinspection PyUnresolvedReferences
from xen_moe_cpp import Config, EventHandle
from .utils import EventOverlap
from .backend import get_backend, CommBackend

class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        backend: the communication backend (GPU or TPU).
    """

    num_sms: int = 20

    def __init__(self, group: dist.ProcessGroup,
                 num_nvl_bytes: int = 0, num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False, num_qps_per_rank: int = 12,
                 device_type: str = 'gpu', batch_size: int = 128,
                 hidden_dim: int = 1024, num_experts: int = 8) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            device_type: type of device ('gpu' or 'tpu').
            batch_size: the batch size for auto-tuning buffer sizes.
            hidden_dim: the hidden dimension for auto-tuning buffer sizes.
            num_experts: the number of experts for auto-tuning buffer sizes.
        """
        # Initialize the communication backend
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.low_latency_mode = low_latency_mode
        self.device_type = device_type
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Auto-tune buffer sizes if not specified
        if num_nvl_bytes == 0 or num_rdma_bytes == 0:
            if device_type == 'gpu':
                # Get device properties for auto-tuning
                device = torch.cuda.current_device()
                prop = torch.cuda.get_device_properties(device)
                
                # Get optimal buffer sizes
                buffer_sizes = xen_moe_cpp.AutoTuneConfig.get_buffer_sizes(
                    batch_size, hidden_dim, num_experts
                )
                
                # Override with user-specified values if provided
                self.num_nvl_bytes = num_nvl_bytes if num_nvl_bytes > 0 else buffer_sizes.nvl_bytes
                self.num_rdma_bytes = num_rdma_bytes if num_rdma_bytes > 0 else buffer_sizes.rdma_bytes
            else:
                # Default sizes for TPU
                self.num_nvl_bytes = batch_size * hidden_dim * 2  # 2x for double buffering
                self.num_rdma_bytes = batch_size * hidden_dim * 4  # 4x for RDMA buffering
        else:
            self.num_nvl_bytes = num_nvl_bytes
            self.num_rdma_bytes = num_rdma_bytes

        # Initialize backend
        self.backend = get_backend(
            device_type,
            num_tokens=self.num_nvl_bytes // 2,  # Approximate number of tokens
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_workers=self.group_size
        )

        # Initialize C++ runtime for GPU backend
        if device_type == 'gpu':
            self.runtime = xen_moe_cpp.Buffer(
                self.rank,
                self.group_size,
                self.num_nvl_bytes,
                self.num_rdma_bytes,
                low_latency_mode
            )

            # Synchronize device IDs
            device_ids = [None, ] * self.group_size
            local_device_id = self.runtime.get_local_device_id()
            dist.all_gather_object(device_ids, local_device_id, group)

            # Synchronize IPC handles
            ipc_handles = [None, ] * self.group_size
            local_ipc_handle = self.runtime.get_local_ipc_handle()
            dist.all_gather_object(ipc_handles, local_ipc_handle, group)

            # Synchronize NVSHMEM unique IDs
            root_unique_id = None
            if self.rank == 0:
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            root_unique_id = dist.broadcast_object_list([root_unique_id], 0, group)[0]
            self.runtime.sync(root_unique_id)

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'backend'):
            self.backend.cleanup()
        if hasattr(self, 'runtime'):
            del self.runtime

    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[Tuple] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None,
                 num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None,
                 num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 expert_alignment: int = 1,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventOverlap] = None,
                 async_finish: bool = False,
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
                  Optional[torch.Tensor],
                  Optional[torch.Tensor],
                  List[int],
                  Tuple,
                  EventOverlap]:
        """
        Dispatch tokens to different ranks using the appropriate backend.
        """
        if self.device_type == 'gpu':
            # Use GPU backend with NVSHMEM
            if self.runtime.get_num_rdma_ranks() > 1:
                return self.internode_dispatch(
                    x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank,
                    is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights,
                    expert_alignment, config, previous_event, async_finish,
                    allocate_on_comm_stream
                )
            else:
                return self.intranode_dispatch(
                    x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank,
                    is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights,
                    expert_alignment, config, previous_event, async_finish,
                    allocate_on_comm_stream
                )
        else:
            # Use TPU backend with JAX/XLA
            x, x_scales = x if isinstance(x, tuple) else (x, None)
            dispatched = self.backend.dispatch(
                x,
                topk_idx,
                num_tokens_per_expert.size(0) if num_tokens_per_expert is not None else 8,
                self.group_size
            )
            return (dispatched, x_scales) if x_scales is not None else dispatched, \
                   topk_idx, topk_weights, \
                   num_tokens_per_expert.tolist() if num_tokens_per_expert is not None else None, \
                   handle, EventOverlap(None)

    def combine(self, x: torch.Tensor, handle: Tuple,
                topk_weights: Optional[torch.Tensor] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventOverlap] = None,
                async_finish: bool = False,
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine expert outputs using the appropriate backend.
        """
        if self.device_type == 'gpu':
            # Use GPU backend with NVSHMEM
            if self.runtime.get_num_rdma_ranks() > 1:
                return self.internode_combine(
                    x, handle, topk_weights, config,
                    previous_event, async_finish, allocate_on_comm_stream
                )
            else:
                return self.intranode_combine(
                    x, handle, topk_weights, config,
                    previous_event, async_finish, allocate_on_comm_stream
                )
        else:
            # Use TPU backend with JAX/XLA
            combined = self.backend.combine(
                x,
                handle[0],  # expert_assignments
                (x.size(0), x.size(1), x.size(2))  # original_shape
            )
            return combined, topk_weights, EventOverlap(None)

    # Keep existing methods for GPU backend
    def intranode_dispatch(self, *args, **kwargs):
        """Intranode dispatch implementation for GPU backend."""
        return self.runtime.intranode_dispatch(*args, **kwargs)

    def intranode_combine(self, *args, **kwargs):
        """Intranode combine implementation for GPU backend."""
        return self.runtime.intranode_combine(*args, **kwargs)

    def internode_dispatch(self, *args, **kwargs):
        """Internode dispatch implementation for GPU backend."""
        return self.runtime.internode_dispatch(*args, **kwargs)

    def internode_combine(self, *args, **kwargs):
        """Internode combine implementation for GPU backend."""
        return self.runtime.internode_combine(*args, **kwargs)

    def clean_low_latency_buffer(self, *args, **kwargs):
        """Clean low latency buffer for GPU backend."""
        if self.device_type == 'gpu':
            self.runtime.clean_low_latency_buffer(*args, **kwargs)

    def low_latency_dispatch(self, *args, **kwargs):
        """Low latency dispatch for GPU backend."""
        if self.device_type == 'gpu':
            return self.runtime.low_latency_dispatch(*args, **kwargs)
        raise NotImplementedError("Low latency dispatch not supported for TPU backend")

    def low_latency_combine(self, *args, **kwargs):
        """Low latency combine for GPU backend."""
        if self.device_type == 'gpu':
            return self.runtime.low_latency_combine(*args, **kwargs)
        raise NotImplementedError("Low latency combine not supported for TPU backend")

    def get_next_low_latency_combine_buffer(self, *args, **kwargs):
        """Get next low latency combine buffer for GPU backend."""
        if self.device_type == 'gpu':
            return self.runtime.get_next_low_latency_combine_buffer(*args, **kwargs)
        raise NotImplementedError("Low latency combine buffer not supported for TPU backend")

    # Keep existing static methods
    @staticmethod
    def get_low_latency_rdma_size_hint(*args, **kwargs):
        """Get low latency RDMA size hint."""
        return xen_moe_cpp.get_low_latency_rdma_size_hint(*args, **kwargs)

    @staticmethod
    def get_dispatch_config(*args, **kwargs):
        """Get dispatch config."""
        return Buffer.get_dispatch_config(*args, **kwargs)

    @staticmethod
    def get_combine_config(*args, **kwargs):
        """Get combine config."""
        return Buffer.get_combine_config(*args, **kwargs) 