from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
import torch
import jax
import jax.numpy as jnp
from jax.experimental import maps
import numpy as np

class CommBackend(ABC):
    """Abstract base class for communication backends."""
    
    @abstractmethod
    def dispatch(self, 
                input_tensor: Union[torch.Tensor, jnp.ndarray],
                expert_assignments: Union[torch.Tensor, jnp.ndarray],
                num_experts: int,
                num_workers: int,
                use_fp8: bool = False,
                async_mode: bool = False) -> Union[torch.Tensor, jnp.ndarray]:
        """Dispatch tokens to experts.
        
        Args:
            input_tensor: Input tensor of shape [batch_size, seq_len, hidden_dim]
            expert_assignments: Expert assignments of shape [batch_size, seq_len]
            num_experts: Number of experts
            num_workers: Number of workers
            use_fp8: Whether to use FP8 precision
            async_mode: Whether to use async dispatch
            
        Returns:
            Dispatched tensor of shape [num_experts, tokens_per_expert, hidden_dim]
        """
        pass
    
    @abstractmethod
    def combine(self,
               input_tensor: Union[torch.Tensor, jnp.ndarray],
               expert_assignments: Union[torch.Tensor, jnp.ndarray],
               original_shape: Tuple[int, int, int],
               use_fp8: bool = False,
               async_mode: bool = False) -> Union[torch.Tensor, jnp.ndarray]:
        """Combine expert outputs.
        
        Args:
            input_tensor: Input tensor of shape [num_experts, tokens_per_expert, hidden_dim]
            expert_assignments: Expert assignments of shape [batch_size, seq_len]
            original_shape: Original shape of the input tensor [batch_size, seq_len, hidden_dim]
            use_fp8: Whether to use FP8 precision
            async_mode: Whether to use async combine
            
        Returns:
            Combined tensor of shape [batch_size, seq_len, hidden_dim]
        """
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the communication backend.
        
        Args:
            **kwargs: Additional initialization parameters
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the backend."""
        pass

class GPUCommBackend(CommBackend):
    """GPU communication backend using NVSHMEM."""
    
    def __init__(self):
        self.initialized = False
        self.buffer = None
        self.low_latency_mode = False
    
    def initialize(self, **kwargs) -> None:
        """Initialize NVSHMEM and create communication buffers."""
        if self.initialized:
            return
            
        import xen_moe_cpp as xm
        
        # Initialize NVSHMEM
        xm.init()
        
        # Get configuration
        self.low_latency_mode = kwargs.get('low_latency_mode', False)
        num_tokens = kwargs.get('num_tokens', 1024)
        hidden_dim = kwargs.get('hidden_dim', 1024)
        num_experts = kwargs.get('num_experts', 8)
        num_workers = kwargs.get('num_workers', 1)
        
        # Create communication buffer
        self.buffer = xm.Buffer(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_workers=num_workers,
            low_latency_mode=self.low_latency_mode
        )
        
        self.initialized = True
    
    def dispatch(self,
                input_tensor: torch.Tensor,
                expert_assignments: torch.Tensor,
                num_experts: int,
                num_workers: int,
                use_fp8: bool = False,
                async_mode: bool = False) -> torch.Tensor:
        """Dispatch tokens to experts using NVSHMEM."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
            
        # Ensure tensors are on GPU
        input_tensor = input_tensor.cuda()
        expert_assignments = expert_assignments.cuda()
        
        if self.low_latency_mode:
            # Use low-latency dispatch
            return self.buffer.low_latency_dispatch(
                input_tensor,
                expert_assignments,
                num_experts,
                num_workers,
                use_fp8,
                async_mode
            )
        else:
            # Use standard dispatch
            return self.buffer.dispatch(
                input_tensor,
                expert_assignments,
                num_experts,
                num_workers
            )
    
    def combine(self,
               input_tensor: torch.Tensor,
               expert_assignments: torch.Tensor,
               original_shape: Tuple[int, int, int],
               use_fp8: bool = False,
               async_mode: bool = False) -> torch.Tensor:
        """Combine expert outputs using NVSHMEM."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
            
        # Ensure tensors are on GPU
        input_tensor = input_tensor.cuda()
        expert_assignments = expert_assignments.cuda()
        
        if self.low_latency_mode:
            # Use low-latency combine
            return self.buffer.low_latency_combine(
                input_tensor,
                expert_assignments,
                original_shape,
                use_fp8,
                async_mode
            )
        else:
            # Use standard combine
            return self.buffer.combine(
                input_tensor,
                expert_assignments,
                original_shape
            )
    
    def cleanup(self) -> None:
        """Clean up NVSHMEM resources."""
        if not self.initialized:
            return
            
        if self.buffer is not None:
            self.buffer.free()
            self.buffer = None
            
        import xen_moe_cpp as xm
        xm.finalize()
        
        self.initialized = False

class TPUCommBackend(CommBackend):
    """TPU communication backend using JAX/XLA."""
    
    def __init__(self):
        self.initialized = False
        self.device_mesh = None
        self.num_workers = 1
        self.precision = 'float32'
        self.use_ici = True
    
    def initialize(self, **kwargs) -> None:
        """Initialize TPU device mesh and communication primitives."""
        if self.initialized:
            return
            
        # Get configuration
        self.precision = kwargs.get('precision', 'float32')
        self.use_ici = kwargs.get('use_ici', True)
        
        # Get TPU device count
        devices = jax.devices()
        self.num_workers = len(devices)
        
        # Create device mesh for all-to-all communication
        self.device_mesh = maps.Mesh(
            devices,
            axis_names=('workers',)
        )
        
        # JIT compile communication primitives
        self._jit_dispatch = jax.jit(
            self._dispatch_impl,
            device_mesh=self.device_mesh,
            in_axis_resources=('workers', None),
            out_axis_resources=('workers',)
        )
        
        self._jit_combine = jax.jit(
            self._combine_impl,
            device_mesh=self.device_mesh,
            in_axis_resources=('workers', None),
            out_axis_resources=('workers',)
        )
        
        self.initialized = True
    
    def dispatch(self,
                input_tensor: jnp.ndarray,
                expert_assignments: jnp.ndarray,
                num_experts: int,
                num_workers: int,
                use_fp8: bool = False,
                async_mode: bool = False) -> jnp.ndarray:
        """Dispatch tokens to experts using JAX/XLA."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
            
        # Convert precision if needed
        if use_fp8:
            input_tensor = self._convert_to_fp8(input_tensor)
        
        return self._jit_dispatch(
            input_tensor,
            expert_assignments,
            num_experts,
            num_workers,
            use_fp8,
            async_mode
        )
    
    def combine(self,
               input_tensor: jnp.ndarray,
               expert_assignments: jnp.ndarray,
               original_shape: Tuple[int, int, int],
               use_fp8: bool = False,
               async_mode: bool = False) -> jnp.ndarray:
        """Combine expert outputs using JAX/XLA."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
            
        # Convert precision if needed
        if use_fp8:
            input_tensor = self._convert_to_fp8(input_tensor)
        
        return self._jit_combine(
            input_tensor,
            expert_assignments,
            original_shape,
            use_fp8,
            async_mode
        )
    
    def _dispatch_impl(self,
                      input_tensor: jnp.ndarray,
                      expert_assignments: jnp.ndarray,
                      num_experts: int,
                      num_workers: int,
                      use_fp8: bool,
                      async_mode: bool) -> jnp.ndarray:
        """Implementation of dispatch using JAX/XLA all-to-all."""
        # Reshape input for all-to-all
        batch_size, seq_len, hidden_dim = input_tensor.shape
        tokens_per_worker = (batch_size * seq_len) // num_workers
        
        # Reshape and pad if necessary
        if (batch_size * seq_len) % num_workers != 0:
            pad_size = num_workers - ((batch_size * seq_len) % num_workers)
            input_tensor = jnp.pad(
                input_tensor.reshape(-1, hidden_dim),
                ((0, pad_size), (0, 0))
            )
            expert_assignments = jnp.pad(
                expert_assignments.reshape(-1),
                (0, pad_size)
            )
        
        # Reshape for all-to-all
        input_tensor = input_tensor.reshape(num_workers, tokens_per_worker, hidden_dim)
        expert_assignments = expert_assignments.reshape(num_workers, tokens_per_worker)
        
        # Perform all-to-all communication
        with maps.mesh(self.device_mesh.devices, ('workers',)):
            # Custom call to TPU all-to-all implementation
            output = jax.pmap(
                lambda x: jax.custom_call(
                    'xen_moe_tpu_dispatch',
                    x,
                    expert_assignments,
                    num_experts=num_experts,
                    num_workers=num_workers,
                    use_fp8=use_fp8,
                    async_mode=async_mode
                )
            )(input_tensor)
        
        return output
    
    def _combine_impl(self,
                     input_tensor: jnp.ndarray,
                     expert_assignments: jnp.ndarray,
                     original_shape: Tuple[int, int, int],
                     use_fp8: bool,
                     async_mode: bool) -> jnp.ndarray:
        """Implementation of combine using JAX/XLA all-to-all."""
        batch_size, seq_len, hidden_dim = original_shape
        num_workers = self.num_workers
        tokens_per_worker = (batch_size * seq_len) // num_workers
        
        # Perform all-to-all communication
        with maps.mesh(self.device_mesh.devices, ('workers',)):
            # Custom call to TPU all-to-all implementation
            output = jax.pmap(
                lambda x: jax.custom_call(
                    'xen_moe_tpu_combine',
                    x,
                    expert_assignments,
                    original_shape=original_shape,
                    use_fp8=use_fp8,
                    async_mode=async_mode
                )
            )(input_tensor)
        
        # Reshape back to original shape
        output = output.reshape(batch_size, seq_len, hidden_dim)
        
        return output
    
    def _convert_to_fp8(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Convert tensor to FP8 using BF16 scaling."""
        # Find max absolute value for scaling
        max_abs = jnp.max(jnp.abs(tensor))
        scale = 127.0 / max_abs
        
        # Convert to BF16 first
        bf16 = jax.lax.convert_element_type(tensor, jnp.bfloat16)
        
        # Scale to FP8 range
        scaled = bf16 * scale
        
        # Convert to int8
        return jax.lax.convert_element_type(scaled, jnp.int8)
    
    def cleanup(self) -> None:
        """Clean up TPU resources."""
        if not self.initialized:
            return
            
        self.device_mesh = None
        self.initialized = False

def get_backend(device_type: str, **kwargs) -> CommBackend:
    """Get the appropriate communication backend.
    
    Args:
        device_type: Type of device ('gpu' or 'tpu')
        **kwargs: Additional initialization parameters
        
    Returns:
        Initialized communication backend
    """
    if device_type.lower() == 'gpu':
        backend = GPUCommBackend()
    elif device_type.lower() == 'tpu':
        backend = TPUCommBackend()
    else:
        raise ValueError(f"Unsupported device type: {device_type}")
    
    backend.initialize(**kwargs)
    return backend 