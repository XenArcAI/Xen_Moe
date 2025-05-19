import unittest
import torch
import jax
import jax.numpy as jnp
import numpy as np
import os
from xen_moe.backend import get_backend, CommBackend
from xen_moe.buffer import Buffer

class TestCommBackend(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 8
        self.hidden_dim = 16
        self.num_experts = 4
        self.num_workers = 2
        
        # Create test data
        self.input_tensor = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim
        )
        self.expert_assignments = torch.randint(
            0,
            self.num_experts,
            (self.batch_size, self.seq_len)
        )
        self.original_shape = (
            self.batch_size,
            self.seq_len,
            self.hidden_dim
        )
        
        # Create JAX test data
        self.input_tensor_jax = jnp.array(self.input_tensor.numpy())
        self.expert_assignments_jax = jnp.array(self.expert_assignments.numpy())
        
        # Set up environment variables for distributed testing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = str(self.num_workers)
        os.environ['RANK'] = '0'  # Will be overridden in each test
    
    def _init_gpu_process_group(self):
        """Initialize PyTorch distributed process group for GPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Initialize process group
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.num_workers,
            rank=int(os.environ['RANK'])
        )
        
        # Create communication group
        group = torch.distributed.new_group()
        
        return group
    
    def _init_tpu_process_group(self):
        """Initialize JAX distributed process group for TPU."""
        if not jax.devices():
            self.skipTest("TPU not available")
            
        # Initialize JAX distributed
        jax.distributed.initialize()
        
        # Create device mesh
        devices = jax.devices()
        mesh = jax.experimental.maps.Mesh(
            devices,
            axis_names=('workers',)
        )
        
        return mesh
    
    def test_gpu_backend(self):
        """Test GPU communication backend with NVLink."""
        try:
            # Initialize GPU process group
            group = self._init_gpu_process_group()
            
            # Create buffer with GPU backend
            buffer = Buffer(
                group=group,
                device_type='gpu',
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                low_latency_mode=False  # Use NVLink for intranode
            )
            
            # Test dispatch
            dispatched = buffer.dispatch(
                self.input_tensor,
                self.expert_assignments,
                num_tokens_per_rank=torch.tensor([self.batch_size * self.seq_len // self.num_workers]),
                num_tokens_per_expert=torch.tensor([self.batch_size * self.seq_len // self.num_experts])
            )
            
            self.assertEqual(
                dispatched[0].shape[0],  # First element of tuple is the dispatched tensor
                self.num_experts
            )
            self.assertEqual(
                dispatched[0].shape[2],
                self.hidden_dim
            )
            
            # Test combine
            combined = buffer.combine(
                dispatched[0],
                dispatched[1],  # Handle from dispatch
                topk_weights=None
            )
            
            self.assertEqual(combined[0].shape, self.original_shape)
            
            # Clean up
            buffer.cleanup()
            torch.distributed.destroy_process_group()
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                self.skipTest("CUDA not available")
            raise
    
    def test_tpu_backend(self):
        """Test TPU communication backend with ICI."""
        try:
            # Initialize TPU process group
            mesh = self._init_tpu_process_group()
            
            # Create buffer with TPU backend
            buffer = Buffer(
                group=None,  # Not needed for TPU
                device_type='tpu',
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                use_ici=True  # Use ICI for intranode
            )
            
            # Test dispatch
            dispatched = buffer.dispatch(
                self.input_tensor_jax,
                self.expert_assignments_jax,
                num_tokens_per_rank=jnp.array([self.batch_size * self.seq_len // self.num_workers]),
                num_tokens_per_expert=jnp.array([self.batch_size * self.seq_len // self.num_experts])
            )
            
            self.assertEqual(
                dispatched[0].shape[0],  # First element of tuple is the dispatched tensor
                self.num_experts
            )
            self.assertEqual(
                dispatched[0].shape[2],
                self.hidden_dim
            )
            
            # Test combine
            combined = buffer.combine(
                dispatched[0],
                dispatched[1],  # Handle from dispatch
                topk_weights=None
            )
            
            self.assertEqual(combined[0].shape, self.original_shape)
            
            # Clean up
            buffer.cleanup()
            
        except RuntimeError as e:
            if "TPU" in str(e):
                self.skipTest("TPU not available")
            raise
    
    def test_invalid_device_type(self):
        """Test invalid device type."""
        with self.assertRaises(ValueError):
            get_backend('invalid_device')
    
    def test_uninitialized_backend(self):
        """Test using uninitialized backend."""
        backend = CommBackend()
        
        with self.assertRaises(NotImplementedError):
            backend.dispatch(
                self.input_tensor,
                self.expert_assignments,
                self.num_experts,
                self.num_workers
            )
        
        with self.assertRaises(NotImplementedError):
            backend.combine(
                self.input_tensor,
                self.expert_assignments,
                self.original_shape
            )
        
        with self.assertRaises(NotImplementedError):
            backend.initialize()
        
        with self.assertRaises(NotImplementedError):
            backend.cleanup()

if __name__ == '__main__':
    unittest.main() 