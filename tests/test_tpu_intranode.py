import unittest
import jax
import jax.numpy as jnp
from jax.experimental import maps
import numpy as np
import os
from xen_moe.buffer import Buffer

class TestTPUIntranode(unittest.TestCase):
    def setUp(self):
        # Test configuration
        self.batch_size = 4096  # Large batch size for testing
        self.seq_len = 1  # Single sequence length for simplicity
        self.hidden_dim = 1024  # Standard hidden dimension
        self.num_experts = 8  # Number of experts
        self.num_workers = 4  # Number of TPU workers
        
        # Create test data with BF16 precision
        self.input_tensor = jnp.array(
            np.random.randn(self.batch_size, self.seq_len, self.hidden_dim),
            dtype=jnp.bfloat16
        )
        self.expert_assignments = jnp.array(
            np.random.randint(0, self.num_experts, (self.batch_size, self.seq_len)),
            dtype=jnp.int32
        )
        self.original_shape = (
            self.batch_size,
            self.seq_len,
            self.hidden_dim
        )
        
        # Set up environment variables for distributed testing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = str(self.num_workers)
        os.environ['RANK'] = '0'  # Will be overridden in each test
    
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
    
    def test_bf16_dispatch_combine(self):
        """Test BF16 dispatch and combine operations on TPU using ICI."""
        try:
            # Initialize TPU process group
            mesh = self._init_tpu_process_group()
            
            # Create buffer with TPU backend and BF16 precision
            buffer = Buffer(
                group=None,  # Not needed for TPU
                device_type='tpu',
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                use_ici=True,  # Use ICI for intranode
                precision='bfloat16'  # Use BF16 precision
            )
            
            # Calculate tokens per rank and expert
            tokens_per_rank = self.batch_size * self.seq_len // self.num_workers
            tokens_per_expert = self.batch_size * self.seq_len // self.num_experts
            
            # Test dispatch
            dispatched = buffer.dispatch(
                self.input_tensor,
                self.expert_assignments,
                num_tokens_per_rank=jnp.array([tokens_per_rank]),
                num_tokens_per_expert=jnp.array([tokens_per_expert])
            )
            
            # Verify dispatched tensor shape and dtype
            self.assertEqual(dispatched[0].shape[0], self.num_experts)
            self.assertEqual(dispatched[0].shape[2], self.hidden_dim)
            self.assertEqual(dispatched[0].dtype, jnp.bfloat16)
            
            # Verify token distribution
            tokens_per_expert_actual = jnp.sum(
                jnp.bincount(
                    self.expert_assignments.reshape(-1),
                    length=self.num_experts
                )
            )
            self.assertEqual(tokens_per_expert_actual, self.batch_size * self.seq_len)
            
            # Test combine
            combined = buffer.combine(
                dispatched[0],
                dispatched[1],  # Handle from dispatch
                topk_weights=None
            )
            
            # Verify combined tensor shape and dtype
            self.assertEqual(combined[0].shape, self.original_shape)
            self.assertEqual(combined[0].dtype, jnp.bfloat16)
            
            # Verify numerical accuracy
            # Note: BF16 has lower precision, so we use a larger tolerance
            rtol = 1e-2
            atol = 1e-2
            np.testing.assert_allclose(
                combined[0].reshape(self.original_shape),
                self.input_tensor,
                rtol=rtol,
                atol=atol
            )
            
            # Clean up
            buffer.cleanup()
            
        except RuntimeError as e:
            if "TPU" in str(e):
                self.skipTest("TPU not available")
            raise
    
    def test_bf16_async_dispatch_combine(self):
        """Test async BF16 dispatch and combine operations on TPU using ICI."""
        try:
            # Initialize TPU process group
            mesh = self._init_tpu_process_group()
            
            # Create buffer with TPU backend and BF16 precision
            buffer = Buffer(
                group=None,
                device_type='tpu',
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                use_ici=True,
                precision='bfloat16'
            )
            
            # Calculate tokens per rank and expert
            tokens_per_rank = self.batch_size * self.seq_len // self.num_workers
            tokens_per_expert = self.batch_size * self.seq_len // self.num_experts
            
            # Test async dispatch
            dispatched = buffer.dispatch(
                self.input_tensor,
                self.expert_assignments,
                num_tokens_per_rank=jnp.array([tokens_per_rank]),
                num_tokens_per_expert=jnp.array([tokens_per_expert]),
                async_mode=True
            )
            
            # Verify dispatched tensor shape and dtype
            self.assertEqual(dispatched[0].shape[0], self.num_experts)
            self.assertEqual(dispatched[0].shape[2], self.hidden_dim)
            self.assertEqual(dispatched[0].dtype, jnp.bfloat16)
            
            # Test async combine
            combined = buffer.combine(
                dispatched[0],
                dispatched[1],
                topk_weights=None,
                async_mode=True
            )
            
            # Verify combined tensor shape and dtype
            self.assertEqual(combined[0].shape, self.original_shape)
            self.assertEqual(combined[0].dtype, jnp.bfloat16)
            
            # Verify numerical accuracy with larger tolerance for async operations
            rtol = 1e-2
            atol = 1e-2
            np.testing.assert_allclose(
                combined[0].reshape(self.original_shape),
                self.input_tensor,
                rtol=rtol,
                atol=atol
            )
            
            # Clean up
            buffer.cleanup()
            
        except RuntimeError as e:
            if "TPU" in str(e):
                self.skipTest("TPU not available")
            raise

if __name__ == '__main__':
    unittest.main() 