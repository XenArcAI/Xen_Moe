import unittest
import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from xen_moe.buffer import Buffer

@dataclass
class BufferConfig:
    """Configuration for buffer auto-tuning."""
    batch_size: int
    hidden_dim: int
    num_experts: int
    num_workers: int
    buffer_size: int
    precision: str
    device_type: str
    interconnect: str
    async_mode: bool = False

@dataclass
class TuningResult:
    """Results from buffer auto-tuning."""
    config: BufferConfig
    latency: float
    throughput: float
    memory_usage: float

class TestBufferAutoTune(unittest.TestCase):
    def setUp(self):
        # Common test configuration
        self.batch_sizes = [1024, 2048, 4096, 8192]
        self.hidden_dims = [512, 1024, 2048]
        self.num_experts = 8
        self.num_workers = 4
        
        # Buffer size configurations (in MB)
        self.buffer_sizes = [64, 128, 256, 512, 1024]
        
        # Set up environment variables for distributed testing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = str(self.num_workers)
        os.environ['RANK'] = '0'
        
        # Initialize device-specific configurations
        self.device_configs = self._get_device_configs()
    
    def _get_device_configs(self) -> List[BufferConfig]:
        """Get device-specific configurations for auto-tuning."""
        configs = []
        
        # TPU configurations
        if jax.devices():
            # ICI (intranode) configurations
            configs.extend([
                BufferConfig(
                    batch_size=bs,
                    hidden_dim=hd,
                    num_experts=self.num_experts,
                    num_workers=self.num_workers,
                    buffer_size=bs,
                    precision='bfloat16',
                    device_type='tpu',
                    interconnect='ici',
                    async_mode=async_mode
                )
                for bs in self.batch_sizes
                for hd in self.hidden_dims
                for async_mode in [True, False]
            ])
            
            # DCI (internode) configurations
            configs.extend([
                BufferConfig(
                    batch_size=bs,
                    hidden_dim=hd,
                    num_experts=self.num_experts,
                    num_workers=self.num_workers,
                    buffer_size=bs,
                    precision='bfloat16',
                    device_type='tpu',
                    interconnect='dci',
                    async_mode=async_mode
                )
                for bs in self.batch_sizes
                for hd in self.hidden_dims
                for async_mode in [True, False]
            ])
        
        # GPU configurations
        if torch.cuda.is_available():
            # NVLink configurations
            configs.extend([
                BufferConfig(
                    batch_size=bs,
                    hidden_dim=hd,
                    num_experts=self.num_experts,
                    num_workers=self.num_workers,
                    buffer_size=bs,
                    precision='float16',
                    device_type='gpu',
                    interconnect='nvlink',
                    async_mode=async_mode
                )
                for bs in self.batch_sizes
                for hd in self.hidden_dims
                for async_mode in [True, False]
            ])
            
            # RDMA configurations
            configs.extend([
                BufferConfig(
                    batch_size=bs,
                    hidden_dim=hd,
                    num_experts=self.num_experts,
                    num_workers=self.num_workers,
                    buffer_size=bs,
                    precision='float16',
                    device_type='gpu',
                    interconnect='rdma',
                    async_mode=async_mode
                )
                for bs in self.batch_sizes
                for hd in self.hidden_dims
                for async_mode in [True, False]
            ])
        
        return configs
    
    def _init_device(self, config: BufferConfig) -> Tuple[Optional[torch.distributed.ProcessGroup], Optional[jax.experimental.maps.Mesh]]:
        """Initialize device-specific process group."""
        if config.device_type == 'gpu':
            if not torch.cuda.is_available():
                return None, None
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=config.num_workers,
                rank=int(os.environ['RANK'])
            )
            return torch.distributed.new_group(), None
        else:  # TPU
            if not jax.devices():
                return None, None
            jax.distributed.initialize()
            devices = jax.devices()
            mesh = jax.experimental.maps.Mesh(
                devices,
                axis_names=('workers',)
            )
            return None, mesh
    
    def _create_test_data(self, config: BufferConfig):
        """Create test data based on configuration."""
        if config.device_type == 'gpu':
            input_tensor = torch.randn(
                config.batch_size,
                1,  # seq_len
                config.hidden_dim,
                dtype=torch.float16 if config.precision == 'float16' else torch.float32
            ).cuda()
            expert_assignments = torch.randint(
                0,
                config.num_experts,
                (config.batch_size, 1),
                dtype=torch.int32
            ).cuda()
            return input_tensor, expert_assignments
        else:  # TPU
            input_tensor = jnp.array(
                np.random.randn(config.batch_size, 1, config.hidden_dim),
                dtype=jnp.bfloat16 if config.precision == 'bfloat16' else jnp.float32
            )
            expert_assignments = jnp.array(
                np.random.randint(0, config.num_experts, (config.batch_size, 1)),
                dtype=jnp.int32
            )
            return input_tensor, expert_assignments
    
    def _measure_performance(self, config: BufferConfig, buffer: Buffer, input_tensor, expert_assignments) -> TuningResult:
        """Measure performance metrics for a given configuration."""
        num_iterations = 10
        latencies = []
        memory_usage = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            # Dispatch
            dispatched = buffer.dispatch(
                input_tensor,
                expert_assignments,
                num_tokens_per_rank=config.batch_size // config.num_workers,
                num_tokens_per_expert=config.batch_size // config.num_experts,
                async_mode=config.async_mode
            )
            
            # Combine
            combined = buffer.combine(
                dispatched[0],
                dispatched[1],
                topk_weights=None,
                async_mode=config.async_mode
            )
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            # Measure memory usage
            if config.device_type == 'gpu':
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            else:
                # For TPU, we can't directly measure memory usage
                memory_usage.append(0.0)
        
        avg_latency = sum(latencies) / len(latencies)
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0.0
        
        # Calculate throughput (tokens per second)
        throughput = config.batch_size / avg_latency
        
        return TuningResult(
            config=config,
            latency=avg_latency,
            throughput=throughput,
            memory_usage=avg_memory
        )
    
    def test_auto_tune(self):
        """Test auto-tuning of communication buffers."""
        results: List[TuningResult] = []
        
        for config in self.device_configs:
            try:
                # Initialize device
                group, mesh = self._init_device(config)
                if group is None and mesh is None:
                    continue
                
                # Create buffer
                buffer = Buffer(
                    group=group,
                    device_type=config.device_type,
                    batch_size=config.batch_size,
                    hidden_dim=config.hidden_dim,
                    num_experts=config.num_experts,
                    use_ici=(config.interconnect == 'ici'),
                    precision=config.precision
                )
                
                # Create test data
                input_tensor, expert_assignments = self._create_test_data(config)
                
                # Measure performance
                result = self._measure_performance(
                    config,
                    buffer,
                    input_tensor,
                    expert_assignments
                )
                results.append(result)
                
                # Clean up
                buffer.cleanup()
                if config.device_type == 'gpu':
                    torch.distributed.destroy_process_group()
                
            except Exception as e:
                print(f"Error testing configuration {config}: {str(e)}")
                continue
        
        # Find best configuration for each device type and interconnect
        best_configs: Dict[str, TuningResult] = {}
        for result in results:
            key = f"{result.config.device_type}_{result.config.interconnect}"
            if key not in best_configs or result.latency < best_configs[key].latency:
                best_configs[key] = result
        
        # Print results
        print("\nBest configurations by device type and interconnect:")
        for key, result in best_configs.items():
            print(f"\n{key.upper()}:")
            print(f"  Batch size: {result.config.batch_size}")
            print(f"  Hidden dim: {result.config.hidden_dim}")
            print(f"  Buffer size: {result.config.buffer_size}MB")
            print(f"  Async mode: {result.config.async_mode}")
            print(f"  Latency: {result.latency:.3f}s")
            print(f"  Throughput: {result.throughput:.2f} tokens/s")
            print(f"  Memory usage: {result.memory_usage:.2f}MB")
        
        # Verify that we have at least one valid configuration
        self.assertGreater(len(results), 0, "No valid configurations were tested")

if __name__ == '__main__':
    unittest.main() 