
import pytest
import argparse
import torch
from unittest.mock import MagicMock, patch
from soda import ModelTracer

class TestParallelismLogic:
    """
    Tests for multi-GPU parallelism logic in ModelTracer.
    These are purely logic tests that mock the CUDA environment.
    """

    def _make_args(self, num_gpus=1, parallelism="tp"):
        args = argparse.Namespace()
        args.model = "gpt2"
        args.device = "cuda"
        args.num_gpus = num_gpus
        args.parallelism = parallelism
        args.compile_type = "eager"
        args.precision = "bfloat16"
        args.seed = 42
        args.batch_size = 1
        args.seq_len = 128
        args.max_new_tokens = 1
        args.warmup = 0
        args.output_dir = MagicMock()
        args.microbench = False
        return args

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=4)
    @patch('torch.device')
    @patch('torch.cuda.manual_seed_all')
    def test_tp_device_map(self, mock_seed, mock_device, mock_count, mock_avail):
        mock_dev_obj = MagicMock()
        mock_device.return_value = mock_dev_obj
        
        args = self._make_args(num_gpus=2, parallelism="tp")
        tracer = ModelTracer(args)
        kwargs = tracer.get_kwargs()
        assert kwargs["device_map"] == "balanced"
        assert tracer.parallelism == "tp"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=4)
    @patch('torch.device')
    @patch('torch.cuda.manual_seed_all')
    def test_dp_device_map(self, mock_seed, mock_device, mock_count, mock_avail):
        mock_dev_obj = MagicMock()
        mock_device.return_value = mock_dev_obj

        args = self._make_args(num_gpus=2, parallelism="dp")
        tracer = ModelTracer(args)
        kwargs = tracer.get_kwargs()
        assert isinstance(kwargs["device_map"], dict)
        assert kwargs["device_map"][""] == mock_dev_obj
        assert tracer.parallelism == "dp"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=4)
    @patch('torch.device')
    @patch('torch.cuda.manual_seed_all')
    @patch('torch.nn.DataParallel')
    def test_dp_wrapping(self, mock_dp, mock_seed, mock_device, mock_count, mock_avail):
        # Prevent any CUDA calls during setup
        with patch('torch.cuda.synchronize'), \
             patch('torch.cuda.memory_allocated', return_value=0), \
             patch('torch.cuda.memory_reserved', return_value=0), \
             patch('torch.cuda.reset_peak_memory_stats'):
            
            args = self._make_args(num_gpus=2, parallelism="dp")
            tracer = ModelTracer(args)
            
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            tracer.load_decoder = MagicMock(return_value=(mock_model, MagicMock()))
            
            tracer.setup()
            mock_dp.assert_called_once_with(mock_model)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=4)
    @patch('torch.device')
    @patch('torch.cuda.manual_seed_all')
    @patch('torch.distributed.is_initialized', return_value=True)
    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_fsdp_wrapping(self, mock_fsdp, mock_init, mock_dist, mock_seed, mock_device, mock_count, mock_avail):
        with patch('torch.cuda.synchronize'), \
             patch('torch.cuda.memory_allocated', return_value=0), \
             patch('torch.cuda.memory_reserved', return_value=0), \
             patch('torch.cuda.reset_peak_memory_stats'):

            args = self._make_args(num_gpus=2, parallelism="fsdp")
            tracer = ModelTracer(args)
            
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            tracer.load_decoder = MagicMock(return_value=(mock_model, MagicMock()))
            
            tracer.setup()
            mock_fsdp.assert_called_once()
