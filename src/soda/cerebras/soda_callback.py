"""
SODA Callback for Cerebras ModelZoo Trainer.
Hooks into the Trainer's lifecycle to intercept execution metrics seamlessly.
"""
import logging
import time
from pathlib import Path

try:
    import cerebras.pytorch as cstorch
    from cerebras.modelzoo.trainer.callbacks import Callback
except ImportError:
    # Fallbacks for non-Cerebras environments if parsed blindly
    Callback = object
    cstorch = None

from soda.cerebras.tracer import CerebrasTracer

LOGGER = logging.getLogger("soda.cerebras.callback")

class SodaCallback(Callback):
    """
    A ModelZoo Callback that injects the SODA step metrics profiler into 
    the active training/validate loop without altering ModelZoo code.
    """
    def __init__(self):
        super().__init__()
        self.tracer = None
        self.rate_tracker = None

    def _setup_tracer(self, trainer, loop_type="fit"):
        class MockArgs:
            def __init__(self, output_dir, model_name):
                self.output_dir = output_dir
                self.model = model_name
                
        output_dir = trainer.model_dir / "soda_output"
        model_name = "unknown_model"
        if hasattr(trainer, "model") and trainer.model is not None:
            if hasattr(trainer.model, "__class__"):
                model_name = trainer.model.__class__.__name__
            elif callable(trainer.model):
                model_name = getattr(trainer.model, "__name__", str(trainer.model))
        elif hasattr(trainer, "model_name"):
            model_name = trainer.model_name
            
        args = MockArgs(output_dir=str(output_dir), model_name=model_name)
        self.tracer = CerebrasTracer(args)
        if cstorch:
            self.rate_tracker = cstorch.utils.tracker.RateTracker()
        
        LOGGER.info(f"Initialized SodaCallback for Cerebras ModelZoo ({loop_type}). Output dir: {output_dir}")

    def on_enter_fit(self, trainer, stack, train_dataloader, val_dataloader, loop):
        self._setup_tracer(trainer, "fit")

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        self._setup_tracer(trainer, "validate")

    @cstorch.step_closure
    def _log_metrics_step(self, current_step, batch_size, start_time):
        end_time = time.time()
        host_turnaround_s = end_time - start_time if start_time else 0.0
        
        if self.rate_tracker:
            if batch_size:
                self.rate_tracker.add(batch_size)
            rate = self.rate_tracker.rate()
            global_rate = self.rate_tracker.global_rate()
        else:
            rate, global_rate = 0.0, 0.0
            
        if self.tracer:
            self.tracer.log_step_metrics(current_step, 0.0, rate, global_rate, host_turnaround_s)

    def _handle_batch_start(self, trainer, batch, batch_idx):
        if cstorch is None or not self.tracer:
            return
            
        if getattr(trainer, "is_first_iteration", False) and self.rate_tracker:
            self.rate_tracker.reset()

        batch_size = 16
        if isinstance(batch, dict):
            for v in batch.values():
                if hasattr(v, "shape") and len(v.shape) > 0:
                    batch_size = v.shape[0]
                    break
        elif hasattr(batch, "shape") and len(batch.shape) > 0:
            batch_size = batch.shape[0]

        start_time = time.time()
        self._log_metrics_step(batch_idx, batch_size, start_time)

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self._handle_batch_start(trainer, batch, batch_idx)

    def on_validate_batch_start(self, trainer, model, batch, batch_idx):
        self._handle_batch_start(trainer, batch, batch_idx)

    def on_fit_end(self, trainer, loop):
        if self.tracer:
            self.tracer.export_metrics()
            LOGGER.info("SODA hardware metrics exported successfully (fit).")

    def on_validate_end(self, trainer, loop):
        if self.tracer:
            self.tracer.export_metrics()
            LOGGER.info("SODA hardware metrics exported successfully (validate).")

