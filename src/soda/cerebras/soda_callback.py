"""
SODA Callback for Cerebras ModelZoo Trainer.
Hooks into the Trainer's lifecycle to intercept execution metrics seamlessly.
"""
import logging
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
    the active training loop without altering ModelZoo code.
    """
    def __init__(self):
        super().__init__()
        self.tracer = None
        self.rate_tracker = None

    def on_enter_fit(self, trainer, stack, train_dataloader, val_dataloader, loop):
        """
        Called when the fit method is entered. We initialize our CerebrasTracer here.
        """
        class MockArgs:
            def __init__(self, output_dir, model_name):
                self.output_dir = output_dir
                self.model = model_name
                
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
        
        LOGGER.info(f"Initialized SodaCallback for Cerebras ModelZoo. Output dir: {output_dir}")

    @cstorch.step_closure
    def _log_metrics_step(self, current_step, batch_size):
        if self.rate_tracker:
            if batch_size:
                self.rate_tracker.add(batch_size)
            rate = self.rate_tracker.rate()
            global_rate = self.rate_tracker.global_rate()
        else:
            rate, global_rate = 0.0, 0.0
            
        if self.tracer:
            self.tracer.log_step_metrics(current_step, 0.0, rate, global_rate)

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        """
        Hook the step closure logic here. 
        Note: The actual closure logic executes asynchronously when the WSE returns.
        """
        if cstorch is None or not self.tracer:
            return
            
        if trainer.is_first_iteration and self.rate_tracker:
            self.rate_tracker.reset()

        # Infer batch size safely from batch dict/tensor
        batch_size = 16
        if isinstance(batch, dict):
            for v in batch.values():
                if hasattr(v, "shape") and len(v.shape) > 0:
                    batch_size = v.shape[0]
                    break
        elif hasattr(batch, "shape") and len(batch.shape) > 0:
            batch_size = batch.shape[0]

        self._log_metrics_step(batch_idx, batch_size)

    def on_fit_end(self, trainer, loop):
        """
        Export SODA trace when training finishes.
        """
        if self.tracer:
            self.tracer.export_metrics()
            LOGGER.info("SODA hardware metrics exported successfully.")
