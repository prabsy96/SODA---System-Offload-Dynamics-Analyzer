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
        # We will initialize the tracer when the fit starts, so we can access trainer config.
        self.tracer = None

    def on_enter_fit(self, trainer, stack, train_dataloader, val_dataloader, loop):
        """
        Called when the fit method is entered. We initialize our CerebrasTracer here.
        """
        # We can extract the model_dir or output_dir from trainer
        class MockArgs:
            def __init__(self, output_dir, model_name):
                self.output_dir = output_dir
                self.model = model_name
                
        output_dir = trainer.model_dir / "soda_output"
        model_name = getattr(trainer, "model_name", "unknown_model")
        
        args = MockArgs(output_dir=str(output_dir), model_name=model_name)
        self.tracer = CerebrasTracer(args)
        
        LOGGER.info(f"Initialized SodaCallback for Cerebras ModelZoo. Output dir: {output_dir}")

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        """
        Hook the step closure logic here. 
        Note: The actual closure logic executes asynchronously when the WSE returns.
        """
        if cstorch is None or not self.tracer:
            return
            
        @cstorch.step_closure
        def log_soda_metrics(current_step):
            # Access the backend executor directly to pull the rates
            executor = cstorch.backend().executor
            if executor and hasattr(executor, 'profiler'):
                rate = executor.profiler.rate()
                global_rate = executor.profiler.global_rate()
                # Dummy loss extraction since step_closure might not have access to loss here trivially
                loss_val = 0.0 
                self.tracer.log_step_metrics(current_step, loss_val, rate, global_rate)

        # We pass the current step into the closure
        log_soda_metrics(batch_idx)

    def on_fit_end(self, trainer, loop):
        """
        Export SODA trace when training finishes.
        """
        if self.tracer:
            self.tracer.export_metrics()
            LOGGER.info("SODA hardware metrics exported successfully.")
