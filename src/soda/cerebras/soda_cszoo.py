"""
SODA Wrapper for Cerebras ModelZoo CLI (cszoo).
Injects the SODA Hardware Profiling Callback into any cszoo run.
"""
import sys
import logging

try:
    from cerebras.modelzoo.cli.main import ModelZooCLI
    from cerebras.modelzoo.trainer.trainer import Trainer
except ImportError:
    print("Error: cerebras.modelzoo not found. Ensure you are in the Cerebras environment.")
    sys.exit(1)

from soda.cerebras.soda_callback import SodaCallback

LOGGER = logging.getLogger("soda.cerebras.cszoo")
logging.basicConfig(level=logging.INFO)

def _patch_trainer():
    """
    Monkey-patch the ModelZoo Trainer to automatically inject the SodaCallback.
    This avoids requiring the user to modify their YAML configs.
    """
    _orig_init = Trainer.__init__

    def _soda_init(self, *args, **kwargs):
        # Initialize original trainer
        _orig_init(self, *args, **kwargs)
        # Inject our tracer callback
        soda_cb = SodaCallback()
        if isinstance(self.callbacks, list):
            self.callbacks.append(soda_cb)
        else:
            self.callbacks["SodaCallback"] = soda_cb
        LOGGER.info("SODA Callback injected into Cerebras Trainer.")

    Trainer.__init__ = _soda_init

def main():
    print("=" * 60)
    print(" SODA Hardware Profiling Wrapper for Cerebras (cszoo) ")
    print("=" * 60)
    
    # Patch the Trainer
    _patch_trainer()
    
    # Run the standard ModelZoo CLI
    # It will parse sys.argv and execute the action inside __init__
    ModelZooCLI()

if __name__ == "__main__":
    main()
