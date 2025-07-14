class SodaError(Exception):
    """Base exception for soda"""
    pass

class ModelLoadingError(SodaError):
    """Issue during model load from hugginface"""
    pass

class ProfilingError(SodaError):
    """Issue during profiling phase"""
    pass

class TraceError(SodaError):
    """Issue parsing the trace json files"""
    pass

class AnalysisError(SodaError):
    """Issue during the analysis phase"""
    pass
