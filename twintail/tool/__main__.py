from twintail.utils.log import set_global_logging

class Tools:
    def __init__(self, log_level=20, log_file=None):
        set_global_logging(
            log_level=log_level,
            log_file=log_file
        )
        from .gen import gen
        self.gen = gen
        from .extract_img import extract_samples
        self.extract_samples = extract_samples
        from .registration import registration
        self.registration = registration
        from .call_spots import SignalCall
        self.signal_call = SignalCall
        from .preprocessing import PreProcessing
        self.pre_proc = PreProcessing


import fire
fire.Fire(Tools)
