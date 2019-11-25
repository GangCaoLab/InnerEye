from twintail.utils.log import set_logging_fmt

class Tools:
    def __init__(self, log_level=20, log_file=None):
        set_logging_fmt(
            log_level=log_level,
            log_file=log_file
        )
        from .gen import gen
        self.gen = gen
        from .extract_img import extract_samples
        self.extract_samples = extract_samples
        from .registration import registration
        self.registration = registration


import fire
fire.Fire(Tools)
