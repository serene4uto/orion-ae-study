# we will have preprocessing pipeline with fixed order that go from filters group then norms group
from typing import Callable
import numpy as np



class PreprocessingPipeline:

    def __init__(self, filters: list[Callable], norms: list[Callable]):
        pass


    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass

# class 