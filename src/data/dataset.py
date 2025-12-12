from pathlib import Path
from torch.utils.data import Dataset

import pandas as pd
import numpy as np



class OrionAEFrameDataset(Dataset):

    def __init__(
        self,
        data_path: Path,
    ):
        self.data_path = data_path
        
        # check if metadata.csv exists
        if not (self.data_path / 'metadata.csv').exists():
            raise FileNotFoundError(f'Metadata file not found at {self.data_path / 'metadata.csv'}')
        
        # load metadata
        self.metadata = pd.read_csv(self.data_path / 'metadata.csv')

    def __getitem__(self, index):
        pass





    