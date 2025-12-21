
from pathlib import Path
import sys

# More reliable: get the project root from the scripts's location
ROOT_PATH = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd().parent
# Or even better for notebooks:
ROOT_PATH = Path().resolve().parent  # Goes up from notebooks/ folder

sys.path.append(str(ROOT_PATH))

import argparse
import yaml

from src.data.dataset import OrionAEFrameDataset
from src.data.transforms import preprocessing
from src.data.transforms import (
    PreprocessingPipeline, NormPipeline, FilterPipeline, MiscPipeline,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        '--frame-path',
        type=Path,
        default="",
        help='Path to directory containing frame dataset'
    )
    parser.add_argument(
        '--preprocess-config-path',
        type=Path,
        default='',
        help='Path to .yaml preprocess config file for the dataset'
    )
    parser.add_argument(
        '--feature-config-path',
        type=Path,
        default='',
        help='Path to .yaml feature config file for the dataset'
    )
    parser.add_argument(
        '--save-path',
        type=Path,
        default="",
        help='Path to save processed dataset'
    )

    return parser


def process_preprocess_cfg(preprocess_cfg: dict):

    filters = []
    norms = []
    miscs = []

    for filter in preprocess_cfg.get('filters', []):
        filter_name = filter.get('name')
        if filter_name is None:
            raise ValueError("Filter configuration missing 'name' field")
        filter_params = filter.get('params', {})
        filters.append(getattr(preprocessing, filter_name)(**filter_params))

    for norm in preprocess_cfg.get('norms', []):
        norm_name = norm.get('name')
        if norm_name is None:
            raise ValueError("Norm configuration missing 'name' field")
        norm_params = norm.get('params', {})
        norms.append(getattr(preprocessing, norm_name)(**norm_params))

    for misc in preprocess_cfg.get('miscs', []):
        misc_name = misc.get('name')
        if misc_name is None:
            raise ValueError("Misc configuration missing 'name' field")
        misc_params = misc.get('params', {})
        miscs.append(getattr(preprocessing, misc_name)(**misc_params))

    return PreprocessingPipeline(
        filters=FilterPipeline(filters),
        norms=NormPipeline(norms),
        miscs=MiscPipeline(miscs)
    )

def process_feature_cfg(feature_cfg: dict):
    pass


def main():
    args = parse_args()

    preprocess_cfg_path = args.preprocess_config_path
    feature_cfg_path =  args.feature_config_path

    # load yaml config
    with open(preprocess_cfg_path, "r") as f:
        preprocess_cfg = yaml.safe_load(f)

    with open(feature_cfg_path, "r") as f:
        feature_cfg =  yaml.safe_load(f)
    
    # process pipeline cfg
    process_pipeline = process_preprocess_cfg(
        preprocess_cfg=preprocess_cfg
    )



    



    

if __name__ == "__main__":
    pass


