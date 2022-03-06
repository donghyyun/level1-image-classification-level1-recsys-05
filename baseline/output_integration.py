import argparse
import os

import pandas as pd

from dataset import MaskBaseDataset


def merge_with_age_model(test_dir, output_dir, base_ans, age_ans):
    """
    Input
    * base_ans (pd.DataFrame): answer from base network model
    * age_ans (pd.DataFrame): answer from age network model
    """
    output = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    preds = []

    for base_label, age_label in zip(base_ans, age_ans):
        mask_label, gender_label, _ = MaskBaseDataset.decode_multi_class(base_label)
        preds.append(MaskBaseDataset.encode_multi_class(mask_label, gender_label, age_label))
    
    output.ans = preds
    output.to_csv(os.path.join(output_dir, f'output.csv'), index=False)

if __name__ == '__main__':
    base_df = pd.read_csv(os.path.join('./output/label_smoothing/output.csv'))
    age_df = pd.read_csv(os.path.join('./output/efficient_age/output.csv'))
    test_dir = '/opt/ml/input/data/eval'
    output_dir = './output/ensemble_label_smoothing_efficient_age/'

    os.makedirs(output_dir, exist_ok=True)

    merge_with_age_model(test_dir, output_dir, base_df.ans, age_df.ans)
