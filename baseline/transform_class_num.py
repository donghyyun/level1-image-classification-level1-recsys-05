import pandas as pd
import argparse
import os
import re


def trans_ans_num(x):
    mask_label = x // 9
    gender_label = (x % 9) // 3
    age_label = (x % 9) % 3

    return mask_label * 6 + (gender_label - 1) * 3 + age_label if gender_label == 2 else mask_label * 6 + gender_label * 3 + age_label

if __name__ == '__main__':  ## 안건드려도 되는 곳
    parser = argparse.ArgumentParser()
    ### 수정할 output 적어넣기
    # now_path = "/opt/ml/level1-image-classification-level1-recsys-05/baseline/output/output.csv"

    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', "/workspace/level1-image-classification-level1-recsys-05/baseline/output/output?.csv"))
    args = parser.parse_args()

    now_path = args.data_dir
    df = pd.read_csv(now_path)
    df['ans'] = df['ans'].apply(lambda x: trans_ans_num(x))

    name = now_path.split('/')[-1].split('.')[0]

    df.to_csv("changed"+name+".csv", index=False)
    print("complete!")