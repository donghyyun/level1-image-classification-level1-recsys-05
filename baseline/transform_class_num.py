import pandas as pd
import os
import re

def trans_age(q, r):
    if r == 0:
        return 3*q
    elif r == 1 or r == 2:
        return 3*q + 1
    else:
        return 3*q + 2

def trans_class_num(x):
    q, r = divmod(x, 4)
    if q == 0:
        return trans_age(q, r)
    elif q == 1:
        return trans_age(q, r)
    elif q == 2:
        return trans_age(q, r)
    elif q == 3:
        return trans_age(q, r)
    elif q == 4:
        return trans_age(q, r)
    elif q == 5:
        return trans_age(q, r)

if __name__ == '__main__':
    now_path = "/opt/ml/level1-image-classification-level1-recsys-05/baseline/output"

    last_output_dir = os.listdir('./output')
    last_output_dir.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    last_model_dir = os.listdir('./model')
    last_model_dir.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    result = now_path + "/" + last_model_dir[-1] + "_" + last_output_dir[-1]
    output_path = os.path.join(now_path, last_output_dir[-1])

    df = pd.read_csv(output_path)
    df['ans'] = df['ans'].apply(lambda x: trans_class_num(x))

    df.to_csv(result, index=False)
    print("complete!")