import argparse
import os
import multiprocessing
from importlib import import_module

import pandas as pd
import numpy as np
from tqdm import tqdm

import ttach
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TestDataset

def load_model(model_dir, model_name, num_classes=18, device=torch.device("cuda")):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(model_dir, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(args, output_dir):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18

    img_root = os.path.join(args.data_dir, 'images')
    info_path = os.path.join(args.data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        # num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    transforms = ttach.Compose(
        [
            ttach.HorizontalFlip(),
            # ttach.VerticalFlip()
        ]
    )
    models = [load_model(dir, name, device=device).to(device) for dir, name in zip(args.model_dirs, args.models)]
    for model in models:
        model.eval()

    tta_models = [ttach.ClassificationTTAWrapper(model, transforms) for model in models]
    
    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in tqdm(enumerate(loader)):
            images = images.to(device)
            logits = torch.tensor([model(images).cpu().numpy() for model in tta_models])
            pred = torch.mean(logits, 0)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=300, help='input batch size for validing (default: 100)')
    parser.add_argument('--resize', type=tuple, default=(512, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--models', type=list, default=[
        'EfficientNet', 
        'EfficientNet', 
        'EfficientNet',
        # 'EfficientNet',
        # 'EfficientNet'
        ], help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/eval')
    parser.add_argument('--model_dirs', type=list, default=[
        # './model/efficientNet_age_adjust_18',
        './model/efficientNet_best',
        './model/efficientNet_weight_loss',
        # './model/efficientNet_focal',
        './model/efficinetNet_pseudo_labeling_f1_valratio_04_best'
    ])
    parser.add_argument('--output_name', type=str, default='model1_model2')

    args = parser.parse_args()

    output_dir = os.path.join("./output/ensemble", args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    inference(args, output_dir)
