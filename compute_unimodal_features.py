import torch
from utils import networks, experiment_manager, datasets, parsers, metrics, geofiles
from tqdm import tqdm
from pathlib import Path
import numpy as np


def compute_features(cfg: experiment_manager.CfgNode, run_type: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.UrbanExtractionDataset(cfg, dataset=run_type, no_augmentations=True)

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            batch = dataset.__getitem__(index)
            x_sar = batch['x_sar'].to(device)
            x_optical = batch['x_optical'].to(device)
            y_gts = batch['y'].to(device)

            sar_logits, optical_logits = net(x_sar, x_optical)


if __name__ == '__main__':
    args = parsers.testing_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    compute_features(cfg, 'training')

