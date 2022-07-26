import torch
from utils import networks, experiment_manager, datasets, parsers
from torch.utils import data as torch_data
from pathlib import Path


def compute_features(cfg: experiment_manager.CfgNode, run_type: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    dataset = datasets.UrbanExtractionDataset(cfg, dataset=run_type, no_augmentations=True)
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': False,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x_sar = batch['x_sar'].to(device)
            x_optical = batch['x_optical'].to(device)
            net.compute_unimodal_features(x_sar, x_optical)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_mmtm_checkpoint{cfg.INFERENCE_CHECKPOINT}.pt'
    features = {}
    for name, mmtm in net.mmtm_encoder.mmtm_seq.items():
        features[f'{name}_sar'] = mmtm.average_sar_squeeze
        features[f'{name}_optical'] = mmtm.average_sar_squeeze

    torch.save(features, save_file)


if __name__ == '__main__':
    args = parsers.feature_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    compute_features(cfg, 'training')

