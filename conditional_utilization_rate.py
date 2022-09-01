import torch
from utils import networks, experiment_manager, datasets, parsers
from pathlib import Path


def evaluate_model(cfg: experiment_manager.CfgNode, mmtm_mode: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    net.load_unimodal_features()
    net.set_mmtm_mode(mmtm_mode)

    dataset = datasets.SpaceNet7Dataset(cfg)
    with torch.no_grad():
        for index, item in enumerate(dataset):
            x_sar = item['x_sar'].to(device)
            x_optical = item['x_optical'].to(device)
            net.compute_unimodal_features(x_sar.unsqueeze(0), x_optical.unsqueeze(0))

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_mmtm_checkpoint{cfg.INFERENCE_CHECKPOINT}.pt'
    features = {}
    for name, mmtm in net.mmtm_encoder.mmtm_seq.items():
        features[f'{name}_sar'] = mmtm.average_sar_squeeze
        features[f'{name}_optical'] = mmtm.average_sar_squeeze

    torch.save(features, save_file)


if __name__ == '__main__':
    args = parsers.feature_inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    evaluate_model(cfg, 'sar')