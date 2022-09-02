import torch
from utils import networks, experiment_manager, datasets, parsers, metrics, geofiles
from pathlib import Path


def evaluate_model(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, _ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_sar_mm = metrics.MultiThresholdMetric(thresholds, 'sar_mm')
    measurer_sar_um = metrics.MultiThresholdMetric(thresholds, 'sar_um')
    measurer_optical_mm = metrics.MultiThresholdMetric(thresholds, 'optical_mm')
    measurer_optical_um = metrics.MultiThresholdMetric(thresholds, 'optical_um')

    dataset = datasets.SpaceNet7Dataset(cfg)
    with torch.no_grad():
        for item in dataset:
            y = item['y'].to(device)
            x_sar = item['x_sar'].to(device).unsqueeze(0)
            x_optical = item['x_optical'].to(device).unsqueeze(0)

            net.set_mmtm_mode('fusion')
            out_sar_mm, out_optical_mm = net.forward(x_sar, x_optical)
            pred_sar_mm, pred_optical_mm = torch.sigmoid(out_sar_mm).detach(), torch.sigmoid(out_optical_mm).detach()
            measurer_sar_mm.add_sample(y, pred_sar_mm)
            measurer_optical_mm.add_sample(y, pred_optical_mm)

            net.set_mmtm_mode('sar')
            out_sar_um, _ = net.forward(x_sar, x_optical)
            pred_sar_um = torch.sigmoid(out_sar_um).detach()
            measurer_sar_um.add_sample(y, pred_sar_um)

            net.set_mmtm_mode('optical')
            out_optical_um, _ = net.forward(x_sar, x_optical)
            pred_optical_um = torch.sigmoid(out_optical_um).detach()
            measurer_optical_um.add_sample(y, pred_optical_um)

    data = {'config': cfg.NAME}
    for measurer in [measurer_sar_mm, measurer_sar_um, measurer_optical_mm, measurer_optical_um]:
        f1s = measurer.compute_f1()
        precisions, recalls = measurer.precision, measurer.recall

        # best f1 score for passed thresholds
        f1 = f1s.max()
        argmax_f1 = f1s.argmax()

        precision = precisions[argmax_f1]
        recall = recalls[argmax_f1]

        data[f'{measurer.name}_f1'] = f1.item()
        data[f'{measurer.name}_precision'] = precision.item()
        data[f'{measurer.name}_recall'] = recall.item()

    out_file = Path(cfg.PATHS.OUTPUT) / 'cur' / f'{cfg.NAME}.json'
    geofiles.write_json(out_file, data)

    return data


def compute_conditional_utilization_rate(a_sar_mm: float, a_sar_um: float, a_optical_mm: float, a_optical_um: float):
    pass


if __name__ == '__main__':
    args = parsers.evaluation_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    evaluate_model(cfg)
