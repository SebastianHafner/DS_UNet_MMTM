import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from utils import datasets, metrics, experiment_manager


def model_evaluation(net, cfg: experiment_manager.CfgNode, device: str, run_type: str, epoch: float, step: int,
                     max_samples: int = None):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_sar = metrics.MultiThresholdMetric(thresholds)
    measurer_optical = metrics.MultiThresholdMetric(thresholds)
    measurer_fusion = metrics.MultiThresholdMetric(thresholds)

    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=True)

    stop_step = len(dataloader) if max_samples is None else max_samples

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step == stop_step:
                break

            x_sar = batch['x_sar'].to(device)
            x_optical = batch['x_optical'].to(device)
            y_true = batch['y'].to(device)

            logits_sar, logits_optical = net(x_sar, x_optical)
            y_pred_sar, y_pred_optical = torch.sigmoid(logits_sar), torch.sigmoid(logits_optical)

            y_true = y_true.detach()
            y_pred_sar, y_pred_optical = y_pred_sar.detach(), y_pred_optical.detach()
            measurer_sar.add_sample(y_true, y_pred_sar)
            measurer_optical.add_sample(y_true, y_pred_optical)
            y_pred_fusion = (y_pred_sar + y_pred_optical) / 2
            measurer_fusion.add_sample(y_true, y_pred_fusion)

            if cfg.DEBUG:
                break

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    for measurer, name in zip([measurer_sar, measurer_optical, measurer_fusion], ['sar', 'optical', 'fusion']):
        f1s = measurer.compute_f1()
        precisions, recalls = measurer.precision, measurer.recall

        # best f1 score for passed thresholds
        f1 = f1s.max()
        argmax_f1 = f1s.argmax()

        precision = precisions[argmax_f1]
        recall = recalls[argmax_f1]

        wandb.log({f'{run_type} {name} F1': f1,
                   f'{run_type} {name} precision': precision,
                   f'{run_type} {name} recall': recall,
                   'step': step, 'epoch': epoch,
                   })


def model_testing(net, cfg: experiment_manager.CfgNode, device: str, step: int, epoch: float):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.ConditionalUtilizationRate()
    measurer_fusion = metrics.MultiThresholdMetric(thresholds, 'fusion')
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

            pred_fusion = (pred_sar_mm + pred_optical_mm) / 2
            measurer_fusion.add_sample(y, pred_fusion)

            net.set_mmtm_mode('sar')
            out_sar_um, _ = net.forward(x_sar, x_optical)
            pred_sar_um = torch.sigmoid(out_sar_um).detach()
            measurer_sar_um.add_sample(y, pred_sar_um)

            net.set_mmtm_mode('optical')
            out_optical_um, _ = net.forward(x_sar, x_optical)
            pred_optical_um = torch.sigmoid(out_optical_um).detach()
            measurer_optical_um.add_sample(y, pred_optical_um)

        for measurer in [measurer_sar_mm, measurer_sar_um, measurer_optical_mm, measurer_optical_um, measurer_fusion]:
            f1s = measurer.compute_f1()
            precisions, recalls = measurer.precision, measurer.recall

            # best f1 score for passed thresholds
            f1 = f1s.max()
            argmax_f1 = f1s.argmax()

            precision = precisions[argmax_f1]
            recall = recalls[argmax_f1]

            wandb.log({f'test {measurer.name} F1': f1.item(),
                       f'test {measurer.name} precision': precision.item(),
                       f'test {measurer.name} recall': recall.item(),
                       'step': step, 'epoch': epoch,
                       })


def compute_mmtm_features(net, cfg: experiment_manager.CfgNode, device: str, run_type: str):
    net.to(device)
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

    return net
