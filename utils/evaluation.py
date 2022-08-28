import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, metrics


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int, max_samples: int = None):
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


def model_testing(net, cfg, device, step, epoch):
    net.to(device)
    net.eval()

    dataset = datasets.SpaceNet7Dataset(cfg)

    y_true_dict = {'total': []}
    y_pred_sar_dict = {'total': []}
    y_pred_optical_dict = {'total': []}
    y_pred_fusion_dict = {'total': []}

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset.__getitem__(index)
            x_sar = sample['x_sar'].to(device)
            x_optical = sample['x_optical'].to(device)
            y_true = sample['y'].to(device)
            logits_sar, logits_optical = net(x_sar.unsqueeze(0), x_optical.unsqueeze(0))
            y_pred_sar = torch.sigmoid(logits_sar).detach().cpu().numpy()
            y_pred_optical = torch.sigmoid(logits_optical).detach().cpu().numpy()
            y_pred_fusion = (y_pred_sar + y_pred_optical) / 2
            y_true = y_true.detach().cpu().numpy().flatten()

            y_pred_sar = (y_pred_sar > 0.5).flatten()
            y_pred_optical = (y_pred_optical > 0.5).flatten()
            y_pred_fusion = (y_pred_fusion > 0.5).flatten()

            region = sample['region']
            if region not in y_true_dict.keys():
                y_true_dict[region] = [y_true]
                y_pred_sar_dict[region] = [y_pred_sar]
                y_pred_optical_dict[region] = [y_pred_optical]
                y_pred_fusion_dict[region] = [y_pred_fusion]
            else:
                y_true_dict[region].append(y_true)
                y_pred_sar_dict[region].append(y_pred_sar)
                y_pred_optical_dict[region].append(y_pred_optical)
                y_pred_fusion_dict[region].append(y_pred_fusion)

            y_true_dict['total'].append(y_true)
            y_pred_sar_dict['total'].append(y_pred_sar)
            y_pred_optical_dict['total'].append(y_pred_optical)
            y_pred_fusion_dict['total'].append(y_pred_fusion)

    def evaluate_region(region_name: str, y_pred_dict: dict, name: str):
        y_true_region = torch.Tensor(np.concatenate(y_true_dict[region_name])).flatten()
        y_pred_region = torch.Tensor(np.concatenate(y_pred_dict[region_name])).flatten()
        prec = metrics.precision(y_true_region, y_pred_region, dim=0).item()
        rec = metrics.recall(y_true_region, y_pred_region, dim=0).item()
        f1 = metrics.f1_score(y_true_region, y_pred_region, dim=0).item()

        wandb.log({f'{region_name} {name} F1': f1,
                   f'{region_name} {name} precision': prec,
                   f'{region_name} {name} recall': rec,
                   'step': step, 'epoch': epoch,
                   })

    for y_pred_dict, name in zip([y_pred_sar_dict, y_pred_optical_dict, y_pred_fusion_dict],
                                 ['sar','optical', 'fusion']):
        for region in dataset.regions['regions'].values():
            evaluate_region(region, y_pred_dict, name)
        evaluate_region('total', y_pred_dict, name)


