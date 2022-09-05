import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics, experiment_manager


def model_evaluation(net, cfg: experiment_manager.CfgNode, device: str, run_type: str, epoch: float, step: int,
                     include_unimodal: bool = False, max_samples: int = None):
    net.to(device)
    net.eval()

    measurer = metrics.Measurer()
    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=True)

    stop_step = len(dataloader) if max_samples is None else max_samples

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step == stop_step:
                break

            y = batch['y'].to(device)
            x_sar = batch['x_sar'].to(device)
            x_optical = batch['x_optical'].to(device)

            net.set_mmtm_mode('fusion')
            logits_sar, logits_optical = net(x_sar, x_optical)
            pred_sar, pred_optical = torch.sigmoid(logits_sar).detach(), torch.sigmoid(logits_optical).detach()
            measurer.add_sample(y, pred_sar, 'multimodal', 'sar')
            measurer.add_sample(y, pred_optical, 'multimodal', 'optical')
            pred_fusion = (pred_sar + pred_optical) / 2
            measurer.add_sample(y, pred_fusion, 'multimodal', 'fusion')

            if include_unimodal:
                net.set_mmtm_mode('sar')
                out_sar_um, _ = net.forward(x_sar, x_optical)
                pred_sar_um = torch.sigmoid(out_sar_um).detach()
                measurer.add_sample(y, pred_sar_um, 'unimodal', 'sar')

                net.set_mmtm_mode('optical')
                out_optical_um, _ = net.forward(x_sar, x_optical)
                pred_optical_um = torch.sigmoid(out_optical_um).detach()
                measurer.add_sample(y, pred_optical_um, 'unimodal', 'optical')

            if cfg.DEBUG:
                break

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    modes = ['multimodal', 'unimodal'] if include_unimodal else ['multimodal']
    for mode in modes:
        for modality in ['sar', 'optical', 'fusion']:
            if mode == 'unimodal' and modality == 'fusion':
                continue
            f1 = measurer.compute_f1(mode, modality)
            precision = measurer.compute_precision(mode, modality)
            recall = measurer.compute_recall(mode, modality)

            wandb.log({f'{run_type} {mode} {modality} F1': f1.item(),
                       f'{run_type} {mode} {modality} precision': precision.item(),
                       f'{run_type} {mode} {modality} recall': recall.item(),
                       'step': step, 'epoch': epoch,
                       })

    if include_unimodal:
        cur_sar, cur_optical = measurer.compute_conditional_utilization_rates()
        dutil = cur_optical - cur_sar
        wandb.log({f'{run_type} CUR sar F1': cur_sar.item(),
                   f'{run_type} CUR optical F1': cur_optical.item(),
                   f'{run_type} dutil F1': dutil.item(),
                   'step': step, 'epoch': epoch,
                   })


def model_testing(net, cfg: experiment_manager.CfgNode, device: str, step: int, epoch: float,
                  include_unimodal: bool = False):
    net.to(device)
    net.eval()

    measurer = metrics.Measurer()
    dataset = datasets.SpaceNet7Dataset(cfg)

    with torch.no_grad():
        for item in dataset:
            y = item['y'].to(device)
            x_sar = item['x_sar'].to(device).unsqueeze(0)
            x_optical = item['x_optical'].to(device).unsqueeze(0)

            net.set_mmtm_mode('fusion')
            logits_sar_mm, logits_optical_mm = net.forward(x_sar, x_optical)
            pred_sar_mm, pred_optical_mm = torch.sigmoid(logits_sar_mm).detach(), torch.sigmoid(logits_optical_mm).detach()
            measurer.add_sample(y, pred_sar_mm, 'multimodal', 'sar')
            measurer.add_sample(y, pred_sar_mm, 'multimodal', 'optical')

            pred_fusion = (pred_sar_mm + pred_optical_mm) / 2
            measurer.add_sample(y, pred_fusion, 'multimodal', 'fusion')

            if include_unimodal:
                net.set_mmtm_mode('sar')
                logits_sar_um, _ = net.forward(x_sar, x_optical)
                pred_sar_um = torch.sigmoid(logits_sar_um).detach()
                measurer.add_sample(y, pred_sar_um, 'unimodal', 'sar')

                net.set_mmtm_mode('optical')
                logits_optical_um, _ = net.forward(x_sar, x_optical)
                pred_optical_um = torch.sigmoid(logits_optical_um).detach()
                measurer.add_sample(y, pred_optical_um, 'unimodal', 'optical')

    modes = ['multimodal', 'unimodal'] if include_unimodal else ['multimodal']
    for mode in modes:
        for modality in ['sar', 'optical', 'fusion']:
            if mode == 'unimodal' and modality == 'fusion':
                continue

            f1 = measurer.compute_f1(mode, modality)
            precision = measurer.compute_precision(mode, modality)
            recall = measurer.compute_recall(mode, modality)

            wandb.log({f'test {mode} {modality} F1': f1.item(),
                       f'test {mode} {modality} precision': precision.item(),
                       f'test {mode} {modality} recall': recall.item(),
                       'step': step, 'epoch': epoch,
                       })

    if include_unimodal:
        cur_sar, cur_optical = measurer.compute_conditional_utilization_rates()
        dutil = cur_optical - cur_sar
        wandb.log({f'test CUR sar F1': cur_sar.item(),
                   f'test CUR optical F1': cur_optical.item(),
                   f'test dutil F1': dutil.item(),
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
