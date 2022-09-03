import sys
from pathlib import Path
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    sar_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    optical_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        sar_loss_set, optical_loss_set, loss_set = [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            net.set_mmtm_mode('fusion')
            optimizer.zero_grad()

            x_sar = batch['x_sar'].to(device)
            x_optical = batch['x_optical'].to(device)
            y_gts = batch['y'].to(device)

            sar_logits, optical_logits = net(x_sar, x_optical)

            sar_loss = sar_criterion(sar_logits, y_gts)
            sar_loss_set.append(sar_loss.item())

            optical_loss = optical_criterion(optical_logits, y_gts)
            optical_loss_set.append(optical_loss.item())

            loss = sar_loss + optical_loss
            loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step, max_samples=1_000)
                evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start

                wandb.log({
                    'sar_loss': np.mean(sar_loss_set),
                    'optical_loss': np.mean(optical_loss_set),
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                sar_loss_set, optical_loss_set, loss_set = [], [], []

            if cfg.DEBUG:
                evaluation.model_testing(net, cfg, device, global_step, epoch_float, include_unimodal=True)
                break

            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)

        if epoch in save_checkpoints and not cfg.DEBUG:
            # computing average squeeze features for MMTM modules based on the training set
            evaluation.compute_mmtm_features(net, cfg, device, 'training')

            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)

            evaluation.model_testing(net, cfg, device, global_step, epoch_float, include_unimodal=True)
            evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step, include_unimodal=True)
            evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step, include_unimodal=True)
            net.reset_mmtm_features()


if __name__ == '__main__':

    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        entity='spacenet7',
        project=args.project,
        tags=['run', 'urban', 'extraction', 'segmentation', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
