import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from pathlib import Path

from utils import experiment_manager


def create_network(cfg):
    if cfg.MODEL.TYPE == 'ds_unet':
        return DualStreamUNet(cfg)
    elif cfg.MODEL.TYPE == 'ds_unet_mmtmencoder':
        return DualStreamUNetMMTMEncoder(cfg)
    elif cfg.MODEL.TYPE == 'ds_unet_mmtmdecoder':
        return DualStreamUNetMMTMDecoder(cfg)
    else:
        return UNet(cfg)


def save_checkpoint(network, optimizer, epoch: int, step: int, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch: int, cfg: experiment_manager.CfgNode, device):
    net = create_network(cfg)
    net.to(device)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        print(m.weight)
    else:
        print('error')


class DualStreamUNet(nn.Module):

    def __init__(self, cfg):
        super(DualStreamUNet, self).__init__()
        self._cfg = cfg
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        # sentinel-1 sar unet stream
        self.inc_sar = InConv(len(cfg.DATALOADER.SENTINEL1_BANDS), topology[0], DoubleConv)
        self.encoder_sar = Encoder(cfg)
        self.decoder_sar = Decoder(cfg)
        self.outc_sar = OutConv(topology[0], out)

        # sentinel-2 optical unet stream
        self.inc_optical = InConv(len(cfg.DATALOADER.SENTINEL2_BANDS), topology[0], DoubleConv)
        self.encoder_optical = Encoder(cfg)
        self.decoder_optical = Decoder(cfg)
        self.outc_optical = OutConv(topology[0], out)


    def forward(self, x_sar: torch.Tensor, x_optical: torch.Tensor):
        # sar
        x_sar = self.inc_sar(x_sar)
        features_sar = self.encoder_sar(x_sar)
        x_sar = self.decoder_sar(features_sar)
        out_sar = self.outc_sar(x_sar)

        # optical
        x_optical = self.inc_optical(x_optical)
        features_optical = self.encoder_optical(x_optical)
        x_optical = self.decoder_optical(features_optical)
        out_optical = self.outc_optical(x_optical)

        return out_sar, out_optical


class DualStreamUNetMMTMEncoder(nn.Module):

    def __init__(self, cfg):
        super(DualStreamUNetMMTMEncoder, self).__init__()
        self._cfg = cfg
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        # sentinel-1 sar unet stream
        self.inc_sar = InConv(len(cfg.DATALOADER.SENTINEL1_BANDS), topology[0], DoubleConv)
        self.decoder_sar = Decoder(cfg)
        self.outc_sar = OutConv(topology[0], out)

        # sentinel-2 optical unet stream
        self.inc_optical = InConv(len(cfg.DATALOADER.SENTINEL2_BANDS), topology[0], DoubleConv)
        self.decoder_optical = Decoder(cfg)
        self.outc_optical = OutConv(topology[0], out)

        self.mmtm_encoder = MMTMEncoder(cfg)
        self.mmtm_mode = 'fusion'

    def forward(self, x_sar: torch.Tensor, x_optical: torch.Tensor):
        # in convolutions
        x_sar = self.inc_sar(x_sar)
        x_optical = self.inc_optical(x_optical)

        # encoding with mmtm
        features_sar, features_optical = self.mmtm_encoder(x_sar, x_optical)

        # decoding
        x_sar = self.decoder_sar(features_sar)
        x_optical = self.decoder_optical(features_optical)

        # out convolutions
        out_sar = self.outc_sar(x_sar)
        out_optical = self.outc_optical(x_optical)

        return out_sar, out_optical

    def compute_unimodal_features(self, x_sar: torch.Tensor, x_optical: torch.Tensor):
        # in convolutions
        x_sar = self.inc_sar(x_sar)
        x_optical = self.inc_optical(x_optical)
        self.mmtm_encoder.forward(x_sar, x_optical, compute_unimodal_mmtm_features=True)

    def reset_mmtm_features(self):
        for mmtm in self.mmtm_encoder.mmtm_seq.values():
            mmtm.reset_average_squeeze()

    def set_mmtm_mode(self, new_mode: str):
        self.mmtm_mode = new_mode
        for mmtm_module in self.mmtm_encoder.mmtm_seq.values():
            mmtm_module.set_mode(new_mode)


class UNet(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None, enable_outc=True):

        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.enable_outc = enable_outc
        self.outc = OutConv(first_chan, n_classes)

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]  # topography upwards
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer

            layer = Down(in_dim, out_dim, DoubleConv)

            print(f'down{idx + 1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = Up(in_dim, out_dim, DoubleConv)

            print(f'up{idx + 1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x1, x2=None):
        x = x1 if x2 is None else torch.cat((x1, x2), 1)

        x1 = self.inc(x)

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        out = self.outc(x1) if self.enable_outc else x1

        return out


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()
        return inputs


class MMTMEncoder(nn.Module):
    def __init__(self, cfg):
        super(MMTMEncoder, self).__init__()

        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY

        # Variable scale
        down_topo = topology
        down_dict_sar, down_dict_optical, mmtm_dict = OrderedDict(), OrderedDict(), OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]

            # mmtm unit
            mmtm = MMTM(in_dim, in_dim, 1)
            mmtm_dict[f'mmtm{idx + 1}'] = mmtm

            # down units
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer
            down_sar = Down(in_dim, out_dim, DoubleConv)
            down_dict_sar[f'down{idx + 1}_sar'] = down_sar
            down_optical = Down(in_dim, out_dim, DoubleConv)
            down_dict_optical[f'down{idx + 1}_optical'] = down_optical

        self.down_seq_sar = nn.ModuleDict(down_dict_sar)
        self.down_seq_optical = nn.ModuleDict(down_dict_optical)
        self.mmtm_seq = nn.ModuleDict(mmtm_dict)

    def forward(self, x_sar: torch.Tensor, x_optical: torch.Tensor, compute_unimodal_mmtm_features: bool = False) -> tuple:

        inputs_sar, inputs_optical = [x_sar], [x_optical]
        # Downward U:
        for mmtm, down_sar, down_optical in zip(self.mmtm_seq.values(), self.down_seq_sar.values(),
                                                self.down_seq_optical.values()):

            features_sar, features_optical = inputs_sar[-1], inputs_optical[-1]

            if compute_unimodal_mmtm_features:
                mmtm.compute_unimodal_average_squeeze(features_sar, features_optical)

            features_sar_mmtm, features_optical_mmtm = mmtm(features_sar, features_optical)

            # sar
            out_sar = down_sar(features_sar_mmtm)
            inputs_sar.append(out_sar)
            # optical
            out_optical = down_optical(features_optical_mmtm)
            inputs_optical.append(out_optical)

        inputs_sar.reverse()
        inputs_optical.reverse()

        return inputs_sar, inputs_optical


class MMTM(nn.Module):
    def __init__(self, dim_sar, dim_optical, ratio):
        super(MMTM, self).__init__()
        self.dim_sar = dim_sar
        self.dim_optical = dim_optical
        self.dim = dim_sar + dim_optical

        dim_out = int(2 * self.dim / ratio)
        self.fc_squeeze = nn.Linear(self.dim, dim_out)

        self.fc_sar = nn.Linear(dim_out, dim_sar)
        self.fc_optical = nn.Linear(dim_out, dim_optical)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # for unimodal experiments
        # https://discuss.pytorch.org/t/how-to-make-a-tensor-part-of-model-parameters/51037/6
        self.average_sar_squeeze = nn.Parameter(torch.zeros((1, dim_sar), requires_grad=False))
        self.average_optical_squeeze = nn.Parameter(torch.zeros((1, dim_optical), requires_grad=False))
        self.average_n = 0
        self.mode = 'fusion'

    def forward(self, sar, optical):
        squeeze_array = []
        for tensor in [sar, optical]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        if self.mode == 'sar':
            squeeze_array[-1] = self.average_optical_squeeze
        if self.mode == 'optical':
            squeeze_array[0] = self.average_sar_squeeze

        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        sar_out = self.fc_sar(excitation)
        optical_out = self.fc_optical(excitation)

        sar_out = self.sigmoid(sar_out)
        optical_out = self.sigmoid(optical_out)

        dim_diff = len(sar.shape) - len(sar_out.shape)
        sar_out = sar_out.view(sar_out.shape + (1,) * dim_diff)

        dim_diff = len(optical.shape) - len(optical_out.shape)
        optical_out = optical_out.view(optical_out.shape + (1,) * dim_diff)

        return sar * sar_out, optical * optical_out

    def compute_unimodal_average_squeeze(self, sar: torch.Tensor, optical: torch.Tensor):

        n_batch = sar.shape[0]
        new_total_n = self.average_n + n_batch

        sar_tview = sar.detach().view(sar.shape[:2] + (-1,))
        sar_squeeze = torch.mean(sar_tview, dim=-1)
        total_batch_sar_squeeze = torch.sum(sar_squeeze, dim=0).unsqueeze(0)
        new_total_sar_squeeze = self.average_sar_squeeze * self.average_n + total_batch_sar_squeeze
        new_average_sar_squeeze = new_total_sar_squeeze / new_total_n
        self.average_sar_squeeze = nn.Parameter(new_average_sar_squeeze)

        optical_tview = optical.detach().view(optical.shape[:2] + (-1,))
        optical_squeeze = torch.mean(optical_tview, dim=-1)
        total_batch_optical_squeeze = torch.sum(optical_squeeze, dim=0).unsqueeze(0)
        new_total_optical_squeeze = self.average_optical_squeeze * self.average_n + total_batch_optical_squeeze
        new_average_optical_squeeze = new_total_optical_squeeze / new_total_n
        self.average_optical_squeeze = nn.Parameter(new_average_optical_squeeze)

    def reset_average_squeeze(self):
        self.average_sar_squeeze = nn.Parameter(torch.zeros((1, self.dim_sar), requires_grad=False))
        self.average_optical_squeeze = nn.Parameter(torch.zeros((1, self.dim_optical), requires_grad=False))
        self.average_n = 0

    def set_mode(self, new_mode: str):
        self.mode = new_mode


class Decoder(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode, topology: list = None):
        super(Decoder, self).__init__()
        self.cfg = cfg

        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        # Variable scale
        n_layers = len(topology)
        up_topo = [topology[0]]  # topography upwards
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]  # last layer
            up_topo.append(out_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:

        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        return x1


# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
