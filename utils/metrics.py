import torch
import numpy as np
import sys


class MultiThresholdMetric(object):
    def __init__(self, threshold, name: str = None):

        self._thresholds = threshold[ :, None, None, None, None] # [Tresh, B, C, H, W]
        self._data_dims = (-1, -2, -3, -4) # For a B/W image, it should be [Thresh, B, C, H, W],

        self.name = name

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def add_sample(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.bool()[None,...] # [Thresh, B,  C, ...]
        y_pred = y_pred[None, ...]  # [Thresh, B, C, ...]
        y_pred_offset = (y_pred - self._thresholds + 0.5).round().bool()

        self.TP += (y_true & y_pred_offset).sum(dim=self._data_dims).float()
        self.TN += (~y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FP += (y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FN += (~y_true & y_pred_offset).sum(dim=self._data_dims).float()

    @property
    def precision(self):
        if hasattr(self, '_precision'):
            '''precision previously computed'''
            return self._precision

        denom = (self.TP + self.FP).clamp(10e-05)
        self._precision = self.TP / denom
        return self._precision

    @property
    def recall(self):
        if hasattr(self, '_recall'):
            '''recall previously computed'''
            return self._recall

        denom = (self.TP + self.FN).clamp(10e-05)
        self._recall = self.TP / denom
        return self._recall

    def compute_basic_metrics(self):
        '''
        Computes False Negative Rate and False Positive rate
        :return:
        '''

        false_pos_rate = self.FP/(self.FP + self.TN)
        false_neg_rate = self.FN / (self.FN + self.TP)

        return false_pos_rate, false_neg_rate

    def compute_f1(self):
        denom = (self.precision + self.recall).clamp(10e-05)
        return 2 * self.precision * self.recall / denom


class ConditionalUtilizationRate(object):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.epsilon = 10e-05

        self.data = {
            'multimodal': {
                'sar': {
                    'TP': 0,
                    'TN': 0,
                    'FP': 0,
                    'FN': 0,
                },
                'optical': {
                    'TP': 0,
                    'TN': 0,
                    'FP': 0,
                    'FN': 0,
                },
                'fusion': {
                    'TP': 0,
                    'TN': 0,
                    'FP': 0,
                    'FN': 0,
                },
            },
            'unimodal': {
                'sar': {
                    'TP': 0,
                    'TN': 0,
                    'FP': 0,
                    'FN': 0,
                },
                'optical': {

                }
            }
        }

    def add_sample(self, y_true: torch.Tensor, y_pred: torch.Tensor, mode: str, modality: str):
        y_true = y_true.bool().flatten()
        y_pred = (y_pred > self.threshold).flatten()
        self.data[mode][modality]['TP'] += torch.sum((y_true & y_pred)).float()
        self.data[mode][modality]['TN'] += torch.sum((~y_true & ~y_pred)).float()
        self.data[mode][modality]['FP'] += torch.sum((~y_true & y_pred)).float()
        self.data[mode][modality]['FN'] += torch.sum((y_true & ~y_pred)).float()


    def get_true_positives(self, mode: str, modality: str):
        return self.data[mode][modality]['TP']

    def get_false_positives(self, mode: str, modality: str):
        return self.data[mode][modality]['FP']

    def get_false_negatives(self, mode: str, modality: str):
        return self.data[mode][modality]['FN']

    def get_precision(self, mode: str, modality: str) -> float:
        tp = self.get_true_positives(mode, modality)
        fp = self.get_false_positives(mode, modality)
        denom = (tp + fp) + self.epsilon
        return tp / denom

    def get_recall(self, mode: str, modality: str):
        tp = self.get_true_positives(mode, modality)
        fn = self.get_false_negatives(mode, modality)
        denom = (tp + fn) + self.epsilon
        return tp / denom

    def compute_f1(self, mode: str, modality: str):
        precision = self.get_precision(mode, modality)
        recall = self.get_recall(mode, modality)
        denom = precision + recall + self.epsilon
        return 2 * precision * recall / denom

    def get_conditional_utilization_rates(self):
        # for sar
        f1_score_mm_optical = self.compute_f1('multimodal', 'optical')
        f1_score_um_optical = self.compute_f1('unimodal', 'optical')
        cur_sar = (f1_score_mm_optical - f1_score_um_optical) / f1_score_mm_optical

        # for optical
        f1_score_mm_sar = self.compute_f1('multimodal', 'sar')
        f1_score_um_sar = self.compute_f1('unimodal', 'sar')
        cur_optical = (f1_score_mm_sar - f1_score_um_sar) / f1_score_mm_sar

        return cur_sar, cur_optical


def true_pos(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * torch.round(y_pred), dim=dim)  # Only sum along H, W axis, assuming no C


def false_pos(y_true, y_pred, dim=0):
    return torch.sum((1. - y_true) * torch.round(y_pred), dim=dim)


def false_neg(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * (1. - torch.round(y_pred)), dim=dim)


def precision(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int):
    TP = true_pos(y_true, y_pred, dim)
    FP = false_pos(y_true, y_pred, dim)
    denom = TP + FP
    denom = torch.clamp(denom, 10e-05)
    return TP / denom


def recall(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int):
    TP = true_pos(y_true, y_pred, dim)
    FN = false_neg(y_true, y_pred, dim)
    denom = TP + FN
    denom = torch.clamp(denom, 10e-05)
    return true_pos(y_true, y_pred, dim) / denom


def f1_score(gts:torch.Tensor, preds:torch.Tensor, multi_threashold_mode=False, dim=(-1, -2)):
    # FIXME Does not operate proper
    gts = gts.float()
    preds = preds.float()

    if multi_threashold_mode:
        gts = gts[:, None, ...] # [B, Thresh, ...]
        gts = gts.expand_as(preds)

    with torch.no_grad():
        recall_val = recall(gts, preds, dim)
        precision_val = precision(gts, preds, dim)
        denom = torch.clamp( (recall_val + precision_val), 10e-5)

        f1 = 2. * recall_val * precision_val / denom

    return f1


def f1_score_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    p = precision_from_prob(y_prob, y_true, threshold=threshold)
    r = recall_from_prob(y_prob, y_true, threshold=threshold)
    return 2 * (p * r) / (p + r + sys.float_info.epsilon)


def true_positives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    tp = np.sum(np.logical_and(y_pred, y_true))
    return tp.astype(np.int64)


def true_negatives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
    return tn.astype(np.int64)


def false_positives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
    return fp.astype(np.int64)


def false_negatives_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_pred = y_prob > threshold
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
    return fn.astype(np.int64)


def precision_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fp = false_positives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fp)


def recall_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fn = false_negatives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fn + sys.float_info.epsilon)


def iou_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    fp = false_positives_from_prob(y_prob, y_true, threshold)
    fn = false_negatives_from_prob(y_prob, y_true, threshold)
    return tp / (tp + fp + fn)


def kappa_from_prob(y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    tp = true_positives_from_prob(y_prob, y_true, threshold)
    tn = true_negatives_from_prob(y_prob, y_true, threshold)
    fp = false_positives_from_prob(y_prob, y_true, threshold)
    fn = false_negatives_from_prob(y_prob, y_true, threshold)
    nominator = 2 * (tp * tn - fn * fp)
    denominator = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    return nominator / denominator


def root_mean_square_error(y_pred: np.ndarray, y_true: np.ndarray):
    return np.sqrt(np.sum(np.square(y_pred - y_true)) / np.size(y_true))
