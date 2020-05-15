import torch.utils.data as data
import numpy as np
import torch
import os
from random import shuffle

from src.utils import build_LFR_features, wav_padding, label_padding
from src.const import Const
from src.make_fbank import compute_fbank_from_file


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """
    def __init__(self, *args, feature_dim, char_list, path_list, label_list, LFR_m=1, LFR_n=1, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = LFRCollate(feature_dim=feature_dim, char_list=char_list,
                                     path_list=path_list, label_list=label_list, LFR_m=LFR_m, LFR_n=LFR_n)


class LFRCollate(object):
    """Build this wrapper to pass arguments(LFR_m, LFR_n) to _collate_fn"""
    def __init__(self, feature_dim, char_list, path_list, label_list, LFR_m=1, LFR_n=1):
        self.path_list = path_list
        self.label_list = label_list
        self.LFR_m = LFR_m
        self.LFR_n = LFR_n
        self.feature_dim = feature_dim
        self.char_list = char_list

    def __call__(self, batch):
        return _collate_fn(batch, self.feature_dim, self.char_list,
                           self.LFR_m, self.LFR_n, self.path_list, self.label_list)


def _collate_fn(batch, feature_dim, char_list, LFR_m, LFR_n, path_list, label_list):
    batch = batch[0]
    sub_list = batch['batch_list']
    label_lst = []
    input_lst = []
    error_count = []
    shuffle(sub_list)
    for i in sub_list:
        try:
            feature, label = get_fbank_and_hanzi_data(i, feature_dim, char_list, path_list, label_list)
            # 长度大于1600帧，过长，跳过
            if len(feature) > 1600:
                continue
            input_data = build_LFR_features(feature, LFR_m, LFR_n)
            label_lst.append(label)
            input_lst.append(input_data)
        except ValueError:
            error_count.append(i)
            continue
    # 删除异常语音信息
    if error_count != []:
        input_lst = np.delete(input_lst, error_count, axis=0)
        label_lst = np.delete(label_lst, error_count, axis=0)
    pad_wav_data, pad_lengths = wav_padding(input_lst)
    pad_target_data, _ = label_padding(label_lst, Const.IGNORE_ID)
    padded_input = torch.from_numpy(pad_wav_data).float()
    input_lengths = torch.from_numpy(pad_lengths)
    padded_target = torch.from_numpy(pad_target_data).long()
    return padded_input, input_lengths, padded_target


def get_fbank_and_hanzi_data(index, feature_dim, char_list, path_list, label_list):
    '''
    获取一条语音数据的Fbank与拼音信息
    :param index: 索引位置
    :return: 返回相应信息
    '''
    data_path = Const.SpeechDataPath
    noise_path = Const.NoiseOutPath
    try:
        # Fbank特征提取函数(从feature_python)
        file = os.path.join(data_path, path_list[index])
        noise_file = os.path.join(noise_path, path_list[index])
        feature = compute_fbank_from_file(file, feature_dim=feature_dim) if os.path.isfile(file) else \
            compute_fbank_from_file(noise_file, feature_dim=feature_dim)
        label = han2id(label_list[index], char_list)
        # 将错误数据进行抛出异常,并处理
        return feature, label
    except ValueError:
        raise ValueError


def han2id(line, vocab):
    """
    文字转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
    :param line:
    :param vocab:
    :return:
    """
    try:
        line = line.strip()
        res = []
        for han in line:
            if han == Const.PAD_FLAG:
                res.append(Const.PAD)
            elif han == Const.SOS_FLAG:
                res.append(Const.SOS)
            elif han == Const.EOS_FLAG:
                res.append(Const.EOS)
            else:
                res.append(vocab.index(han))
        return res
    except ValueError:
        raise ValueError