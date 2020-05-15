from enum import IntEnum
import os

# 外网机
# SpeechDataPath 音频数据文件目录
# '../../../speech_data/'
# NoiseOutPath 噪声文件目录
# '/usr/corpus/noise_data'

# server
# SpeechDataPath 音频数据文件目录
# '/data/speech_data/'
# NoiseOutPath 噪声文件目录
# '/data/speech_data/noise_data'

# mac
# SpeechDataPath 音频数据文件目录
# '../../../speech_data/'
# NoiseOutPath 噪声文件目录
# '../../../speech_data/noise_data'

ServerId = 2


class ServerIndex(IntEnum):
    Linux = 0
    Server = 1
    Mac = 2


class Const:
    # SOS为起始标识符
    # EOS为结束标志符
    IGNORE_ID = -1
    PAD = 0
    SOS = 1
    EOS = 2
    PAD_FLAG = '<pad>'
    SOS_FLAG = '<sos>'
    EOS_FLAG = '<eos>'
    Flag_List = ['<pad>', '<sos>', '<eos>']

    # 噪声文件
    NoiseDataTxT = '../data/noise_data.txt'

    # 模型文件
    ModelFolder = '../../model_log/checkpoint/'
    ModelTensorboard = '../../model_log/tensorboard/'

    # 预测结果保存路径
    PredResultFolder = '../../model_log/pred/'

    if ServerId == ServerIndex.Linux:
        SpeechDataPath = '/home/chengli/matrix/speech_data/'
        NoiseOutPath = '/usr/corpus/noise_data/'

    elif ServerId == ServerIndex.Server:
        SpeechDataPath = '/data/speech_data'
        NoiseOutPath = '/data/speech_data/data'

    elif ServerId == ServerIndex.Mac:
        SpeechDataPath = '/Volumes/disk/speech_data'
        NoiseOutPath = '../../../speech_data/noise_data/'


if ServerId == ServerIndex.Server:
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"

