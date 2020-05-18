import wave
import numpy as np
import soundfile as sf
from python_speech_features import logfbank
# from sklearn import preprocessing


def compute_fbank_from_file(file, feature_dim=80):
    signal, sample_rate = sf.read(file)
    feature = compute_fbank_from_api(signal, sample_rate, nfilt=feature_dim)
    return feature


def compute_fbank_from_api(signal, sample_rate, nfilt):
    """
    Fbank特征提取, 结果进行零均值归一化操作
    :param wav_file: 文件路径
    :return: feature向量
    """
    feature = logfbank(signal, sample_rate, nfilt=nfilt)
    # feature = preprocessing.scale(feature)
    return feature


def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename, "rb")
    num_frame = wav.getnframes()
    # 获取声道数
    num_channel = wav.getnchannels()
    # 获取帧速率
    framerate = wav.getframerate()
    # 读取全部的帧
    str_data = wav.readframes(num_frame)
    # 关闭流
    wav.close()
    # 将声音文件数据转换为数组矩阵形式
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, num_channel
    wave_data = wave_data.T
    return wave_data, framerate


if __name__ == '__main__':
    wav_file = '/Volumes/disk/speech_data/data_thchs30/dev/C22_596.wav'
    wav, fr = read_wav_data(wav_file)
    print('----wav----')
    print(wav.shape)
    print(wav)
    print(fr)
    data = []
    for i in wav[0]:
        data.append(i/max(wav[0]))
    print(data)
    signal, sample_rate = sf.read(wav_file)
    print('----sf----')
    print(signal.shape)
    print(signal)
    print(sample_rate)