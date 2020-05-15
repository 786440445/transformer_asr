import soundfile as sf
from python_speech_features import logfbank
from sklearn import preprocessing


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
    feature = preprocessing.scale(feature)
    return feature