"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.
"""
import torch.utils.data as data

from src.const import Const
from src.param import DataHparams
from src.load_corpus import LoadData


class AudioDataset(data.Dataset):
    def __init__(self, type, batch_size):
        super(AudioDataset, self).__init__()
        hparams = DataHparams()
        parser = hparams.parser
        data_hp = parser.parse_args()
        corpus_data = LoadData(data_hp, type)

        self.path_lst = corpus_data.path_lst
        self.han_lst = corpus_data.han_lst
        self.path_count = len(self.path_lst)
        self.batch_size = batch_size
        self.data_path = Const.SpeechDataPath
        self.noise_path = Const.NoiseOutPath
        # 随机选取batch_size个wav数据组成一个batch_wav_data
        batch_nums = self.path_count // self.batch_size
        rest = self.path_count % self.batch_size
        index_list = list(range(0, self.path_count))
        batch_list = []
        # 多加一个表示最后的一个个数不足的batch
        if rest != 0:
            batch_nums += 1
        for i in range(batch_nums):
            begin = i * batch_size
            end = min(self.path_count, begin + batch_size)
            batch_index = index_list[begin: end]
            dict = {'batch_list': batch_index}
            batch_list.append(dict)
        self.minibatch = batch_list

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)