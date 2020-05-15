import sys
import csv
sys.path.append('../../')
from src.utils import process_dict
from src.dataset import AudioDataset
from src.dataloader import AudioDataLoader

if __name__ == '__main__':
    PATH = '../'
    char_list, sos_id, eos_id = process_dict(PATH + 'tmp/train_chars.txt')
    vocab_size = len(char_list)
    cv_dataset = AudioDataset('dev', 3)

    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=1,
                                feature_dim=80, char_list=char_list,
                                path_list=cv_dataset.path_lst, label_list=cv_dataset.han_lst,
                                LFR_m=1, LFR_n=1)

    for i, (data) in enumerate(cv_loader):
        inputs, length, target = data
        print(i)
        print(inputs)
        print(inputs.size())
        print(length)
        print(target)
        print(target.size())
        f = open('./input_data' + '_' + str(i) + '.csv', "w") # 创建csv文件
        writer = csv.writer(f)
        writer.writerows(inputs.cpu().numpy().tolist()[0])
