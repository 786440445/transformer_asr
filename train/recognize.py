#!/usr/bin/env python
import argparse
import torch
import random
import sys
sys.path.append('../../')
from transformer import Transformer
from utils import process_dict, GetEditDistance, build_LFR_features
from dataset import AudioDataset
from dataloader import get_fbank_and_hanzi_data

import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--dict', type=str, default='../../tmp/train_chars.txt',
                    help='Dictionary which should include <unk> <sos> <eos>')
# model
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--shuffle', default=1, type=int)
parser.add_argument('--feature_dim', default=80, type=int)
parser.add_argument('--model_path', type=str, default='../../model_log/checkpoint/epoch148.pth.tar',
                    help='Path to model file created by training')
# decode
parser.add_argument('--count', default=7176, type=int,
                    help='decoder count')
parser.add_argument('--beam_size', default=5, type=int,
                    help='Beam size')
parser.add_argument('--nbest', default=1, type=int,
                    help='Nbest size')
parser.add_argument('--decode_max_len', default=100, type=int,
                    help='Max output length. If ==0 (default), it uses a '
                    'end-detect function to automatically find maximum '
                    'hypothesis lengths')


def recognize(args):
    model, LFR_m, LFR_n = Transformer.load_model(args.model_path)
    print(model)
    model.eval()
    model.cuda()
    char_list, sos_id, eos_id = process_dict(args.dict)
    assert model.decoder.sos_id == sos_id and model.decoder.eos_id == eos_id
    tr_dataset = AudioDataset('test', args.batch_size)
    path_list = tr_dataset.path_lst
    label_list = tr_dataset.han_lst
    num_data = tr_dataset.path_count
    ran_num = random.randint(0, num_data - 1)

    num = args.count
    words_num = 0
    word_error_num = 0
    seq_error = 0
    data = ''
    with torch.no_grad():
        for index in range(num):
            try:
                print('\nthe ', index + 1, 'th example.')
                data += 'the ' + str(index+1) + 'th example.\n'
                index = (ran_num + index) % num_data
                standard_label = label_list[index]
                feature, label = get_fbank_and_hanzi_data(index, args.feature_dim, char_list, path_list, label_list)
                if len(feature) > 1600:
                    continue
                input = build_LFR_features(feature, args.LFR_m, args.LFR_n)
                input = torch.from_numpy(input).float()
                input_length = torch.tensor([input.size(0)], dtype=torch.int)
                input = input.cuda()
                nbest_hyps = model.recognize(input, input_length, char_list, args)
                pred_label = nbest_hyps[0]['yseq'][1:-1]
                pred_res = ''.join([char_list[index] for index in pred_label])
                print("stand:", label)
                print("pred :", pred_label)
                data += "stand:" + str(standard_label) + '\n'
                data += "pred :" + str(pred_res) + '\n'
                words_n = len(label)
                words_num += words_n
                word_distance = GetEditDistance(pred_label, label)
                if (word_distance <= words_n):
                    word_error_num += word_distance
                else:
                    word_error_num += words_n

                if pred_label != label:
                    seq_error += 1
            except ValueError:
                continue
    print('WER = ', (1 - word_error_num / words_num) * 100, '%')
    print('CER = ', (1 - seq_error / args.count) * 100, '%')
    data += 'WER = ' + str((1 - word_error_num / words_num) * 100) + '%'
    data += 'CER = ' + str((1 - seq_error / args.count) * 100) + '%'
    with open('../../model_log/pred/test_' + str(args.count) + '.txt', 'w', encoding='utf-8') as f:
        f.writelines(data)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    recognize(args)
