#!/usr/bin/env python3
import difflib
import numpy as np
from src.const import Const

def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


def process_dict(dict_path):
    with open(dict_path, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').strip() for entry in dictionary]
    char_list = Const.Flag_List + char_list
    return char_list, Const.SOS, Const.EOS


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    char_list, sos_id, eos_id = process_dict(path)
    print(char_list, sos_id, eos_id)

# * ------------------ recognition related ------------------ *


def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:-1]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(
            hyp, char_list)

        # copy ground-truth
        out_dic = dict(js['output'][0].items())

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            print('groundtruth: %s' % out_dic['text'])
            print('prediction : %s' % out_dic['rec_text'])

    return new_js


# -- Transformer Related --
import torch

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask

# word error rate------------------------------------
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost

def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else:
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def wav_padding(wav_data_lst):
    feature_dim = wav_data_lst[0].shape[1]
    # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
    wav_lens = np.array([len(data) for data in wav_data_lst])
    # 取一个batch中的最长
    wav_max_len = max(wav_lens)
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, feature_dim), dtype=np.float32)
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens


def label_padding(label_data_lst, pad_idx):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len), dtype=np.int32)
    new_label_data_lst += pad_idx
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens