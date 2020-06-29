#!/usr/bin/env python
import argparse
import torch
import sys
import os
home_dir = os.getcwd()
sys.path.append(home_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.decoder import Decoder
from src.encoder import Encoder
from src.transformer import Transformer
from src.solver import Solver
from src.utils import process_dict
from src.optimizer import TransformerOptimizer
from src.dataset import AudioDataset
from src.dataloader import AudioDataLoader
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Training "
    "(Transformer framework).")
parser.add_argument('--dict', type=str,
                    default=os.path.join(home_dir, r'tmp\train_chars.txt'),
                    help='Dictionary which should include <unk> <sos> <eos>')
# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')
# Network architecture
# encoder
parser.add_argument('--d_input', default=80, type=int,
                    help='Dim of encoder input (before LFR)')
parser.add_argument('--d_low_dim', default=64, type=int,
                    help='Low Dim of encoder input')
parser.add_argument('--n_layers_enc', default=6, type=int,
                    help='Number of encoder stacks')
parser.add_argument('--n_head', default=8, type=int,
                    help='Number of Multi Head Attention (MHA)')
parser.add_argument('--d_k', default=64, type=int,
                    help='Dimension of key')
parser.add_argument('--d_v', default=64, type=int,
                    help='Dimension of value')
parser.add_argument('--d_model', default=512, type=int,
                    help='Dimension of model')
parser.add_argument('--d_inner', default=2048, type=int,
                    help='Dimension of inner')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='Dropout rate')
parser.add_argument('--pe_maxlen', default=1000, type=int,
                    help='Positional Encoding max len')
# decoder
parser.add_argument('--d_word_vec', default=512, type=int,
                    help='Dim of decoder embedding')
parser.add_argument('--n_layers_dec', default=6, type=int,
                    help='Number of decoder stacks')
parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
                    help='share decoder embedding with decoder projection')
# Loss
parser.add_argument('--label_smoothing', default=0.1, type=float,
                    help='label smoothing')

# Training config
parser.add_argument('--epochs', default=150, type=int,
                    help='Number of maximum epochs')
# minibatch
parser.add_argument('--shuffle', default=1, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
parser.add_argument('--feature_dim', default=80, type=int,
                    help='feature dimension of data')
# optimizer
parser.add_argument('--init_lr', default=1, type=float,
                    help='tunable scalar multiply to learning rate')
parser.add_argument('--warmup_steps', default=4000, type=int,
                    help='warmup steps')
# save and load model
parser.add_argument('--save_folder', default='./model_log/checkpoint/',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=1, type=int,
                    help='Frequency of printing training infomation')


def main(args):
    # load dictionary and generate char_list, sos_id, eos_id
    char_list, sos_id, eos_id = process_dict(args.dict)
    vocab_size = len(char_list)
    tr_dataset = AudioDataset('train', args.batch_size)
    cv_dataset = AudioDataset('dev', args.batch_size)

    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                shuffle=args.shuffle,
                                feature_dim=args.feature_dim, char_list=char_list,
                                path_list=tr_dataset.path_lst, label_list=tr_dataset.han_lst,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                feature_dim=args.feature_dim, char_list=char_list,
                                path_list=cv_dataset.path_lst, label_list=cv_dataset.han_lst,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n)

    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    encoder = Encoder(args.d_input * args.LFR_m, args.d_low_dim,
                      args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id, eos_id, vocab_size,
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)
    model = Transformer(encoder, decoder)
    print(model)
    model.cuda()
    # optimizer
    optimizier = TransformerOptimizer(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.init_lr,
        args.d_model,
        args.warmup_steps)

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

