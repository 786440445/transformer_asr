import os
import time
import torch

from tensorboardX import SummaryWriter

from src.loss import cal_performance
from src.const import Const


class Solver(object):
    """
    """

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']

        self.model = model
        self.optimizer = optimizer

        # Low frame rate feature
        self.LFR_m = args.LFR_m
        self.LFR_n = args.LFR_n

        # Training config
        self.epochs = args.epochs
        self.label_smoothing = args.label_smoothing
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):
        write = SummaryWriter('../../model_log/tensorboard')
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            write.add_scalar('loss', tr_avg_loss, epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d_%.3f.pth.tar' % (epoch + 1, val_loss))
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)


            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        loader = self.tr_loader if not cross_valid else self.cv_loader
        # batch_nums = self.train_batch_nums if not cross_valid else self.verify_batch_nums

        for i, (data) in enumerate(loader):
            padded_input, input_lengths, padded_target = data
            # padded_input = padded_input.cuda()
            # input_lengths = input_lengths.cuda()
            # padded_target = padded_target.cuda()

            pred, gold = self.model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold,
                                              smoothing=self.label_smoothing)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            non_pad_mask = gold.ne(Const.IGNORE_ID)
            n_word = non_pad_mask.sum().item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

        return total_loss / (i + 1)
