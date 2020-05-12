import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)

import os
from itertools import chain
import numpy as np
from tqdm import tqdm

from dataset_factory import DatasetSet
from models.decoder import Decoder
from models.encoder import Encoder
from models.domain_classifier import DomainClassifier
from utils.helper_functions import cross_entropy_loss, LossMeter, wrap_cuda
from config import config
from utils.logger import Logger


class Trainer:
    def __init__(self, config):
        self.config = config
        self.config.data.n_datasets = len(config.data.datasets)

        print("No of datasets used:", self.config.data.n_datasets)

        torch.manual_seed(config.env.seed)
        torch.cuda.manual_seed(config.env.seed)
        self.expPath = self.config.env.expPath

        self.logger = Logger("Training", "logs/training.log")
        self.data = [DatasetSet(data_path, config.data.seq_len, config.data) for data_path in config.data.datasets]

        self.losses_recon = [LossMeter(f'recon {i}') for i in range(self.config.data.n_datasets)]
        self.loss_d_right = LossMeter('d')
        self.loss_total = [LossMeter(f'total {i}') for i in range(self.config.data.n_datasets)]

        self.evals_recon = [LossMeter(f'recon {i}') for i in range(self.config.data.n_datasets)]
        self.eval_d_right = LossMeter('eval d')
        self.eval_total = [LossMeter(f'eval total {i}') for i in range(self.config.data.n_datasets)]

        self.encoder = Encoder(config.encoder)
        self.decoders = torch.nn.ModuleList([Decoder(config.decoder) for _ in range(self.config.data.n_datasets)])
        self.classifier = DomainClassifier(config.domain_classifier, num_classes=self.config.data.n_datasets)

        states = None
        if config.env.checkpoint:

            checkpoint_args_path = os.path.dirname(config.env.checkpoint) + '/args.pth'
            checkpoint_args = torch.load(checkpoint_args_path)

            self.start_epoch = checkpoint_args[-1] + 1
            states = [torch.load(self.config.env.checkpoint + f'_{i}.pth')
                      for i in range(self.config.data.n_datasets)]

            self.encoder.load_state_dict(states[0]['encoder_state'])
            for i in range(self.config.data.n_datasets):
                self.decoders[i].load_state_dict(states[i]['decoder_state'])
            self.classifier.load_state_dict(states[0]['discriminator_state'])
            self.logger.info('Loaded checkpoint parameters')

            raise NotImplementedError
        else:
            self.start_epoch = 0

        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()
        for i, decoder in enumerate(self.decoders):
            self.decoders[i] = torch.nn.DataParallel(decoder).cuda()

        self.model_optimizers = [optim.Adam(chain(self.encoder.parameters(), decoder.parameters()), lr=config.data.lr)
                                 for decoder in self.decoders]

        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=config.data.lr)

        if config.env.checkpoint and config.env.load_optimizer:
            for i in range(self.config.data.n_datasets):
                self.model_optimizers[i].load_state_dict(states[i]['model_optimizer_state'])

            self.classifier_optimizer.load_state_dict(states[0]['d_optimizer_state'])

        self.lr_managers = []
        for i in range(self.config.data.n_datasets):
            self.lr_managers.append(
                torch.optim.lr_scheduler.ExponentialLR(self.model_optimizers[i], config.data.lr_decay))
            self.lr_managers[i].last_epoch = self.start_epoch
            self.lr_managers[i].step()

    def eval_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        z = self.encoder(x)
        y = self.decoders[dset_num](x, z)
        z_logits = self.classifier(z)

        z_classification = torch.max(z_logits, dim=1)[1]

        z_accuracy = (z_classification == dset_num).float().mean()

        self.eval_d_right.add(z_accuracy.data.item())

        # discriminator_right = F.cross_entropy(z_logits, dset_num).mean()
        discriminator_right = F.cross_entropy(z_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()
        recon_loss = cross_entropy_loss(y, x)

        self.evals_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        total_loss = discriminator_right.data.item() * self.config.domain_classifier.d_lambda + \
                     recon_loss.mean().data.item()

        self.eval_total[dset_num].add(total_loss)

        return total_loss

    def train_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        # Optimize D - classifier right
        z = self.encoder(x)
        z_logits = self.classifier(z)
        discriminator_right = F.cross_entropy(z_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()
        loss = discriminator_right * self.config.domain_classifier.d_lambda
        self.loss_d_right.add(loss.data.item())

        self.classifier_optimizer.zero_grad()
        loss.backward()
        if self.config.domain_classifier.grad_clip is not None:
            clip_grad_value_(self.classifier.parameters(), self.config.domain_classifier.grad_clip)

        self.classifier_optimizer.step()

        # optimize G - reconstructs well, classifier wrong
        z = self.encoder(x_aug)
        y = self.decoders[dset_num](x, z)
        z_logits = self.classifier(z)

        discriminator_wrong = - F.cross_entropy(z_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()

        if not (-100 < discriminator_right.data.item() < 100):
            self.logger.debug(f'z_logits: {z_logits.detach().cpu().numpy()}')
            self.logger.debug(f'dset_num: {dset_num}')

        recon_loss = cross_entropy_loss(y, x)
        self.losses_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        loss = (recon_loss.mean() + self.config.domain_classifier.d_lambda * discriminator_wrong)

        self.model_optimizers[dset_num].zero_grad()
        loss.backward()
        if self.config.domain_classifier.grad_clip is not None:
            clip_grad_value_(self.encoder.parameters(), self.config.domain_classifier.grad_clip)
            clip_grad_value_(self.decoders[dset_num].parameters(), self.config.domain_classifier.grad_clip)

        self.model_optimizers[dset_num].step()

        self.loss_total[dset_num].add(loss.data.item())

        return loss.data.item()

    def train_epoch(self, epoch):
        for meter in self.losses_recon:
            meter.reset()
        self.loss_d_right.reset()
        for i in range(len(self.loss_total)):
            self.loss_total[i].reset()

        self.encoder.train()
        self.classifier.train()
        for decoder in self.decoders:
            decoder.train()

        n_batches = self.config.data.epoch_len

        with tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
            for batch_num in range(n_batches):
                if self.config.data.short and batch_num == 3:
                    break

                dset_num = batch_num % self.config.data.n_datasets

                x, x_aug = next(self.data[dset_num].train_iter)

                x = wrap_cuda(x)
                x_aug = wrap_cuda(x_aug)
                batch_loss = self.train_batch(x, x_aug, dset_num)

                train_enum.set_description(f'Train (loss: {batch_loss:.2f}) epoch {epoch}')
                train_enum.update()

    def evaluate_epoch(self, epoch):
        for meter in self.evals_recon:
            meter.reset()
        self.eval_d_right.reset()
        for i in range(len(self.eval_total)):
            self.eval_total[i].reset()

        self.encoder.eval()
        self.classifier.eval()
        for decoder in self.decoders:
            decoder.eval()

        n_batches = int(np.ceil(self.config.data.epoch_len / 10))

        with tqdm(total=n_batches) as valid_enum, \
                torch.no_grad():
            for batch_num in range(n_batches):
                if self.config.data.short and batch_num == 10:
                    break

                dset_num = batch_num % self.config.data.n_datasets

                x, x_aug = next(self.data[dset_num].valid_iter)

                x = wrap_cuda(x)
                x_aug = wrap_cuda(x_aug)
                batch_loss = self.eval_batch(x, x_aug, dset_num)

                valid_enum.set_description(f'Test (loss: {batch_loss:.2f}) epoch {epoch}')
                valid_enum.update()

    @staticmethod
    def format_losses(meters):
        losses = [meter.summarize_epoch() for meter in meters]
        return ', '.join('{:.4f}'.format(x) for x in losses)

    def train_losses(self):
        meters = [*self.losses_recon, self.loss_d_right]
        return self.format_losses(meters)

    def eval_losses(self):
        meters = [*self.evals_recon, self.eval_d_right]
        return self.format_losses(meters)

    def train(self):
        best_eval = [float('inf') for _ in range(self.config.data.n_datasets)]

        # Begin!
        for epoch in range(self.start_epoch, self.start_epoch + self.config.env.epochs):
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            self.logger.info(f'Epoch %s - Train loss: (%s), Test loss (%s)',
                             epoch, self.train_losses(), self.eval_losses())
            for i in range(len(self.lr_managers)):
                self.lr_managers[i].step()

            for dataset_id in range(self.config.data.n_datasets):
                val_loss = self.eval_total[dataset_id].summarize_epoch()

                if val_loss < best_eval[dataset_id]:
                    self.save_model(f'bestmodel_{dataset_id}.pth', dataset_id)
                    best_eval[dataset_id] = val_loss

                if not self.config.env.save_per_epoch:
                    self.save_model(f'lastmodel_{dataset_id}.pth', dataset_id)
                else:
                    self.save_model(f'lastmodel_{epoch}_rank_{dataset_id}.pth', dataset_id)

                torch.save([self.config, epoch], '%s/args.pth' % self.expPath)

                self.logger.debug('Ended epoch')

    def save_model(self, filename, decoder_id):
        save_path = self.expPath / filename

        torch.save({'encoder_state': self.encoder.module.state_dict(),
                    'decoder_state': self.decoders[decoder_id].module.state_dict(),
                    'discriminator_state': self.classifier.module.state_dict(),
                    'model_optimizer_state': self.model_optimizers[decoder_id].state_dict(),
                    'dataset': decoder_id,
                    'd_optimizer_state': self.classifier_optimizer.state_dict()
                    },
                   save_path)

        self.logger.debug(f'Saved model to {save_path}')


def main():
    Trainer(config).train()


if __name__ == '__main__':
    main()
