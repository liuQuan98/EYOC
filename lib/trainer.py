# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import os.path as osp
import gc
import logging
import numpy as np
import json
from easydict import EasyDict as edict
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash

from pytorch3d.ops.knn import knn_points
import pytorch3d
from scripts.SC2_PCR.SC2_PCR import Matcher

import MinkowskiEngine as ME


class AlignmentTrainer:
    def __init__(
        self,
        config,
        data_loader,
        val_data_loader=None,
    ):
        num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

        # Model initialization
        Model = load_model(config.model)
        model = Model(num_feats,
                      config.model_n_out,
                      bn_momentum=config.bn_momentum,
                      normalize_feature=config.normalize_feature,
                      conv1_kernel_size=config.conv1_kernel_size,
                      D=3)

        if config.weights:
            print(f"Loading weight from {config.weights}")
            checkpoint = torch.load(config.weights)
            model.load_state_dict(checkpoint['state_dict'])

        logging.info(model)

        self.config = config
        self.model = model
        self.max_epoch = config.max_epoch
        self.save_freq = config.save_freq_epoch
        self.val_max_iter = config.val_max_iter
        self.val_epoch_freq = config.val_epoch_freq

        self.best_val_metric = config.best_val_metric
        self.best_val_epoch = -np.inf
        self.best_val = -np.inf

        if config.use_gpu and not torch.cuda.is_available():
            logging.warning(
                'Warning: There\'s no CUDA support on this machine, '
                'training is performed on CPU.')
            raise ValueError('GPU not available, but cuda flag set')

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = getattr(optim, config.optimizer)(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, config.exp_gamma)

        self.start_epoch = 1
        self.checkpoint_dir = config.out_dir

        ensure_dir(self.checkpoint_dir)
        json.dump(config,
                  open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4,
                  sort_keys=False)

        self.iter_size = config.iter_size
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        self.test_valid = True if self.val_data_loader is not None else False
        self.log_step = int(np.sqrt(self.config.batch_size))
        self.model = self.model.to(self.device)
        self.writer = SummaryWriter(logdir=config.out_dir)

        if config.resume is not None:
            if osp.isfile(config.resume):
                logging.info("=> loading checkpoint '{}'".format(config.resume))
                state = torch.load(config.resume)
                if not config.finetune_restart:
                    self.start_epoch = state['epoch']
                    self.scheduler.load_state_dict(state['scheduler'])
                    self.optimizer.load_state_dict(state['optimizer'])
                    if 'best_val' in state.keys():
                        self.best_val = state['best_val']
                        self.best_val_epoch = state['best_val_epoch']
                        self.best_val_metric = state['best_val_metric']
                else:
                    logging.info("=> Finetuning, will only load model weights.")
                model.load_state_dict(state['state_dict'])
            else:
                raise ValueError(
                    f"=> no checkpoint found at '{config.resume}'")

    def train(self):
        """
        Full training logic
        """
        # Baseline random feature performance
        if self.test_valid:
            pass
            # with torch.no_grad():
            #     self.epoch = 0
            #     val_dict = self._valid_epoch()

            # for k, v in val_dict.items():
            #     self.writer.add_scalar(f'val/{k}', v, 0)

        for epoch in range(self.start_epoch, self.max_epoch + 1):
            lr = self.scheduler.get_lr()
            logging.info(f" Epoch: {epoch}, LR: {lr}")
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)
            self.scheduler.step()

            if self.test_valid and epoch % self.val_epoch_freq == 0:
                with torch.no_grad():
                    val_dict = self._valid_epoch()

                for k, v in val_dict.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                if self.best_val < val_dict[self.best_val_metric]:
                    logging.info(
                        f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
                    )
                    self.best_val = val_dict[self.best_val_metric]
                    self.best_val_epoch = epoch
                    self._save_checkpoint(epoch, 'best_val_checkpoint')
                else:
                    logging.info(
                        f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
                    )

    def _save_checkpoint(self, epoch, filename='checkpoint'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_val': self.best_val,
            'best_val_epoch': self.best_val_epoch,
            'best_val_metric': self.best_val_metric
        }
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        logging.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)


class ContrastiveLossTrainer(AlignmentTrainer):
    def __init__(
        self,
        config,
        data_loader,
        val_data_loader=None,
    ):
        if val_data_loader is not None:
            assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
        AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
        self.neg_thresh = config.neg_thresh
        self.pos_thresh = config.pos_thresh
        self.neg_weight = config.neg_weight

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.t() + T

    def generate_rand_negative_pairs(self,
                                     positive_pairs,
                                     hash_seed,
                                     N0,
                                     N1,
                                     N_neg=0):
        """
    Generate random negative pairs
    """
        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)
        if N_neg < 1:
            N_neg = positive_pairs.shape[0] * 2
        pos_keys = _hash(positive_pairs, hash_seed)

        neg_pairs = np.floor(
            np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
                np.int64)
        neg_keys = _hash(neg_pairs, hash_seed)
        mask = np.isin(neg_keys, pos_keys, assume_unique=False)
        return neg_pairs[np.logical_not(mask)]

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0

        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()

        iter_size = self.iter_size
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)

        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

        # Main training
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                # Caffe iter size
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                # pairs consist of (xyz1 index, xyz0 index)
                sinput0 = ME.SparseTensor(
                    input_dict['sinput0_F'].to(self.device),
                    coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(
                    input_dict['sinput1_F'].to(self.device),
                    coordinates=input_dict['sinput1_C'].to(self.device))
                F1 = self.model(sinput1).F

                N0, N1 = len(sinput0), len(sinput1)

                pos_pairs = input_dict['correspondences']
                neg_pairs = self.generate_rand_negative_pairs(
                    pos_pairs, max(N0, N1), N0, N1)
                pos_pairs = pos_pairs.long().to(self.device)
                neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

                neg0 = F0.index_select(0, neg_pairs[:, 0])
                neg1 = F1.index_select(0, neg_pairs[:, 1])
                pos0 = F0.index_select(0, pos_pairs[:, 0])
                pos1 = F1.index_select(0, pos_pairs[:, 1])

                # Positive loss
                pos_loss = (pos0 - pos1).pow(2).sum(1)

                # Negative loss
                neg_loss = F.relu(self.neg_thresh - (
                    (neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

                pos_loss_mean = pos_loss.mean() / iter_size
                neg_loss_mean = neg_loss.mean() / iter_size

                # Weighted loss
                loss = pos_loss_mean + self.neg_weight * neg_loss_mean
                loss.backward(
                )  # To accumulate gradient, zero gradients only at the begining of iter_size
                batch_loss += loss.item()
                batch_pos_loss += pos_loss_mean.item()
                batch_neg_loss += neg_loss_mean.item()

            self.optimizer.step()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            # Print logs
            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss,
                                       start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) // iter_size, batch_loss,
                            batch_pos_loss, batch_neg_loss) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}"
                    .format(data_meter.avg, total_timer.avg -
                            data_meter.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()

    def _valid_epoch(self):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
        ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        for batch_idx in range(tot_num_data):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            # pairs consist of (xyz1 index, xyz0 index)
            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device),
                                      coordinates=input_dict['sinput0_C'].to(
                                          self.device))
            F0 = self.model(sinput0).F

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device),
                                      coordinates=input_dict['sinput1_C'].to(
                                          self.device))
            F1 = self.model(sinput1).F
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'][0], input_dict[
                'pcd1'][0], input_dict['T_gt'][0]
            xyz0_corr, xyz1_corr = self.find_corr(xyz0,
                                                  xyz1,
                                                  F0,
                                                  F1,
                                                  subsample_size=5000)
            T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos(
                (np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(
                xyz0_corr,
                xyz1_corr,
                T_gt,
                thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

            if batch_idx % 100 == 0 and batch_idx > 0:
                logging.info(' '.join([
                    f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
                    f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
                    f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
                    f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
                ]))
                data_timer.reset()

        logging.info(' '.join([
            f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        return {
            "loss": loss_meter.avg,
            "rre": rre_meter.avg,
            "rte": rte_meter.avg,
            'feat_match_ratio': feat_match_ratio.avg,
            'hit_ratio': hit_ratio_meter.avg
        }

    def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
        subsample = len(F0) > subsample_size
        if subsample_size > 0 and subsample:
            N0 = min(len(F0), subsample_size)
            N1 = min(len(F1), subsample_size)
            inds0 = np.random.choice(len(F0), N0, replace=False)
            inds1 = np.random.choice(len(F1), N1, replace=False)
            F0, F1 = F0[inds0], F1[inds1]

        # Compute the nn
        nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
        if subsample_size > 0 and subsample:
            return xyz0[inds0], xyz1[inds1[nn_inds]]
        else:
            return xyz0, xyz1[nn_inds]

    def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
        xyz0 = self.apply_transform(xyz0, T_gth)
        dist = torch.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
        return (dist < thresh).float().mean().item()


class HardestContrastiveLossTrainer(ContrastiveLossTrainer):
    def contrastive_hardest_negative_loss(self,
                                          F0,
                                          F1,
                                          positive_pairs,
                                          num_pos=5192,
                                          num_hn_samples=2048,
                                          thresh=None):
        """
        Generate negative pairs
        """
        # print(F0.shape)
        # print(F1.shape)
        # print(positive_pairs.shape)
        # raise ValueError
        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        if N_pos_pairs > num_pos:
            pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        # Find negatives for all F1[positive_pairs[:, 1]]
        subF0, subF1 = F0[sel0], F1[sel1]

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        D01 = pdist(posF0, subF1, dist_type='L2')
        D10 = pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = _hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                sinput0 = ME.SparseTensor(
                    input_dict['sinput0_F'].to(self.device),
                    coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(
                    input_dict['sinput1_F'].to(self.device),
                    coordinates=input_dict['sinput1_C'].to(self.device))

                F1 = self.model(sinput1).F

                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=self.config.num_pos_per_batch *
                    self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch *
                    self.config.batch_size)

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss,
                                       start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) // iter_size, batch_loss,
                            batch_pos_loss, batch_neg_loss) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}"
                    .format(data_meter.avg, total_timer.avg -
                            data_meter.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()


class TripletLossTrainer(ContrastiveLossTrainer):
    def triplet_loss(self,
                     F0,
                     F1,
                     positive_pairs,
                     num_pos=1024,
                     num_hn_samples=None,
                     num_rand_triplet=1024):
        """
    Generate negative pairs
    """
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = _hash(positive_pairs, hash_seed)
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        # Random triplets
        rand_inds = np.random.choice(num_pos_pairs,
                                     min(num_pos_pairs, num_rand_triplet),
                                     replace=False)
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1,
                                     min(N1, num_rand_triplet),
                                     replace=False)

        # Remove positives from negatives
        rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(
            np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] -
                                    F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] -
                                    F1[negatives]).pow(2).sum(1) + 1e-7)

        loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()

        return loss, pos_dist.mean(), rand_neg_dist.mean()

    def _train_epoch(self, epoch):
        config = self.config

        gc.collect()
        self.model.train()

        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        pos_dist_meter, neg_dist_meter = AverageMeter(), AverageMeter()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_loss = 0
            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                # pairs consist of (xyz1 index, xyz0 index)
                sinput0 = ME.SparseTensor(
                    input_dict['sinput0_F'].to(self.device),
                    coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(
                    input_dict['sinput1_F'].to(self.device),
                    coordinates=input_dict['sinput1_C'].to(self.device))
                F1 = self.model(sinput1).F

                pos_pairs = input_dict['correspondences']
                loss, pos_dist, neg_dist = self.triplet_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=config.triplet_num_pos * config.batch_size,
                    num_hn_samples=config.triplet_num_hn * config.batch_size,
                    num_rand_triplet=config.triplet_num_rand *
                    config.batch_size)
                loss /= iter_size
                loss.backward()
                batch_loss += loss.item()
                pos_dist_meter.update(pos_dist)
                neg_dist_meter.update(neg_dist)

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss,
                                       start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) // iter_size, batch_loss,
                            pos_dist_meter.avg, neg_dist_meter.avg) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}"
                    .format(data_meter.avg, total_timer.avg -
                            data_meter.avg, total_timer.avg))
                pos_dist_meter.reset()
                neg_dist_meter.reset()
                data_meter.reset()
                total_timer.reset()


class HardestTripletLossTrainer(TripletLossTrainer):
    def triplet_loss(self,
                     F0,
                     F1,
                     positive_pairs,
                     num_pos=1024,
                     num_hn_samples=512,
                     num_rand_triplet=1024):
        """
    Generate negative pairs
    """
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        # Find negatives for all F1[positive_pairs[:, 1]]
        subF0, subF1 = F0[sel0], F1[sel1]

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        D01 = pdist(posF0, subF1, dist_type='L2')
        D10 = pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = _hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        # Random triplets
        rand_inds = np.random.choice(num_pos_pairs,
                                     min(num_pos_pairs, num_rand_triplet),
                                     replace=False)
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1,
                                     min(N1, num_rand_triplet),
                                     replace=False)

        # Remove positives from negatives
        rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(
            np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] -
                                    F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] -
                                    F1[negatives]).pow(2).sum(1) + 1e-7)

        loss = F.relu(
            torch.cat([
                rand_pos_dist + self.neg_thresh - rand_neg_dist,
                pos_dist[mask0] + self.neg_thresh - D01min[mask0],
                pos_dist[mask1] + self.neg_thresh - D10min[mask1]
            ])).mean()

        return loss, pos_dist.mean(), (D01min.mean() +
                                       D10min.mean()).item() / 2


class CorrespondenceExtensionTrainer(ContrastiveLossTrainer):
    def __init__(
        self,
        config,
        data_loader,
        val_data_loader=None,
    ):
        num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

        self.config = config

        # Model initialization
        Model = load_model(config.model)
        model = Model(num_feats,
                      config.model_n_out,
                      bn_momentum=config.bn_momentum,
                      normalize_feature=config.normalize_feature,
                      conv1_kernel_size=config.conv1_kernel_size,
                      D=3)

        if config.weights:
            print(f"Loading weight from {config.weights}")
            checkpoint = torch.load(config.weights)
            model.load_state_dict(checkpoint['state_dict'])

        logging.info("Training model structure:")
        logging.info(model)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Labeler model initialization
        if config.labeler_dir:
            labeler_config = json.load(
                open(config.labeler_dir + '/config.json', 'r'))
            labeler_config = edict(labeler_config)
            LabelerModel = load_model(labeler_config.model)
            labeler_model = LabelerModel(
                num_feats,
                labeler_config.model_n_out,
                bn_momentum=labeler_config.bn_momentum,
                normalize_feature=labeler_config.normalize_feature,
                conv1_kernel_size=labeler_config.conv1_kernel_size,
                D=3)
            if hasattr(config, "labeler_weight") and config.labeler_weight != '':
                labeler_model_path = config.labeler_weight
            else:
                labeler_model_path = config.labeler_dir + "/checkpoint.pth"
            labeler_checkpoint = torch.load(labeler_model_path)
            labeler_model.load_state_dict(labeler_checkpoint['state_dict'])

            self.labeler_max_dist = labeler_config.pair_max_dist
            self.labeler = labeler_model.to(self.device)
            logging.info("Labeler model structure:")
            logging.info(self.labeler)
        else:
            self.labeler = None
            # logging.info("Labeler is disabled: using base mode training.")

        # SC2_PCR initialization
        self.use_sc2pcr = config.use_SC2_PCR

        if self.use_sc2pcr:
            config_sc2pcr = json.load(open('scripts/SC2_PCR/config_json/config_KITTI.json', 'r'))
            config_sc2pcr = edict(config_sc2pcr)
            for key, item in config_sc2pcr.items():
                config[key] = item
            self.matcher = Matcher(inlier_threshold=config.inlier_threshold,
                                   num_node=config.num_node,
                                   use_mutual=config.use_mutual,
                                   d_thre=config.d_thre,
                                   num_iterations=config.num_iterations,
                                   ratio=config.ratio,
                                   nms_radius=config.nms_radius,
                                   max_points=config.max_points,
                                   k1=config.k1,
                                   k2=config.k2)

        self.neg_thresh = config.neg_thresh
        self.pos_thresh = config.pos_thresh
        self.neg_weight = config.neg_weight

        self.config = config
        self.model = model
        self.max_epoch = config.max_epoch
        self.save_freq = config.save_freq_epoch
        self.val_max_iter = config.val_max_iter
        self.val_epoch_freq = config.val_epoch_freq

        self.best_val_metric = config.best_val_metric
        self.best_val_epoch = -np.inf
        self.best_val = -np.inf

        if config.use_gpu and not torch.cuda.is_available():
            logging.warning(
                'Warning: There\'s no CUDA support on this machine, '
                'training is performed on CPU.')
            raise ValueError('GPU not available, but cuda flag set')

        self.optimizer = getattr(optim, config.optimizer)(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, config.exp_gamma)

        self.start_epoch = 1
        self.checkpoint_dir = config.out_dir

        ensure_dir(self.checkpoint_dir)
        json.dump(config,
                  open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4,
                  sort_keys=False)

        self.iter_size = config.iter_size
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        self.test_valid = True if self.val_data_loader is not None else False
        # self.test_valid = False
        self.log_step = int(np.sqrt(self.config.batch_size))
        self.model = self.model.to(self.device)
        self.writer = SummaryWriter(logdir=config.out_dir)

        # debug parameters! delete them during publish.
        self.plot_similarity = False
        self.record_sim_dataset = False

        if config.resume is not None:
            if osp.isfile(config.resume):
                logging.info("=> loading checkpoint '{}'".format(
                    config.resume))
                state = torch.load(config.resume)
                self.start_epoch = state['epoch']
                model.load_state_dict(state['state_dict'])
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])

                if 'best_val' in state.keys():
                    self.best_val = state['best_val']
                    self.best_val_epoch = state['best_val_epoch']
                    self.best_val_metric = state['best_val_metric']
            else:
                raise ValueError(
                    f"=> no checkpoint found at '{config.resume}'")

    def contrastive_hardest_negative_loss(self,
                                          F0,
                                          F1,
                                          positive_pairs,
                                          num_pos=5192,
                                          num_hn_samples=2048,
                                          thresh=None):
        """
        Generate negative pairs
        """
        # print(F0.shape)
        # print(F1.shape)
        # print(positive_pairs.shape)
        # raise ValueError
        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        if N_pos_pairs > num_pos:
            pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        # Find negatives for all F1[positive_pairs[:, 1]]
        subF0, subF1 = F0[sel0], F1[sel1]

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        D01 = pdist(posF0, subF1, dist_type='L2')
        D10 = pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = _hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2
    
    def calculate_ratio_test(self, dists):
        """
        Calculate weights for matches based on the ratio between kNN distances.
        Input:
            (N, P, 2) Cosine Distance between point and nearest 2 neighbors
        Output:
            (N, P, 1) Weight based on ratio; higher is more unique match
        """
        # Convert points so that 0 means perfect similarity and clamp to avoid numerical
        # instability
        dists = (1 - dists).clamp(min=1e-9)

        # Ratio -- close to 0 is completely unique; 1 is same feature
        ratio = dists[:, :, 0:1] / dists[:, :, 1:2]
        # Weight -- Convert so that higher is more unique
        weight = 1 - ratio

        return weight
    
    def get_topk_matches(self, dists, idx, num_corres: int):
        num_corres = min(num_corres, dists.shape[1])
        dist, idx_source = torch.topk(dists, k=num_corres, dim=1)
        idx_target = idx.gather(1, idx_source)
        return idx_source, idx_target, dist
    
    def get_dataset_name(self):
        fullname = type(self.data_loader.dataset).__name__.lower()
        for name in ["kitti", "nuscenes", "waymo"]:
            if name in fullname:
                return name
        return "notfound"
    
    def match_and_filter_corr(self, C_batch_0, F_batch_0, C_batch_1, F_batch_1, radius=20, feature_filter="Lowe", spatial_filter='Spherical', frame_distance=None):
        """
            Match correspondences through k-nearest-neighbor, filter them using lowe's ratio, then apply spherical filtering
            which eliminates correspondences whose either end is too close to LiDAR
        Input:
            - C_batch_0:    [bs, ni, 3]
            - F_batch_0:    [bs, ni, d]
            - C_batch_1:    [bs, nj, 3]
            - F_batch_1:    [bs, nj, d]
                - note that ni and nj might change, while d is the fixed feature dimension
            - radius:       threshold to cut nearby correspondences around LiDARs if filter=='Spherical'
            - filter:       Specifies the filtering method, must be one in ["Spherical", "Similarity"]
            - frame_distance:   [bs], bears the frame interval of each input point cloud pair if filter=='Similarity'

        Output:
            - matches:      [N, 2], the collated correspondence indexes where N=sum(ni)
            - uncollated_matches: [bs, ni, d], the uncollated correspondence indexes
        """
        for i in range(len(C_batch_0)):
            C_batch_0[i] = C_batch_0[i].to(F_batch_0[0].device)
            C_batch_1[i] = C_batch_1[i].to(F_batch_0[0].device)

        num_corres = 5000
        P1 = pytorch3d.structures.Pointclouds(C_batch_0, features=F_batch_0)
        P2 = pytorch3d.structures.Pointclouds(C_batch_1, features=F_batch_1)

        P1_F = P1.features_padded()
        P2_F = P2.features_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()

        assert feature_filter in ["None", "Lowe"]
        assert spatial_filter in ["Spherical", "Similarity", "None"]

        K = 1 if feature_filter =="None" else 2

        dists_1, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=K)
        dists_2, idx_2, _ = knn_points(P2_F, P1_F, P2_N, P1_N, K=K)

        idx_1 = idx_1[:, :, 0:1]
        idx_2 = idx_2[:, :, 0:1]

        if feature_filter =="Lowe":
            cosine_1 = 1 - 0.5 * dists_1
            cosine_2 = 1 - 0.5 * dists_2

            weights_1 = self.calculate_ratio_test(cosine_1)
            weights_2 = self.calculate_ratio_test(cosine_2)
        else:
            weights_1 = dists_1[:, :, 0:1]
            weights_2 = dists_2[:, :, 0:1]

        # Get topK matches in both directions
        n_corres_1 = min(num_corres, P1_N.min())
        n_corres_2 = min(num_corres, P2_N.min())
        
        if n_corres_1 < num_corres or n_corres_2 < num_corres:
            pass
            # print(f"Min corresponds is {n_corres_1} and {n_corres_2}")

        m12_idx1, m12_idx2, m12_dist = self.get_topk_matches(weights_1, idx_1, n_corres_1)
        m21_idx2, m21_idx1, m21_dist = self.get_topk_matches(weights_2, idx_2, n_corres_2)
        # cosine_1 = cosine_1[:, :, 0:1].gather(1, m12_idx1)
        # cosine_2 = cosine_2[:, :, 0:1].gather(1, m21_idx2)

        # concatenate into correspondances. Shape: (B, ni, 1), (B, ni, 1)
        matches_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        matches_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)

        # recover the index biases for each batched point cloud pair
        length_1_padded = [0] + [len(feats) for feats in F_batch_0]
        length_2_padded = [0] + [len(feats) for feats in F_batch_1]
        bias_1 = torch.cumsum(torch.Tensor(length_1_padded[:-1]), 0)
        bias_2 = torch.cumsum(torch.Tensor(length_2_padded[:-1]), 0)

        # re-collate the dispatched matching-indices to shape (N, 2)
        matches_biased_1 = [match + bias for (match, bias) in zip(matches_idx1, bias_1)]
        matches_biased_2 = [match + bias for (match, bias) in zip(matches_idx2, bias_2)]
        match_1 = torch.cat(matches_biased_1, dim=0)
        match_2 = torch.cat(matches_biased_2, dim=0)

        matches = torch.cat([match_1, match_2], dim=1)
        # Spherical filtering that eliminates correspondences whose either end is too close to LiDAR.
        # Emperically, Lowe filter must precede spherical filter in order to get the best result.
        # Note that 'matches' is not spherically filtered, since it is not used in the current version.
        uncollated_matches = []
        for i in range(len(C_batch_0)):
            if spatial_filter == 'None':
                mask = torch.ones_like(torch.norm(C_batch_0[i][matches_idx1[i].squeeze(1)], dim=1)).bool()
            if spatial_filter == 'Spherical':
                mask_1 = torch.norm(C_batch_0[i][matches_idx1[i].squeeze(1)], dim=1) > radius
                mask_2 = torch.norm(C_batch_1[i][matches_idx2[i].squeeze(1)], dim=1) > radius
                mask = mask_1 & mask_2
            elif spatial_filter == 'Similarity':
                d0 = torch.norm(C_batch_0[i][matches_idx1[i].squeeze(1)], dim=1)
                d1 = torch.norm(C_batch_1[i][matches_idx2[i].squeeze(1)], dim=1)

                # convert central distance to (min(d0,d1), |d1-d0|) coordinate for table lookup
                d1_tmp = torch.abs(d0-d1)
                d0 = torch.min(torch.vstack([d0, d1]), dim=0).values
                d1 = d1_tmp

                # get the mask based on similarity filtering of filed similarity records
                if not hasattr(self, 'dist_sim_map'):
                    dataset = self.config.pretraining_dataset
                    maps = np.load(f"config/dist_sim_plot/{dataset}_distSimPlot.npz", allow_pickle=True)["res"].tolist()
                    self.dist_sim_map = {}
                    for indice in range(6):
                        self.dist_sim_map[indice] = torch.tensor(maps[indice]).to(d0.device)

                # determine the coordinate with rounding in similarity lookup tables
                frame_index = min(max(0, frame_distance[i]//5), 5)
                xlim, ylim = self.dist_sim_map[frame_index].shape
                frame_to_ygrid_size = {0: 1, 1: 1.5, 2: 2, 3: 2.5, 4: 2.5, 5: 2.5}
                gridsize = [5,frame_to_ygrid_size[frame_index]]
                d0 = (d0/gridsize[0]).long()
                d1 = (d1/gridsize[1]).long()

                d0[torch.where(d0 < 0)] = 0  # round out-of-bound input to the limited map
                d1[torch.where(d1 < 0)] = 0
                d0[torch.where(d0 >= ylim)] = ylim-1
                d1[torch.where(d1 >= xlim)] = xlim-1
                mask = self.dist_sim_map[frame_index][d1, d0] > self.config.similarity_thresh

            uncollated_matches.append(torch.cat([matches_idx1[i][mask], matches_idx2[i][mask]], dim=1))

        return matches.cpu().detach(), uncollated_matches
    
    def corr_through_registration(self, input_dict, uncollated_pairs):
        T_ransac = []
        C_batch_0, C_batch_1, fitnesses, success_est = [], [], [], []

        # ToDo: Fix SC2-PCR so that this loop can be batched and parallelized
        for i in range(len(uncollated_pairs)):
            src_keypts_corr = input_dict["pcd0"][i][uncollated_pairs[i][:,0]][None,:,:]
            tgt_keypts_corr = input_dict["pcd1"][i][uncollated_pairs[i][:,1]][None,:,:]
            result, fitness = self.matcher.SC2_PCR(src_keypts_corr, tgt_keypts_corr)
            result = result[0]
            fitnesses.append(fitness)
            T_ransac.append(result.cpu().float().numpy())
            C_batch_0.append((input_dict["pcd0"][i] @ result[:3,:3].T + result[:3,3].T).to(self.device))
            C_batch_1.append(input_dict["pcd1"][i].to(self.device))

        mutual=False

        P1 = pytorch3d.structures.Pointclouds(C_batch_0)
        P2 = pytorch3d.structures.Pointclouds(C_batch_1)

        P1_F = P1.points_padded()
        P2_F = P2.points_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()

        if mutual:
            pass
            # _, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=1)
            # idx_1 = idx_1[:, :, 0]
            # _, idx_2, _ = knn_points(P2_F, P1_F, P2_N, P1_N, K=1)
            # idx_2 = idx_2[:, :, 0]

            # bias_1 = torch.cumsum(torch.Tensor([0] + P1_N.tolist()), 0).long()
            # bias_2 = torch.cumsum(torch.Tensor([0] + P2_N.tolist()), 0).long()

            # correspondences = []
            # for i, (p1_length, p2_length) in enumerate(zip(P1_N, P2_N)):
            #     if not success_est[i] == 0:
            #         pos_sel_1 = torch.randperm(p1_length)[:min(p1_length, 5000)].to(self.device)
            #         pos_sel_2 = torch.randperm(p2_length)[:min(p2_length, 5000)].to(self.device)
            #         idx_sel_1 = idx_1[i][pos_sel_1]
            #         idx_sel_2 = idx_2[i][pos_sel_2]
            #         corr_1 = torch.cat([pos_sel_1, idx_sel_2], dim=0) + bias_1[i]
            #         corr_2 = torch.cat([idx_sel_1, pos_sel_2], dim=0) + bias_2[i]
            #         correspondences.append(torch.cat([corr_1.unsqueeze(1), corr_2.unsqueeze(1)], dim=1))
        else:
            _, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=1)
            idx_1 = idx_1[:, :, 0]
            bias_1 = torch.cumsum(torch.Tensor([0] + P1_N.tolist()), 0).long()
            bias_2 = torch.cumsum(torch.Tensor([0] + P2_N.tolist()), 0).long()

            correspondences = []
            uncollated_corr = []
            for i, (p1_length, p2_length) in enumerate(zip(P1_N, P2_N)):
                if not success_est[i] == 0:
                    pos_sel_1 = torch.randperm(p1_length)[:min(p1_length, 5000)].to(self.device)

                    pose = torch.Tensor(T_ransac[i]).to(self.device)
                    src_keypts_corr = input_dict["pcd0"][i][pos_sel_1.long()] @ pose[:3,:3].T + pose[:3,3].T
                    tgt_keypts_corr = input_dict["pcd1"][i][idx_1[i][pos_sel_1].long()]
                    within_range_mask = (torch.norm(src_keypts_corr - tgt_keypts_corr, dim=1) < 2) # keep all correspondences within 2m error
                    # the bound here is intentionally set loose due to tolerate possible pose errors

                    # mask the out-of-range corrs
                    pos_sel_1 = pos_sel_1[within_range_mask]
                    pos_sel_2 = idx_1[i][pos_sel_1]
                    uncollated_corr.append(torch.cat([pos_sel_1.unsqueeze(1), pos_sel_2.unsqueeze(1)], dim=1))
                    
                    pos_sel_1 += bias_1[i]
                    pos_sel_2 += bias_2[i]
                    correspondences.append(torch.cat([pos_sel_1.unsqueeze(1), pos_sel_2.unsqueeze(1)], dim=1))

        correspondences = torch.cat(correspondences, dim=0)
        return T_ransac, correspondences, success_est, fitnesses, uncollated_corr
    
    def _get_dist_similarity_plot(self, C0, C1, 
                                  F0: torch.Tensor, F1: torch.Tensor, corr): 
        corr = corr[torch.randperm(len(corr))[:5000]]
        corr = corr.long()

        d0_corr = torch.norm(torch.cat(C0, dim=0)[corr[:,0]], dim=1).to('cpu')
        d1_corr = torch.norm(torch.cat(C1, dim=0)[corr[:,1]], dim=1).to('cpu')

        f0_corr = F0[corr[:,0]]
        f1_corr = F1[corr[:,1]]

        dot = torch.sum(f0_corr * f1_corr, dim=1)
        cosine = dot / (torch.norm(f0_corr, dim=1) * torch.norm(f1_corr, dim=1))
        return torch.vstack([d0_corr, d1_corr, cosine.to('cpu')]).T.contiguous().tolist()

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size

        dist_sim_log = []
        trans_est_log = []
        trans_gt_log = []
        fitnesses_log = []

        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        labeler_hit_meter = AverageMeter()

        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                F_in_0, C_in_0 = input_dict['sinput0_F'].to(
                    self.device), input_dict['sinput0_C'].to(self.device)
                sinput0 = ME.SparseTensor(F_in_0, coordinates=C_in_0)
                if not self.plot_similarity:
                    F0 = self.model(sinput0).F

                F_in_1, C_in_1 = input_dict['sinput1_F'].to(
                    self.device), input_dict['sinput1_C'].to(self.device)
                sinput1 = ME.SparseTensor(F_in_1, coordinates=C_in_1)
                if not self.plot_similarity:
                    F1 = self.model(sinput1).F

                if self.labeler:
                    # Extension mode training, use labeler to obtain pseudo correspondence label
                    with torch.no_grad():
                        encoded_0 = self.labeler(sinput0)
                        encoded_1 = self.labeler(sinput1)
                        _, batch_enc_feats_0 = \
                            encoded_0.decomposed_coordinates_and_features
                        _, batch_enc_feats_1 = \
                            encoded_1.decomposed_coordinates_and_features
                        
                        # We use filtering in both metric space and euclidean space to create cleaner "correspondence labels".
                        # This is the key to the labeler self-improving upon longer-distance point cloud pairs.
                        pos_pair_tmp, uncollated_pairs = self.match_and_filter_corr(input_dict["pcd0"], batch_enc_feats_0,
                                                                                    input_dict["pcd1"], batch_enc_feats_1,
                                                                                    radius=20, 
                                                                                    feature_filter=self.config.feature_filter,
                                                                                    spatial_filter=self.config.spatial_filter,
                                                                                    frame_distance=input_dict['frame_distance'])
                        
                        # The correspondences above are still noisy and not suitable for direct training of the student.
                        # Instead, we apply SC2-PCR to select the inliers and perform a registration.
                        # The estimated pose is highly likely correct and can be used to re-calculate nearest-neighbor correspondences.
                        if self.config.use_sc2_filtering:
                            try:
                                T_ransac, pos_pairs, _, _, uncollated_pairs = self.corr_through_registration(input_dict, uncollated_pairs)
                            except Exception as e:
                                print(e)    # sometimes SC2-PCR will fail in various ways due to empty correspondence. We just catch and ignore them.
                                continue
                            pos_pairs = pos_pairs.cpu()
                        else:
                            pos_pairs = pos_pair_tmp

                        # obtain the inlier ratio for labeler during training
                        for batch_idx in range(len(input_dict["pcd0"])):
                            xyz0_indice = uncollated_pairs[batch_idx][:,0]
                            xyz1_indice = uncollated_pairs[batch_idx][:,1]
                            xyz0_corr = input_dict["pcd0"][batch_idx][xyz0_indice]
                            xyz1_corr = input_dict["pcd1"][batch_idx][xyz1_indice]
                            hit_ratio = self.evaluate_hit_ratio(
                                xyz0_corr,
                                xyz1_corr,
                                input_dict["T_gt"][batch_idx].to(self.device),
                                thresh=self.config.hit_ratio_thresh)
                            labeler_hit_meter.update(hit_ratio)

                        # visualization code
                        # for t_pred, t_gt in zip(T_ransac, input_dict["T_gt"]):
                        #     t_pred = torch.from_numpy(t_pred)
                        #     rte = np.linalg.norm(t_pred[:3, 3] - t_gt[:3, 3])
                        #     rre = np.arccos((np.trace(t_pred[:3, :3].T @ t_gt[:3, :3]) - 1) / 2) *180 / np.pi
                        #     print(f"{rte=}, {rre=}")

                        # if self.plot_similarity:
                        #     trans_gt_log  += [item.cpu().tolist() for item in input_dict['T_gt']]
                        #     trans_est_log += [item.tolist() for item in T_ransac]

                        # trans_gt = np.array([item.cpu().tolist() for item in input_dict['T_gt']])
                        # trans_pred = np.array([item.tolist() for item in T_ransac])
                        # len_0 = np.cumsum([len(item) for item in input_dict["pcd0"]])
                        # len_1 = np.cumsum([len(item) for item in input_dict["pcd1"]])
                        # len_matches = np.cumsum([len(item) for item in uncollated_pairs])
                        # batch_enc_coords_0 = np.concatenate([coords.cpu().numpy() for coords in input_dict["pcd0"]], axis=0)
                        # batch_enc_coords_1 = np.concatenate([coords.cpu().numpy() for coords in input_dict["pcd1"]], axis=0)
                        # matched_pairs = np.concatenate([matchs.cpu().numpy() for matchs in uncollated_pairs], axis=0)
                        # np.savez("point_clouds", pcd_1=batch_enc_coords_0, pcd_2=batch_enc_coords_1, 
                        #          trans=trans_gt, trans_pred=trans_pred, match=matched_pairs,
                        #          len1=len_0, len2=len_1, len_matches=len_matches)
                        # print("pcd saved!!!!")
                        # raise ValueError
                else:
                    # Base mode training, use identity pose to obtain pseudo correspondence label for two consecutive frames
                    pos_pairs = input_dict['correspondences']

                if self.plot_similarity:
                    dist_sim_log += self._get_dist_similarity_plot(input_dict["pcd0"], input_dict["pcd1"], 
                                                                   encoded_0.F, encoded_1.F, input_dict['correspondences'])
                    continue

                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=self.config.num_pos_per_batch *
                    self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch *
                    self.config.batch_size)

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss,
                                       start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) // iter_size, batch_loss,
                            batch_pos_loss, batch_neg_loss) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}"
                    .format(data_meter.avg, total_timer.avg -
                            data_meter.avg, total_timer.avg)
                    + "\tLabeler IR: {:.3f}".format(labeler_hit_meter.avg)
                )
                data_meter.reset()
                total_timer.reset()

        if self.plot_similarity:
            np.savez(f"{self.get_dataset_name()}_dist_sim_plot_{1}_{self.data_loader.dataset.MAX_DIST}", log=dist_sim_log)
            success_meter = AverageMeter()
            for t_pred, t_gt in zip(trans_est_log, trans_gt_log):
                t_pred = np.array(t_pred)
                t_gt = np.array(t_gt)
                rte = np.linalg.norm(t_pred[:3, 3] - t_gt[:3, 3])
                rre = np.arccos((np.trace(t_pred[:3, :3].T @ t_gt[:3, :3]) - 1) / 2) *180 / np.pi
                
                rte_thresh = 2
                rre_thresh = 5
                if rte < rte_thresh and not np.isnan(rre) and rre < rre_thresh:
                    success_meter.update(1)
                else:
                    success_meter.update(0)
            print(f"{rte_thresh=}, {rre_thresh=}")
            print(f"Labeler trained with max_pair_dist={self.labeler_max_dist} labels data {self.data_loader.dataset.MAX_DIST}m apart at {100*success_meter.avg:.2f}% RR.")
            raise ValueError
            

class ContinuousCorrExtensionTrainer(CorrespondenceExtensionTrainer):
    def search_nearest_neighbor(self, input_dict):
        T_gt = []
        C_batch_0, C_batch_1 = [], []

        # ToDo: Fix SC2-PCR so that this loop can be batched and parallelized
        for i in range(len(input_dict["pcd0"])):
            result = input_dict["T_gt"][i]
            T_gt.append(result)
            C_batch_0.append((input_dict["pcd0"][i] @ result[:3,:3].T + result[:3,3].T).to(self.device))
            C_batch_1.append(input_dict["pcd1"][i].to(self.device))

        P1 = pytorch3d.structures.Pointclouds(C_batch_0)
        P2 = pytorch3d.structures.Pointclouds(C_batch_1)

        P1_F = P1.points_padded()
        P2_F = P2.points_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()

        _, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=1)
        idx_1 = idx_1[:, :, 0]
        bias_1 = torch.cumsum(torch.Tensor([0] + P1_N.tolist()), 0).long()
        bias_2 = torch.cumsum(torch.Tensor([0] + P2_N.tolist()), 0).long()

        correspondences = []
        uncollated_corr = []
        for i, (p1_length, p2_length) in enumerate(zip(P1_N, P2_N)):
            pos_sel_1 = torch.randperm(p1_length)[:min(p1_length, 5000)].to(self.device)

            src_keypts_corr = C_batch_0[i][pos_sel_1.long()]
            tgt_keypts_corr = C_batch_1[i][idx_1[i][pos_sel_1].long()]
            within_range_mask = (torch.norm(src_keypts_corr - tgt_keypts_corr, dim=1) < 2) # keep all correspondences within 2m error
            # the bound here is intentionally set loose due to tolerate possible pose errors

            # mask the out-of-range corrs
            pos_sel_1 = pos_sel_1[within_range_mask]
            pos_sel_2 = idx_1[i][pos_sel_1]
            uncollated_corr.append(torch.cat([pos_sel_1.unsqueeze(1), pos_sel_2.unsqueeze(1)], dim=1))
            
            pos_sel_1 += bias_1[i]
            pos_sel_2 += bias_2[i]
            correspondences.append(torch.cat([pos_sel_1.unsqueeze(1), pos_sel_2.unsqueeze(1)], dim=1))
        correspondences = torch.cat(correspondences, dim=0)
        return correspondences.cpu().detach()

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0

        # If the epoch reaches a bunch of predefined thresholds,
        # the dataset is extended to contain longer-distance pairs.
        curr_distance = self.data_loader.dataset.update_extension_distance(epoch)

        # save the intermediate models when we are at some important intermediate positions, for plotting only
        # if curr_distance and curr_distance in [2,6,11,16,21,26]:
        #     self._save_checkpoint(epoch-1, filename=f"{curr_distance-1}_checkpoint")

        self.epoch = epoch
        base_mode_flag = self.data_loader.dataset.is_base_dataset()

        if not base_mode_flag or self.config.skip_initialization:
            if not self.labeler:
                # initialize the labeler to the current model (not that specified in LABELER_DIR config!)
                Model = load_model(self.config.model)
                self.labeler = Model(1,
                            self.config.model_n_out,
                            bn_momentum=self.config.bn_momentum,
                            normalize_feature=self.config.normalize_feature,
                            conv1_kernel_size=self.config.conv1_kernel_size,
                            D=3)
                self.labeler = self.labeler.to(self.device)
                self.labeler.load_state_dict(self.model.state_dict())
                self.num_updates = 1
            elif self.config.sync_strategy == "Sync":
                # fix the labeler at the beginning of an epoch (fully synchronized)
                self.labeler.load_state_dict(self.model.state_dict())
            elif self.config.sync_strategy == "EMA":
                decay = self.config.ema_decay
                debias = (1 - decay ** self.num_updates)
                for (labeler_param, model_param) in zip(self.labeler.state_dict().values(), self.model.state_dict().values()):
                    labeler_param.copy_((decay * labeler_param + (1-decay) * model_param) / debias)
                self.num_updates += 1
            else:
                raise NotImplementedError
        else:
            logging.info("Labeler is disabled: using base mode training.")

        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size

        dist_sim_log = []
        trans_est_log = []
        trans_gt_log = []
        fitnesses_log = []

        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        nn_timer, feat_timer, label_timer, back_timer = Timer(), Timer(), Timer(), Timer()
        labeler_hit_meter = AverageMeter()

        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                # torch.cuda.synchronize()
                data_timer.tic()
                input_dict = data_loader_iter.next()

                # torch.cuda.synchronize()
                data_time += data_timer.toc(average=False)
                # nn_timer.tic()

                # # use pytorch3d to accelerate FCGF
                # if len(input_dict["correspondences"]) // self.batch_size < 10:  # primitive way to discriminate whether the batch is filled with dummy correspondences
                #     input_dict["correspondences"] = self.search_nearest_neighbor(input_dict)

                # torch.cuda.synchronize()
                # nn_timer.toc()
                # feat_timer.tic()

                F_in_0, C_in_0 = input_dict['sinput0_F'].to(
                    self.device), input_dict['sinput0_C'].to(self.device)
                sinput0 = ME.SparseTensor(F_in_0, coordinates=C_in_0)
                if not self.plot_similarity:
                    F0 = self.model(sinput0).F

                F_in_1, C_in_1 = input_dict['sinput1_F'].to(
                    self.device), input_dict['sinput1_C'].to(self.device)
                sinput1 = ME.SparseTensor(F_in_1, coordinates=C_in_1)
                if not self.plot_similarity:
                    F1 = self.model(sinput1).F

                # torch.cuda.synchronize()
                # feat_timer.toc()
                # label_timer.tic()

                if not base_mode_flag or self.config.skip_initialization:
                    # Extension mode training, use labeler to obtain pseudo correspondence label
                    with torch.no_grad():
                        encoded_0 = self.labeler(sinput0)
                        encoded_1 = self.labeler(sinput1)
                        _, batch_enc_feats_0 = \
                            encoded_0.decomposed_coordinates_and_features
                        _, batch_enc_feats_1 = \
                            encoded_1.decomposed_coordinates_and_features
                        
                        # We use filtering in both metric space and euclidean space to create cleaner "correspondence labels".
                        # This is the key to the labeler self-improving upon longer-distance point cloud pairs.
                        pos_pair_tmp, uncollated_pairs = self.match_and_filter_corr(input_dict["pcd0"], batch_enc_feats_0,
                                                                                    input_dict["pcd1"], batch_enc_feats_1,
                                                                                    radius=self.config.filter_radius, 
                                                                                    feature_filter=self.config.feature_filter,
                                                                                    spatial_filter=self.config.spatial_filter,
                                                                                    frame_distance=input_dict['frame_distance'])
                        
                        # The correspondences above are still noisy and not suitable for direct training of the student.
                        # Instead, we apply SC2-PCR to select the inliers and perform a registration.
                        # The estimated pose is highly likely correct and can be used to re-calculate nearest-neighbor correspondences.
                        if self.config.use_sc2_filtering:
                            try:
                                T_ransac, pos_pairs, _, fitnesses, uncollated_pairs = self.corr_through_registration(input_dict, uncollated_pairs)
                                if self.record_sim_dataset:
                                    trans_gt_log  += [item.cpu().tolist() for item in input_dict['T_gt']]
                                    trans_est_log += [item.tolist() for item in T_ransac]
                                    fitnesses_log += fitnesses                            
                            except Exception as e:
                                print(e)    # sometimes SC2-PCR will fail in various ways due to empty correspondence. We just catch and ignore them.
                                continue
                            pos_pairs = pos_pairs.cpu()
                        else:
                            pos_pairs = pos_pair_tmp

                        # obtain the inlier ratio for labeler during training
                        for batch_idx in range(len(input_dict["pcd0"])):
                            xyz0_indice = uncollated_pairs[batch_idx][:,0]
                            xyz1_indice = uncollated_pairs[batch_idx][:,1]
                            xyz0_corr = input_dict["pcd0"][batch_idx][xyz0_indice]
                            xyz1_corr = input_dict["pcd1"][batch_idx][xyz1_indice]
                            hit_ratio = self.evaluate_hit_ratio(
                                xyz0_corr,
                                xyz1_corr,
                                input_dict["T_gt"][batch_idx].to(self.device),
                                thresh=self.config.hit_ratio_thresh)
                            labeler_hit_meter.update(hit_ratio)

                        # # visualization code
                        # for t_pred, t_gt in zip(T_ransac, input_dict["T_gt"]):
                        #     t_pred = torch.from_numpy(t_pred)
                        #     rte = np.linalg.norm(t_pred[:3, 3] - t_gt[:3, 3])
                        #     rre = np.arccos((np.trace(t_pred[:3, :3].T @ t_gt[:3, :3]) - 1) / 2) *180 / np.pi
                        #     print(f"{rte=}, {rre=}")

                        # trans_gt = np.array([item.cpu().tolist() for item in input_dict['T_gt']])
                        # trans_pred = np.array([item.tolist() for item in T_ransac])
                        # len_0 = np.cumsum([len(item) for item in input_dict["pcd0"]])
                        # len_1 = np.cumsum([len(item) for item in input_dict["pcd1"]])
                        # len_matches = np.cumsum([len(item) for item in uncollated_pairs])
                        # batch_enc_coords_0 = np.concatenate([coords.cpu().numpy() for coords in input_dict["pcd0"]], axis=0)
                        # batch_enc_coords_1 = np.concatenate([coords.cpu().numpy() for coords in input_dict["pcd1"]], axis=0)
                        # matched_pairs = np.concatenate([matchs.cpu().numpy() for matchs in uncollated_pairs], axis=0)
                        # np.savez("point_clouds_eyoc", pcd_1=batch_enc_coords_0, pcd_2=batch_enc_coords_1, 
                        #          trans=trans_gt, trans_pred=trans_pred, match=matched_pairs,
                        #          len1=len_0, len2=len_1, len_matches=len_matches)
                        # print("pcd saved!!!!")
                        # raise ValueError
                else:
                    # Base mode training, use identity pose to obtain pseudo correspondence label for two consecutive frames
                    pos_pairs = input_dict['correspondences']

                # torch.cuda.synchronize()
                # label_timer.toc()
                # back_timer.tic()

                # if self.plot_similarity:
                #     dist_sim_log += self._get_dist_similarity_plot(input_dict["pcd0"], input_dict["pcd1"], 
                #                                                    encoded_0.F, encoded_1.F, input_dict['correspondences'])
                #     continue

                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=self.config.num_pos_per_batch *
                    self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch *
                    self.config.batch_size)

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()

                # torch.cuda.synchronize()
                # back_timer.toc()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss,
                                       start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) // iter_size, batch_loss,
                            batch_pos_loss, batch_neg_loss) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}"
                    .format(data_meter.avg, total_timer.avg -
                            data_meter.avg, total_timer.avg)
                    + "\tLabeler IR: {:.3f}".format(labeler_hit_meter.avg)
                    # + "\tNN time: {:.3f}".format(nn_timer.avg)
                    # + "\tFeat time: {:.3f}".format(feat_timer.avg)
                    # + "\tLabel time: {:.3f}".format(label_timer.avg)
                    # + "\tBP time: {:.3f}".format(back_timer.avg)
                )

                data_meter.reset()
                total_timer.reset()

        if self.plot_similarity:
            # np.savez(f"{self.get_dataset_name()}_dist_sim_plot_{self.labeler_max_dist}_{self.data_loader.dataset.MAX_DIST}", log=dist_sim_log)
            success_meter = AverageMeter()
            for t_pred, t_gt in zip(trans_est_log, trans_gt_log):
                t_pred = np.array(t_pred)
                t_gt = np.array(t_gt)
                rte = np.linalg.norm(t_pred[:3, 3] - t_gt[:3, 3])
                rre = np.arccos((np.trace(t_pred[:3, :3].T @ t_gt[:3, :3]) - 1) / 2) *180 / np.pi
                
                rte_thresh = 4
                rre_thresh = 10
                if rte < rte_thresh and not np.isnan(rre) and rre < rre_thresh:
                    success_meter.update(1)
                else:
                    success_meter.update(0)
            print(f"{rte_thresh=}, {rre_thresh=}")
            print(f"Labeler trained with max_pair_dist={self.labeler_max_dist} labels data {self.data_loader.dataset.MAX_DIST}m apart at {100*success_meter.avg:.2f}% RR.")
            raise ValueError
        
        if self.record_sim_dataset and epoch % 30 == 1:
            res = {i:{"trans_gt":trans_gt_log[i], "trans_est":trans_est_log[i], "feats":fitnesses_log[i]} 
                   for i in range(len(trans_gt_log))}
            np.savez(f"{type(self.data_loader.dataset).__name__}_epoch{epoch}", res=res)
            print(f"Saved dataset records for epoch {epoch}.")
        
    def _valid_epoch(self):
        # Change the network to evaluation mode
        self.model.eval()

        # # If the epoch reaches a bunch of predefined thresholds,
        # # the dataset is extended to contain longer-distance pairs.
        # change_flag = self.val_data_loader.dataset.update_extension_distance(self.epoch)
        # # reset the evaluation metrics upon val dataloader update.
        # if change_flag:
        #     self.best_val_epoch = -np.inf
        #     self.best_val = -np.inf
        self.val_data_loader.dataset.reset_seed(0)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
        ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        for batch_idx in range(tot_num_data):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            # pairs consist of (xyz1 index, xyz0 index)
            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device),
                                      coordinates=input_dict['sinput0_C'].to(
                                          self.device))
            F0 = self.model(sinput0).F

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device),
                                      coordinates=input_dict['sinput1_C'].to(
                                          self.device))
            F1 = self.model(sinput1).F
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'][0], input_dict[
                'pcd1'][0], input_dict['T_gt'][0]
            xyz0_corr, xyz1_corr = self.find_corr(xyz0,
                                                  xyz1,
                                                  F0,
                                                  F1,
                                                  subsample_size=5000)
            T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos(
                (np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(
                xyz0_corr,
                xyz1_corr,
                T_gt,
                thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

            if batch_idx % 100 == 0 and batch_idx > 0:
                logging.info(' '.join([
                    f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
                    f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
                    f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
                    f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
                ]))
                data_timer.reset()

        logging.info(' '.join([
            f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        return {
            "loss": loss_meter.avg,
            "rre": rre_meter.avg,
            "rte": rte_meter.avg,
            'feat_match_ratio': feat_match_ratio.avg,
            'hit_ratio': hit_ratio_meter.avg
        }


class ContinuousHardestContrastiveTrainer(HardestContrastiveLossTrainer):
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()

        # The extension step
        # If the epoch reaches a bunch of predefined thresholds,
        # the dataset is extended to contain longer-distance pairs.
        self.data_loader.dataset.update_extension_distance(epoch)
        self.epoch = epoch

        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                sinput0 = ME.SparseTensor(
                    input_dict['sinput0_F'].to(self.device),
                    coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(
                    input_dict['sinput1_F'].to(self.device),
                    coordinates=input_dict['sinput1_C'].to(self.device))

                F1 = self.model(sinput1).F

                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=self.config.num_pos_per_batch *
                    self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch *
                    self.config.batch_size)

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss,
                                       start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) // iter_size, batch_loss,
                            batch_pos_loss, batch_neg_loss) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}"
                    .format(data_meter.avg, total_timer.avg -
                            data_meter.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()

    def _valid_epoch(self):
        # Change the network to evaluation mode
        self.model.eval()

        # If the epoch reaches a bunch of predefined thresholds,
        # the dataset is extended to contain longer-distance pairs.
        change_flag = self.val_data_loader.dataset.update_extension_distance(self.epoch)
        # reset the evaluation metrics upon val dataloader update.
        if change_flag:
            self.best_val_epoch = -np.inf
            self.best_val = -np.inf
        self.val_data_loader.dataset.reset_seed(0)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
        ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        for batch_idx in range(tot_num_data):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            # pairs consist of (xyz1 index, xyz0 index)
            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device),
                                      coordinates=input_dict['sinput0_C'].to(
                                          self.device))
            F0 = self.model(sinput0).F

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device),
                                      coordinates=input_dict['sinput1_C'].to(
                                          self.device))
            F1 = self.model(sinput1).F
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'][0], input_dict[
                'pcd1'][0], input_dict['T_gt'][0]
            xyz0_corr, xyz1_corr = self.find_corr(xyz0,
                                                  xyz1,
                                                  F0,
                                                  F1,
                                                  subsample_size=5000)
            T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos(
                (np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(
                xyz0_corr,
                xyz1_corr,
                T_gt,
                thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

            if batch_idx % 100 == 0 and batch_idx > 0:
                logging.info(' '.join([
                    f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
                    f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
                    f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
                    f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
                ]))
                data_timer.reset()

        logging.info(' '.join([
            f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        return {
            "loss": loss_meter.avg,
            "rre": rre_meter.avg,
            "rte": rte_meter.avg,
            'feat_match_ratio': feat_match_ratio.avg,
            'hit_ratio': hit_ratio_meter.avg
        }
    

class ImprovedHardestContrastiveLossTrainer(HardestContrastiveLossTrainer):
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                sinput0 = ME.SparseTensor(
                    input_dict['sinput0_F'].to(self.device),
                    coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(
                    input_dict['sinput1_F'].to(self.device),
                    coordinates=input_dict['sinput1_C'].to(self.device))

                F1 = self.model(sinput1).F

                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=self.config.num_pos_per_batch *
                    self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch *
                    self.config.batch_size)

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss,
                                       start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss,
                                       start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) // iter_size, batch_loss,
                            batch_pos_loss, batch_neg_loss) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}"
                    .format(data_meter.avg, total_timer.avg -
                            data_meter.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()

