import utils
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from builtins import zip as xzip
from mpl_toolkits.mplot3d import Axes3D


class Trainer:

    def __init__(self, args, model):

        self.model = model

        self.ones = torch.ones(1, 1, 3)

        self.optimizer = optim.Adam(model.parameters(), args.learn_rate, weight_decay = args.weight_decay)

        self.thresh_score = args.thresh_score
        self.n_cudas = args.n_cudas
        self.in_frames = args.in_frames
        self.occ_span = args.occ_span

        self.learn_rate = args.learn_rate
        self.num_epochs = args.n_epochs
        self.grad_norm = args.grad_norm
        self.grad_scaling = args.grad_scaling

        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean')

        if args.n_cudas:
            self.ones = self.ones.cuda()
            self.criterion = self.criterion.cuda()


    def train(self, epoch, data_loader):
        self.model.train()
        self.adapt_learn_rate(epoch)

        cudevice = torch.device('cuda')

        n_batches = len(data_loader)

        loss_avg = 0
        total = 0

        for i, (cam_track, mask, cam_gt) in enumerate(data_loader):
            '''
            Args:
                cam_track: (batch, 17 x 3, in_frames) <float32>
                mask: (batch, 1, in_frames) <float32>
                cam_gt: (batch, 17, 3) <float32>
            '''
            if self.n_cudas:
                cam_track = cam_track.to(cudevice)

                mask = mask.to(cudevice)

                cam_gt = cam_gt.to(cudevice)

            batch = mask.size(0)

            cam_spec = self.model(cam_track, mask, self.ones)  # (batch, 17, 3)

            rootrel_spec = cam_spec[:, :-1] - cam_spec[:, -1:]  # (batch, 16, 3)

            rootrel_gt = cam_gt[:, :-1] - cam_gt[:, -1:]  # (batch, 16, 3)

            loss_rootrel = self.criterion(rootrel_spec, rootrel_gt)

            root_spec = cam_spec[:, -1:]

            root_gt = cam_gt[:, -1:]  # (batch, 1, 3)

            loss_root = self.criterion(root_spec, root_gt)

            loss = loss_root + loss_rootrel

            loss_avg += loss.item() * batch

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()

            total += batch

            print '| train Epoch[%d] [%d/%d]  Rootrel Loss: %1.4f  Root Loss: %1.4f' % (epoch, i, n_batches, loss_rootrel.item(), loss_root.item())

        loss_avg /= total

        print ''
        print '=> train Epoch[%d]  Loss: %1.4f' % (epoch, loss_avg)
        print ''

        return dict(train_loss = loss_avg)


    def test(self, epoch, test_loader):
        self.model.eval()

        cudevice = torch.device('cuda')

        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        cam_stats = []

        for i, (cam_track, mask, cam_gt) in enumerate(test_loader):
            '''
            Args:
                cam_track: (batch, 17 x 3, in_frames) <float32>
                mask: (batch, 1, in_frames) <float32>
                cam_gt: (batch, 17, 3) <float32>
            '''
            if self.n_cudas:
                cam_track = cam_track.to(cudevice)

                mask = mask.to(cudevice)

                cam_gt = cam_gt.to(cudevice)

            batch = mask.size(0)

            with torch.no_grad():

                cam_spec = self.model(cam_track, mask, self.ones)  # (batch, 17, 3)

                rootrel_spec = cam_spec[:, :-1] - cam_spec[:, -1:]  # (batch, 16, 3)

                rootrel_gt = cam_gt[:, :-1] - cam_gt[:, -1:]  # (batch, 16, 3)

                loss_rootrel = self.criterion(rootrel_spec, rootrel_gt)

                root_spec = cam_spec[:, -1:]

                root_gt = cam_gt[:, -1:]  # (batch, 1, 3)

                loss_root = self.criterion(root_spec, root_gt)

                loss = loss_root + loss_rootrel

            loss_avg += loss.item() * batch

            total += batch

            print '| test Epoch[%d] [%d/%d] Loss: %1.4f' % (epoch, i, n_batches, loss.item())

            cam_spec = cam_spec.cpu().numpy()

            cam_gt = cam_gt.cpu().numpy()
            '''
            cam_track = cam_track.cpu().numpy().reshape(batch, -1, 3, self.in_frames)

            middle = (self.in_frames - 1) / 2

            offset = self.occ_span + 1

            cam_spec = np.mean(cam_track[:, :, :, [middle - offset, middle + offset]], axis = -1)

            from plot import show_cam

            import matplotlib.pyplot as plt

            for ii in xrange(batch):

                if ii % 10 != 0:
                    continue

                plt.figure(figsize = (16, 12))
                ax = plt.subplot(1, 1, 1, projection = '3d')

                ii_last = cam_track[ii, :, :, -1].transpose() * 100.0
                ii_cam_gt = cam_gt[ii].transpose() * 100.0
                ii_cam_spec = cam_spec[ii].transpose() * 100.0

                show_cam(np.expand_dims(ii_last, axis = 0), ax, color = np.array([0.0, 0.0, 1.0]))
                show_cam(np.expand_dims(ii_cam_gt, axis = 0), ax, color = np.array([0.0, 1.0, 0.0]))
                show_cam(np.expand_dims(ii_cam_spec, axis = 0), ax, color = np.array([1.0, 0.0, 0.0]))

                plt.show()
            '''
            cam_stats.append(utils.analyze(cam_spec, cam_gt, self.thresh_score))

        loss_avg /= total

        record = dict(test_loss = loss_avg)
        record.update(utils.parse_epoch(cam_stats))

        message = '=> test Epoch[%d]' % (epoch)
        message += '  Loss: %1.4f' % (loss_avg)
        message += '  Root: %1.4f' % (record['root'])
        message += '  Mean: %1.4f' % (record['mean'])
        message += '  AuC: %1.4f' % (record['score_auc'])
        message += '  PcK: %1.4f' % (record['score_pck'])

        print ''
        print message

        return record


    def adapt_learn_rate(self, epoch):
        if epoch - 1 < self.num_epochs * 0.6:
            learn_rate = self.learn_rate
        elif epoch - 1 < self.num_epochs * 0.9:
            learn_rate = self.learn_rate * 0.2
        else:
            learn_rate = self.learn_rate * 0.04

        for group in self.optimizer.param_groups:
            group['lr'] = learn_rate
