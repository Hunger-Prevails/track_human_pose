import utils
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from builtins import zip as xzip


class Trainer:

    def __init__(self, args, model):

        assert args.half_acc <= args.n_cudas

        self.model = model

        self.list_params = list(model.parameters())

        if args.half_acc:
            self.copy_params = [param.clone().detach() for param in self.list_params]
            self.model = self.model.half()

            for param in self.copy_params:
                param.requires_grad = True
                param.grad = param.data.new_zeros(param.size())

            self.optimizer = optim.Adam(self.copy_params, args.learn_rate, weight_decay = args.weight_decay)
        else:
            self.optimizer = optim.Adam(self.list_params, args.learn_rate, weight_decay = args.weight_decay)

        self.thresh_score = args.thresh_score
        self.n_cudas = args.n_cudas
        self.n_joints = args.n_joints
        self.half_acc = args.half_acc

        self.learn_rate = args.learn_rate
        self.num_epochs = args.n_epochs
        self.grad_norm = args.grad_norm
        self.grad_scaling = args.grad_scaling

        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean')

        if args.n_cudas:
            self.criterion = self.criterion.cuda()


    def train(self, epoch, data_loader):
        self.model.train()
        self.adapt_learn_rate(epoch)

        cudevice = torch.device('cuda')

        n_batches = len(data_loader)

        loss_avg = 0
        total = 0

        for i, (tracklet, mask, cam_gt) in enumerate(data_loader):
            '''
            Args:
                tracklet: (batch, n_joints x in_features, in_frames) <float32>
                mask: (batch, 1, in_frames) <float32>
                cam_gt: (batch, n_joints x 3)
            '''
            if self.n_cudas:
                tracklet = tracklet.half().to(cudevice) if self.half_acc else tracklet.to(cudevice)

                mask = mask.half().to(cudevice) if self.half_acc else mask.to(cudevice)

                cam_gt = cam_gt.half().to(cudevice) if self.half_acc else cam_gt.to(cudevice)

            batch = tracklet.size(0)

            cam_spec = self.model(tracklet, mask)

            if self.half_acc:
                cam_spec = cam_spec.float()

            loss = self.criterion(cam_gt, cam_spec)

            loss_avg += loss.item() * batch

            if self.half_acc:
                loss *= self.grad_scaling

                for h_param in self.list_params:

                    if h_param.grad is None:
                        continue

                    h_param.grad.detach_()
                    h_param.grad.zero_()

                loss.backward()

                self.optimizer.zero_grad()

                do_update = True

                for c_param, h_param in xzip(self.copy_params, self.list_params):

                    if h_param.grad is None:
                        continue

                    if torch.any(torch.isinf(h_param.grad)):
                        do_update = False
                        print 'update step skipped'
                        break

                    c_param.grad.copy_(h_param.grad)
                    c_param.grad /= self.grad_scaling

                if do_update:
                    nn.utils.clip_grad_norm_(self.copy_params, self.grad_norm)

                    self.optimizer.step()

                    for c_param, h_param in xzip(self.copy_params, self.list_params):
                        h_param.data.copy_(c_param.data)

            else:
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
                self.optimizer.step()

            total += batch

            print '| train Epoch[%d] [%d/%d] Loss: %1.4f' % (epoch, i, n_batches, loss.item())

        loss_avg /= total

        print ''
        print '=> train Epoch[%d]  Loss: %1.4f' % (epoch, loss_avg)

        return dict(train_loss = loss_avg)


    def test(self, epoch, test_loader):
        self.model.eval()

        cudevice = torch.device('cuda')

        n_batches = len(test_loader)

        loss_avg = 0
        total = 0

        cam_stats = []

        for i, (tracklet, mask, cam_gt) in enumerate(data_loader):
            '''
            Args:
                tracklet: (batch, n_joints x in_features, in_frames) <float32>
                mask: (batch, in_frames) <float32>
                cam_gt: (batch, n_joints x 3)
            '''
            if self.n_cudas:
                tracklet = tracklet.half().to(cudevice) if self.half_acc else tracklet.to(cudevice)

                mask = mask.half().to(cudevice) if self.half_acc else mask.to(cudevice)

                cam_gt = cam_gt.half().to(cudevice) if self.half_acc else cam_gt.to(cudevice)

            batch = tracklet.size(0)

            cam_spec = self.model(tracklet, mask)

            if self.half_acc:
                cam_spec = cam_spec.float()

            loss = self.criterion(cam_gt, cam_spec)

            loss_avg += loss.item() * batch

            total += batch

            print '| test Epoch[%d] [%d/%d] Loss: %1.4f' % (epoch, i, n_batches, loss.item())

            cam_spec = cam_spec.cpu.numpy()
            cam_gt = cam_gt.cpu().numpy()

            cam_stats.append(utils.analyze(cam_spec, cam_gt, self.n_joints, self.thresh_score))

        loss_avg /= total

        print ''
        print '=> train Epoch[%d]  Loss: %1.4f' % (epoch, loss_avg)

        record = dict(test_loss = loss_avg)
        record.update(utils.parse_epoch(cam_stats))

        message = '=> test Epoch[%d]' % (epoch)
        message += '  Loss: %1.4f' % (loss_avg)
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
