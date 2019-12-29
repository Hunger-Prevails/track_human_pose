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
        self.decimal = args.decimal
        self.n_cudas = args.n_cudas
        self.num_joints = args.num_joints
        self.in_features = args.in_features
        self.half_acc = args.half_acc

        self.learn_rate = args.learn_rate
        self.num_epochs = args.n_epochs
        self.grad_norm = args.grad_norm
        self.grad_scaling = args.grad_scaling

        self.accept_crit = nn.__dict__[args.accept_crit + 'Loss'](reduction = 'mean')
        self.refine_crit = nn.__dict__[args.refine_crit + 'Loss'](reduction = 'mean')
        self.agnost_crit = nn.__dict__[args.agnost_crit + 'Loss'](reduction = 'mean')

        if args.n_cudas:
            self.accept_crit = self.accept_crit.cuda()
            self.refine_crit = self.refine_crit.cuda()
            self.agnost_crit = self.agnost_crit.cuda()


    def train(self, epoch, data_loader):
        self.model.train()
        self.adapt_learn_rate(epoch)

        cudevice = torch.device('cuda')
        n_batches = len(data_loader)

        accept_loss_avg = 0
        refine_loss_avg = 0
        agnost_loss_avg = 0
        total = 0

        for i, (tracklet, curr_det, verdict, true_cam, key_mask) in enumerate(data_loader):
            '''
            Args:
                tracklet: (batch, num_joints x in_features, n_frames) <float32>
                curr_det: (batch, num_joints x in_features) <float32>
                verdict: (batch, 1) <uint8>
                true_cam: (batch, num_joints, 3) <float32>
                key_mask: (batch, num_joints) <uint8>
            '''
            if self.n_cudas:
                tracklet = tracklet.half().to(cudevice) if self.half_acc else tracklet.to(cudevice)

                curr_det = curr_det.half().to(cudevice) if self.half_acc else curr_det.to(cudevice)

                verdict = verdict.to(cudevice)

                true_cam = true_cam.to(cudevice)

                key_mask = key_mask.to(cudevice)

            batch = verdict.size(0)

            accept, refine, agnost = self.model(tracklet, curr_det)

            if self.half_acc:
                accept = accept.float()
                refine = refine.float()
                agnost = agnost.float()

            accept_loss = self.accept_crit(accept, verdict.float())

            refine_mask = (verdict & key_mask).view(-1)
            agnost_mask = (~verdict & key_mask).view(-1)

            refine_loss = self.refine_crit(refine.view(-1, 3)[refine_mask], true_cam.view(-1, 3)[refine_mask])
            agnost_loss = self.agnost_crit(agnost.view(-1, 3)[agnost_mask], true_cam.view(-1, 3)[agnost_mask])

            accept_loss_avg += accept_loss.item() * batch
            refine_loss_avg += refine_loss.item() * batch
            agnost_loss_avg += agnost_loss.item() * batch

            loss = accept_loss + refine_loss + agnost_loss

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

            message = '| train Epoch[%d] [%d/%d]' % (epoch, i, n_batches)
            message += '  Accept Loss: %1.4f' % (accept_loss.item())
            message += '  Refine Loss: %1.4f' % (refine_loss.item())
            message += '  Agnost Loss: %1.4f' % (agnost_loss.item())

            print message

        accept_loss_avg /= total
        refine_loss_avg /= total
        agnost_loss_avg /= total

        message = '=> train Epoch[%d]' % (epoch)
        message += '  Accept Loss: %1.4f' % (accept_loss_avg)
        message += '  Refine Loss: %1.4f' % (refine_loss_avg)
        message += '  Agnost Loss: %1.4f' % (agnost_loss_avg)

        print ''
        print message

        return dict(accept_train_loss = accept_loss_avg, refine_train_loss = refine_loss_avg, agnost_train_loss = agnost_loss_avg)


    def test(self, epoch, test_loader):
        self.model.eval()

        cudevice = torch.device('cuda')
        n_batches = len(test_loader)

        accept_loss_avg = 0
        refine_loss_avg = 0
        agnost_loss_avg = 0
        total = 0

        cam_stats = []

        for i, (tracklet, curr_det, verdict, true_cam, key_mask) in enumerate(test_loader):
            '''
            Args:
                tracklet: (batch, num_joints x in_features, in_frames) <float32>
                curr_det: (batch, num_joints x in_features) <float32>
                verdict: (batch, 1) <uint8>
                true_cam: (batch, num_joints, 3) <float32>
                key_mask: (batch, num_joints) <uint8>
            '''
            if self.n_cudas:
                tracklet = tracklet.half().to(cudevice) if self.half_acc else tracklet.to(cudevice)

                curr_det = curr_det.half().to(cudevice) if self.half_acc else curr_det.to(cudevice)

                verdict = verdict.to(cudevice)

                true_cam = true_cam.to(cudevice)

                key_mask = key_mask.to(cudevice)

            batch = verdict.size(0)

            with torch.no_grad():

                accept, refine, agnost = self.model(tracklet, curr_det)

                if self.half_acc:
                    accept = accept.float()
                    refine = refine.float()
                    agnost = agnost.float()

                accept_loss = self.accept_crit(accept, verdict.float())

                refine_mask = (verdict & key_mask).view(-1)
                agnost_mask = (~verdict & key_mask).view(-1)

                refine_loss = self.refine_crit(refine.view(-1, 3)[refine_mask], true_cam.view(-1, 3)[refine_mask])
                agnost_loss = self.agnost_crit(agnost.view(-1, 3)[agnost_mask], true_cam.view(-1, 3)[agnost_mask])

            accept_loss_avg += accept_loss.item() * batch
            refine_loss_avg += refine_loss.item() * batch
            agnost_loss_avg += agnost_loss.item() * batch

            total += batch

            message = '| test Epoch[%d] [%d/%d]' % (epoch, i, n_batches)
            message += '  Accept Loss: %1.4f' % (accept_loss.item())
            message += '  Refine Loss: %1.4f' % (refine_loss.item())
            message += '  Agnost Loss: %1.4f' % (agnost_loss.item())

            print message

            accept = accept.cpu().numpy()
            refine = refine.cpu().numpy()
            agnost = agnost.cpu().numpy()

            true_cam = true_cam.cpu().numpy()

            verdict = verdict.cpu().numpy().astype(np.bool)
            key_mask = key_mask.cpu().numpy().astype(np.bool)

            cam_stats.append(utils.analyze(accept, refine, agnost, true_cam, verdict, key_mask, self.thresh_score, self.decimal))

        accept_loss_avg /= total
        refine_loss_avg /= total
        agnost_loss_avg /= total

        record = dict(accept_test_loss = accept_loss_avg, refine_test_loss = refine_loss_avg, agnost_test_loss = agnost_loss_avg)
        record.update(utils.parse_epoch(cam_stats))

        print ''

        message = '=> test Epoch[%d]' % (epoch)
        message += '  Accept Loss: %1.4f' % (accept_loss_avg)
        message += '  Refine Loss: %1.4f' % (refine_loss_avg)
        message += '  Agnost Loss: %1.4f' % (agnost_loss_avg)

        print message

        message = '  precision: %1.4f' % (record['precision'])
        message += '  recall: %1.4f' % (record['recall'])
        message += '  f-measure: %1.4f' % (record['f_measure'])

        print message

        message = '  [refine] mean: %1.4f' % (record['refine_mean'])
        message += '  [refine] auc: %1.4f' % (record['refine_auc'])
        message += '  [refine] pck: %1.4f' % (record['refine_pck'])

        print message

        message = '  [agnost] mean: %1.4f' % (record['agnost_mean'])
        message += '  [agnost] auc: %1.4f' % (record['agnost_auc'])
        message += '  [agnost] pck: %1.4f' % (record['agnost_pck'])

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
