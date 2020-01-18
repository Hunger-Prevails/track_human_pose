import os
import torch
import numpy as np

class Logger:

    def __init__(self, args, state):
        self.state = state if state else dict(best_auc = 0, best_pck = 0, best_root = -1, best_mean = -1, best_epoch = 0, epoch = 0)

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        self.save_path = os.path.join(args.save_path, args.model + '-' + args.suffix)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        assert args.save_record != args.test_only

        self.save_record = args.save_record


    def append(self, test_rec):

        save_path = './profile.pth'

        if os.path.exists(save_path):

            profile = torch.load(save_path)

            profile = dict([(key, profile[key] + [value]) for key, value in test_rec.items()])
        else:
            profile = dict([(key, [value]) for key, value in test_rec.items()])

        torch.save(profile, save_path)

        print '\n=> profile saved to', save_path, '\n'

    def record(self, epoch, train_rec, test_rec, model):

        if torch.typename(model).find('DataParallel') != -1:
            model = model.module

        self.state['epoch'] = epoch

        if train_rec:
            model_file = os.path.join(self.save_path, 'model_%d.pth' % epoch);

            checkpoint = dict()
            checkpoint['state'] = self.state
            checkpoint['model'] = model.state_dict()

            torch.save(checkpoint, model_file)

        if test_rec:
            score_sum = test_rec['score_auc']
            best_sum = self.state['best_auc']

            if score_sum > best_sum:
                self.state['best_epoch'] = epoch

                self.state['best_auc'] = test_rec['score_auc']
                self.state['best_pck'] = test_rec['score_pck']
                self.state['best_root'] = test_rec['root']
                self.state['best_mean'] = test_rec['mean']

                best = os.path.join(self.save_path, 'best.pth')
                torch.save({'best': epoch}, best)

        train_rec.update(test_rec)

        if self.save_record:

            save_path = os.path.join(self.save_path, 'protocol.pth')

            if os.path.exists(save_path):

                protocol = torch.load(save_path)

                protocol = dict([(key, protocol[key] + [value]) for key, value in train_rec.items()])
            else:
                protocol = dict([(key, [value]) for key, value in train_rec.items()])

            torch.save(protocol, save_path)

            print '\n=> protocol saved to', save_path, '\n'

    def final_print(self):

        message = '=> Best Epoch: %3d' % (self.state['best_epoch'])
        message += '  Root: %1.4f' % (self.state['best_root'])
        message += '  Mean: %1.4f' % (self.state['best_mean'])
        message += '  AuC: %1.4f' % (self.state['best_auc'])
        message += '  PcK: %1.4f' % (self.state['best_pck'])

        print ''
        print message
