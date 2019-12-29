import os
import torch
import numpy as np

class Logger:

    def __init__(self, args, state):
        self.state = state if state else dict(f_measure = 0, refine_auc = 0, agnost_auc = 0, best_epoch = 0, epoch = 0)

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        self.save_path = os.path.join(args.save_path, args.model + '-' + args.suffix)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        assert args.save_record != args.test_only

        self.save_record = args.save_record
        self.train_record = torch.load(os.path.join(self.save_path, 'train_record.pth')) if args.resume else None


    def record(self, epoch, train_recs, test_recs, model):

        if torch.typename(model).find('DataParallel') != -1:
            model = model.module

        self.state['epoch'] = epoch

        if train_recs:
            model_file = os.path.join(self.save_path, 'model_%d.pth' % epoch);

            checkpoint = dict()
            checkpoint['state'] = self.state
            checkpoint['model'] = model.state_dict()

            torch.save(checkpoint, model_file)

        if test_recs:
            score_sum = test_recs['f_measure'] + test_recs['refine_auc'] + test_recs['agnost_auc']
            best_sum = self.state['f_measure'] + self.state['refine_auc'] + self.state['agnost_auc']

            if score_sum > best_sum:
                self.state['best_epoch'] = epoch

                self.state['f_measure'] = test_recs['f_measure']
                self.state['refine_auc'] = test_recs['refine_auc']
                self.state['agnost_auc'] = test_recs['agnost_auc']

                best = os.path.join(self.save_path, 'best.pth')
                torch.save({'best': epoch}, best)

        train_recs.update(test_recs)

        if self.save_record:

            if self.train_record:
                keys = [key for key in train_recs]

                records = [self.train_record[key] + [train_recs[key]] for key in train_recs]

                self.train_record = dict(zip(keys, records))
            else:
                keys = [key for key in train_recs]

                records = [[train_recs[key]] for key in train_recs]

                self.train_record = dict(zip(keys, records))

            torch.save(self.train_record, os.path.join(self.save_path, 'train_record.pth'))

            print '\n=> train record saved to', os.path.join(self.save_path, 'train_record.pth')
            print ''


    def final_print(self):
        message = '\n=> Best:  epoch: %3d' % (self.state['best_epoch'])

        message += '  f_measure: %6.3f' % self.state['f_measure']
        message += '  refine_auc: %6.3f' % self.state['refine_auc']
        message += '  agnost_auc: %6.3f' % self.state['agnost_auc']

        print message
