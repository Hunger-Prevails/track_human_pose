import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import args
from datasets import get_data_loader
from log import Logger
from train import Trainer

import tracknet


def create_model(args):

    model = getattr(tracknet, args.model + 'Net')(args)
    state = None

    if args.test_only:
        save_path = os.path.join(args.save_path, args.model + '-' + args.suffix)

        print '=> Loading checkpoint from ' + os.path.join(save_path, 'best.pth')
        assert os.path.exists(save_path), '[!] Checkpoint ' + save_path + ' does not exist'

        best = torch.load(os.path.join(save_path, 'best.pth'))
        best = best['best'];
        
        checkpoint = os.path.join(save_path, 'model_%d.pth' % best)
        checkpoint = torch.load(checkpoint)['model']
        
        model.load_state_dict(checkpoint)

    if args.resume:
        print '=> Loading checkpoint from ' + args.model_path
        checkpoint = torch.load(args.model_path)
        
        model.load_state_dict(checkpoint['model'])

    if args.n_cudas:
        cudnn.benchmark = True
        model = model.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()

    return model, state


def main():
    model, state = create_model(args)
    print '=> Model and criterion are ready'

    if not args.test_only:
        data_loader = get_data_loader(args, 'train')

    test_loader = get_data_loader(args, 'test')
    print '=> Dataloaders are ready'

    logger = Logger(args, state)
    print '=> Logger is ready'

    trainer = Trainer(args, model)
    print '=> Trainer is ready'

    if args.test_only:
        print '=> Start testing'

        test_rec = trainer.test(0, test_loader)

        logger.append(test_rec)
    else:
        print '=> Start training'

        start_epoch = logger.state['epoch'] + 1
        
        for epoch in xrange(start_epoch, args.n_epochs + 1):
            train_rec = trainer.train(epoch, data_loader)
            test_rec = trainer.test(epoch, test_loader)

            logger.record(epoch, train_rec, test_rec, model) 

        logger.final_print()


if __name__ == '__main__':
    main()
