import argparse

parser = argparse.ArgumentParser(description='list of train/test options')

# bool options
parser.add_argument('-half_acc', action='store_true', help='whether to use half precision for speed-up')
parser.add_argument('-decimal', action='store_true', help='whether to compute loss using decimeters instead')
parser.add_argument('-test_only', action='store_true', help='only performs test')
parser.add_argument('-resume', action='store_true', help='whether to continue from a previous checkpoint')
parser.add_argument('-balance', action='store_true', help='whether to balance sampling between positive and negative samples')
parser.add_argument('-save_record', action='store_true', help='Path to save train record')

# required options
parser.add_argument('-model', required=True, help='Backbone architecture')
parser.add_argument('-suffix', required=True, help='Model suffix')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-accept_crit', required=True, help='Loss formulation for accept branch')
parser.add_argument('-refine_crit', required=True, help='Loss formulation for refine branch')
parser.add_argument('-agnost_crit', required=True, help='Loss formulation for agnost branch')

# integer options
parser.add_argument('-in_features', default=6, type=int, help='number of features for each joint')
parser.add_argument('-in_frames', default=31, type=int, help='length of tracklet as input of the network')
parser.add_argument('-num_joints', default=19, type=int, help='number of joints in the dataset')
parser.add_argument('-batch_size', default=256, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-n_cudas', default=1, type=int, help='Number of cuda devices available')
parser.add_argument('-workers', default=1, type=int, help='Number of data-loading workers')
parser.add_argument('-channels', default=1024, type=int, help='Number of convolution channels')
parser.add_argument('-inflation', default=2, type=int, help='inflation rate of dilation among convolutions')
parser.add_argument('-n_blocks', default=3, type=int, help='Number of blocks with skip connection')
parser.add_argument('-kernel', default=3, type=int, help='universal kernel size of the convolutions')
parser.add_argument('-base_index', default=2, type=int, help='index of base joint')
parser.add_argument('-n_prime', default=7, type=int, help='number of joints to take into consideration for a positive association')
parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-stride', default=10, type=int, help='stride with which the samling window slide through time')

# float options
parser.add_argument('-thresh_rel', default=10.0, type=float, help='dist threshold on root-relative cam coord for a positive association')
parser.add_argument('-thresh_cam', default=20.0, type=float, help='dist threshold on cam coord for a positive association')
parser.add_argument('-thresh_score', default=15.0, type=float, help='threshold for score measurement')
parser.add_argument('-dropout', default=0.25, type=float, help='dropout rate')
parser.add_argument('-learn_rate', default=1e-3, type=float, help='Base learning rate for train')
parser.add_argument('-weight_decay', default=4e-5, type=float, help='Weight decay for training')
parser.add_argument('-grad_norm', default=5.0, type=float, help='norm for gradient clip')
parser.add_argument('-grad_scaling', default=32.0, type=float, help='magnitude of loss scaling when performing float16 computation')

# str options
parser.add_argument('-data_name', help='name of dataset')
parser.add_argument('-data_root', help='root path to dataset')
parser.add_argument('-model_path', help='path to benchmark model')

args = parser.parse_args()
