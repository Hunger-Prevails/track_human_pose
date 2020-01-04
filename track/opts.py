import argparse

parser = argparse.ArgumentParser(description='list of train/test options')

# bool options
parser.add_argument('-half_acc', action='store_true', help='whether to use half precision for speed-up')
parser.add_argument('-test_only', action='store_true', help='only performs test')
parser.add_argument('-resume', action='store_true', help='whether to continue from a previous checkpoint')
parser.add_argument('-save_record', action='store_true', help='Path to save train record')

# required options
parser.add_argument('-model', required=True, help='Backbone architecture')
parser.add_argument('-suffix', required=True, help='Model suffix')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-criterion', required=True, help='Loss formulation')
parser.add_argument('-data_name', required=True, help='name of dataset')
parser.add_argument('-data_root', required=True, help='root path to dataset')

# integer options
parser.add_argument('-in_features', default=3, type=int, help='number of features for each joint')
parser.add_argument('-in_frames', default=63, type=int, help='length of tracklet as input of the network')
parser.add_argument('-n_joints', default=17, type=int, help='number of joints in the dataset')
parser.add_argument('-batch_size', default=256, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-n_cudas', default=1, type=int, help='Number of cuda devices available')
parser.add_argument('-workers', default=1, type=int, help='Number of data-loading workers')
parser.add_argument('-channels', default=1024, type=int, help='Number of convolution channels')
parser.add_argument('-inflation', default=2, type=int, help='inflation rate of dilation among convolutions')
parser.add_argument('-n_blocks', default=4, type=int, help='Number of blocks with skip connection')
parser.add_argument('-n_prime', default=7, type=int, help='number of joints to take into consideration for a positive association')
parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-stride', default=5, type=int, help='stride by which the samling window slide through time')

# float options
parser.add_argument('-sigma', default=0.05, type=float, help='sigma for normal sampling on keypoint jitter')
parser.add_argument('-sigma_root_xy', default=0.1, type=float, help='sigma for normal sampling on root-xy jitter')
parser.add_argument('-sigma_root_zz', default=0.2, type=float, help='sigma for normal sampling on root-zz jitter')
parser.add_argument('-beta', default=5.0, type=float, help='reciprocal of lamba for exponential sampling on occlusion duration')
parser.add_argument('-thresh_rel', default=10.0, type=float, help='dist threshold on root-relative cam coord for a positive association')
parser.add_argument('-thresh_cam', default=20.0, type=float, help='dist threshold on cam coord for a positive association')
parser.add_argument('-thresh_score', default=15.0, type=float, help='threshold for score measurement')
parser.add_argument('-dropout', default=0.25, type=float, help='dropout rate')
parser.add_argument('-learn_rate', default=5e-5, type=float, help='Base learning rate for train')
parser.add_argument('-weight_decay', default=4e-5, type=float, help='Weight decay for training')
parser.add_argument('-grad_norm', default=5.0, type=float, help='norm for gradient clip')
parser.add_argument('-grad_scaling', default=32.0, type=float, help='magnitude of loss scaling when performing float16 computation')

# str options
parser.add_argument('-model_path', help='path to benchmark model')

args = parser.parse_args()
